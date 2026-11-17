/*
PPO Clipped Surrogate Loss
advantages, log_pi, log_pi_old: [B, S] float32. For every position:
  r    = exp(log_pi - log_pi_old)
  rc   = clamp(r, 1 - eps, 1 + eps)
  surr = min(r * A, rc * A)
output[0] = -mean(surr) over all B*S positions.

Constraints
 - B ≤ 256, S ≤ 16,384, 0 ≤ eps < 1, log_pi - log_pi_old in [-16, 16]
 - Performance is measured with B = 256, S = 16,384 (4M positions, 48 MB read)
*/

/*
Approach: a bandwidth-bound, deterministic two-pass reduction over the flat
B*S array.

 1. ppo_partial  A fixed grid (≤ 2048 blocks x 256 threads) grid-strides over
                 the data with float4 loads when B*S % 4 == 0, scalar
                 otherwise. Each thread accumulates its own short run in fp32
                 (about 8 float4 = 32 values at the benchmark). That's
                 converted to double for the warp-shuffle and block
                 reduction, and the block total goes to partial[block]
                 (double).
 2. ppo_final    One block sums the partials in double and writes
                 -(sum / (B*S)) as float.

Why double above thread level: ratios can reach e^16 ≈ 9e6, so individual
surrogates span many orders of magnitude, and a 4M-term fp32 tree would lose
digits. The double work is only one value per thread plus a tree per block,
which is negligible even on GPUs with slow FP64. There are no atomics, so the
result is bit-identical run to run.
*/

#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256
#define NWARPS  (THREADS / 32)
#define MAX_BLOCKS 2048

__device__ double g_partial[MAX_BLOCKS];

__device__ __forceinline__ float surr(float a, float lp, float lpo, float lo, float hi)
{
    const float r  = expf(lp - lpo);
    const float rc = fminf(fmaxf(r, lo), hi);
    return fminf(r * a, rc * a);
}

__device__ __forceinline__ double block_sum_d(double v)
{
    __shared__ double ws[NWARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    double t = 0.0;
    if (warp == 0) {
        t = (lane < NWARPS) ? ws[lane] : 0.0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffff, t, o);
    }
    return t;                                      // valid in thread 0
}

// ---------------------------------------------------------------- pass 1
__global__ void __launch_bounds__(THREADS)
ppo_partial(const float* __restrict__ A, const float* __restrict__ lp,
            const float* __restrict__ lpo, long long n, float lo, float hi, bool vec)
{
    const long long tid = (long long)blockIdx.x * THREADS + threadIdx.x;
    const long long stride = (long long)gridDim.x * THREADS;
    float acc = 0.f;

    if (vec) {
        const long long n4 = n >> 2;
        const float4* A4 = reinterpret_cast<const float4*>(A);
        const float4* P4 = reinterpret_cast<const float4*>(lp);
        const float4* O4 = reinterpret_cast<const float4*>(lpo);
        for (long long i = tid; i < n4; i += stride) {
            const float4 a = __ldg(A4 + i), p = __ldg(P4 + i), o = __ldg(O4 + i);
            acc += surr(a.x, p.x, o.x, lo, hi) + surr(a.y, p.y, o.y, lo, hi)
                 + surr(a.z, p.z, o.z, lo, hi) + surr(a.w, p.w, o.w, lo, hi);
        }
    } else {
        for (long long i = tid; i < n; i += stride)
            acc += surr(__ldg(A + i), __ldg(lp + i), __ldg(lpo + i), lo, hi);
    }

    const double t = block_sum_d((double)acc);
    if (threadIdx.x == 0) g_partial[blockIdx.x] = t;
}

// ---------------------------------------------------------------- pass 2
__global__ void __launch_bounds__(THREADS)
ppo_final(float* __restrict__ out, int nBlocks, long long n)
{
    double v = 0.0;
    for (int i = threadIdx.x; i < nBlocks; i += THREADS) v += g_partial[i];
    const double t = block_sum_d(v);
    if (threadIdx.x == 0) out[0] = (float)(-t / (double)n);
}

// ---------------------------------------------------------------- host
// advantages, log_pi, log_pi_old, output are device pointers
extern "C" void solve(const float* advantages, const float* log_pi, const float* log_pi_old,
                      float* output, float clip_eps, int B, int S)
{
    const long long n = (long long)B * S;
    if (n <= 0) return;
    const bool vec = (n % 4) == 0;
    const long long work = vec ? n / 4 : n;
    // ~8 work items per thread at the benchmark, capped at MAX_BLOCKS
    long long blocks = (work + THREADS * 8 - 1) / (THREADS * 8);
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    if (blocks < 1) blocks = 1;

    ppo_partial<<<(unsigned)blocks, THREADS>>>(advantages, log_pi, log_pi_old, n,
                                                1.0f - clip_eps, 1.0f + clip_eps, vec);
    ppo_final<<<1, THREADS>>>(output, (int)blocks, n);
    cudaDeviceSynchronize();
}
