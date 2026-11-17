/*
DPO Sequence Loss
chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps: (B) float32.
  z_i    = beta * ((chosen_i - rejected_i) - (chosen_ref_i - rejected_ref_i))
  output = -mean(logsigmoid(z)) = mean(softplus(-z))
(Spec: leetgpu-challenges #108, tolerance 1e-4. B ≤ 65,536; performance at B = 65,536.)
*/

/*
Approach: a deterministic two-pass reduction.

 - Per element, the margins are formed in the reference's order: policy margin and
   reference margin first, then their difference. With log-probs around -2000 whose
   margins differ by a few units, this grouping keeps the same rounding as the
   reference. Then z = beta * diff.
 - softplus(-z) is evaluated stably as max(-z, 0) + log1p(exp(-|z|)). That gives the
   right answer at z = ±100 (0 and 100), where -log(sigmoid(z)) would underflow to
   log(0) = -inf or lose all precision.
 - Pass 1: ≤ 64 blocks x 256 threads grid-stride over the batch. Each thread sums in
   fp32 over a few elements, then the warp-shuffle and block reduction is done in
   double. Pass 2: one warp adds the block partials in double and writes sum / B.
   There are no atomics, so the result is identical run to run.
At the benchmark the inputs are 1 MB, so this is launch-latency bound (a few µs).
*/

#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256
#define NWARPS (THREADS / 32)
#define MAX_BLOCKS 64

__device__ double g_part[MAX_BLOCKS];

__device__ __forceinline__ float dpo_term(float c, float r, float cr, float rr, float beta)
{
    const float pol = c - r;
    const float ref = cr - rr;
    const float z = beta * (pol - ref);
    return fmaxf(-z, 0.f) + log1pf(expf(-fabsf(z)));     // softplus(-z)
}

__global__ void __launch_bounds__(THREADS)
dpo_partial(const float* __restrict__ c, const float* __restrict__ r,
            const float* __restrict__ cr, const float* __restrict__ rr, float beta, int B)
{
    __shared__ double ws[NWARPS];
    float acc = 0.f;
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < B; i += gridDim.x * THREADS)
        acc += dpo_term(__ldg(c + i), __ldg(r + i), __ldg(cr + i), __ldg(rr + i), beta);

    double v = acc;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0) ws[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = (threadIdx.x < NWARPS) ? ws[threadIdx.x] : 0.0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (threadIdx.x == 0) g_part[blockIdx.x] = v;
    }
}

__global__ void dpo_final(float* __restrict__ out, int nBlocks, int B)
{
    double v = (threadIdx.x < nBlocks) ? g_part[threadIdx.x] : 0.0;
    if (threadIdx.x + 32 < nBlocks) v += g_part[threadIdx.x + 32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) out[0] = (float)(v / (double)B);
}

// chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output are device pointers
extern "C" void solve(const float* chosen_logps, const float* rejected_logps,
                      const float* chosen_ref_logps, const float* rejected_ref_logps, float* output,
                      float beta, int B)
{
    if (B <= 0) return;
    int blocks = (B + THREADS * 4 - 1) / (THREADS * 4);    // ~4 elements per thread
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    dpo_partial<<<blocks, THREADS>>>(chosen_logps, rejected_logps, chosen_ref_logps,
                                     rejected_ref_logps, beta, B);
    dpo_final<<<1, 32>>>(output, blocks, B);
    cudaDeviceSynchronize();
}
