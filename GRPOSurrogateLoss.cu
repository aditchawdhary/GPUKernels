#include <cuda_runtime.h>
#include <math.h>

// GRPO surrogate loss.
//   A[b,g]   = (R[b,g] - mean_g R[b,:]) / (std_pop(R[b,:]) + 1e-8)
//   ratio    = exp(log_pi - log_pi_old)
//   surr     = min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A)
//   kl       = exp(log_ref - log_pi) - (log_ref - log_pi) - 1
//   output   = -mean(surr - beta * kl)      over all B*G*S tokens
//
// Kernel 1: one block per (b, g) row. Warp 0 computes the row's advantage from the G rewards,
//           then the block streams the S tokens (float4 loads when aligned) and writes a
//           per-row partial sum (double).
// Kernel 2: one block sums the B*G partials in a fixed order and writes -sum / (B*G*S).

#define THREADS 256

__device__ __forceinline__ float token_term(float lp, float lpo, float lr, float adv,
                                            float lo, float hi, float beta) {
    float ratio = expf(lp - lpo);
    float clipped = fminf(fmaxf(ratio, lo), hi);
    float surr = fminf(ratio * adv, clipped * adv);
    float d = lr - lp;
    float kl = expf(d) - d - 1.0f;
    return surr - beta * kl;
}

__device__ __forceinline__ double block_sum_d(double v) {
    __shared__ double ws[THREADS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    v = (threadIdx.x < THREADS / 32) ? ws[threadIdx.x] : 0.0;
    if (warp == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;   // valid in thread 0
}

__global__ void __launch_bounds__(THREADS)
grpo_rows(const float* __restrict__ rewards, const float* __restrict__ log_pi,
          const float* __restrict__ log_pi_old, const float* __restrict__ log_ref,
          double* __restrict__ partial, float clip_eps, float beta, int G, int S, bool vec) {
    const int row = blockIdx.x;          // = b * G + g
    const int b = row / G, g = row % G;
    __shared__ float s_adv;

    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float r = lane < G ? rewards[b * G + lane] : 0.0f;
        float sum = r;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        float mean = sum / (float)G;
        float dev = lane < G ? (r - mean) : 0.0f;
        float sq = dev * dev;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        float stdv = sqrtf(sq / (float)G);
        if (lane == g) s_adv = (r - mean) / (stdv + 1e-8f);
    }
    __syncthreads();

    const float adv = s_adv;
    const float lo = 1.0f - clip_eps, hi = 1.0f + clip_eps;
    const size_t base = (size_t)row * S;
    float acc = 0.0f;

    if (vec) {
        const float4* p  = reinterpret_cast<const float4*>(log_pi + base);
        const float4* po = reinterpret_cast<const float4*>(log_pi_old + base);
        const float4* pr = reinterpret_cast<const float4*>(log_ref + base);
        const int n4 = S / 4;
        for (int i = threadIdx.x; i < n4; i += THREADS) {
            float4 a = __ldg(&p[i]), o = __ldg(&po[i]), r = __ldg(&pr[i]);
            acc += token_term(a.x, o.x, r.x, adv, lo, hi, beta);
            acc += token_term(a.y, o.y, r.y, adv, lo, hi, beta);
            acc += token_term(a.z, o.z, r.z, adv, lo, hi, beta);
            acc += token_term(a.w, o.w, r.w, adv, lo, hi, beta);
        }
    } else {
        for (int i = threadIdx.x; i < S; i += THREADS)
            acc += token_term(__ldg(&log_pi[base + i]), __ldg(&log_pi_old[base + i]),
                              __ldg(&log_ref[base + i]), adv, lo, hi, beta);
    }

    double s = block_sum_d((double)acc);
    if (threadIdx.x == 0) partial[row] = s;
}

__global__ void __launch_bounds__(THREADS)
grpo_final(const double* __restrict__ partial, int rows, double count, float* __restrict__ output) {
    double v = 0.0;
    for (int i = threadIdx.x; i < rows; i += THREADS) v += partial[i];
    v = block_sum_d(v);
    if (threadIdx.x == 0) output[0] = (float)(-v / count);
}

static double* g_partial = nullptr;
static int g_cap = 0;

// rewards, log_pi, log_pi_old, log_ref, output are device pointers
extern "C" void solve(const float* rewards, const float* log_pi, const float* log_pi_old,
                      const float* log_ref, float* output, float clip_eps, float beta, int B, int G,
                      int S) {
    const int rows = B * G;
    if (rows > g_cap) {
        if (g_partial) cudaFree(g_partial);
        cudaMalloc(&g_partial, rows * sizeof(double));
        g_cap = rows;
    }
    size_t addr = reinterpret_cast<size_t>(log_pi) | reinterpret_cast<size_t>(log_pi_old) |
                  reinterpret_cast<size_t>(log_ref);
    bool vec = (S % 4 == 0) && ((addr & 15) == 0);

    grpo_rows<<<rows, THREADS>>>(rewards, log_pi, log_pi_old, log_ref, g_partial, clip_eps, beta,
                                 G, S, vec);
    grpo_final<<<1, THREADS>>>(g_partial, rows, (double)rows * S, output);
}
