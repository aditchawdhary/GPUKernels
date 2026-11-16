/*
Group Normalization
X, Y: (N, C, H, W) float32; gamma, beta: (C). Channels split into G contiguous groups.
For each (n, g): mean and (biased) variance over its (C/G) * H * W elements, then
  Y = (X - mean) / sqrt(var + eps) * gamma[c] + beta[c]

Constraints
 - N ≤ 32, C ≤ 1024 (divisible by G), H, W ≤ 128, eps = 1e-5
 - Performance is measured with N = 8, C = 512, H = W = 64, G = 32
*/

/*
Approach: in NCHW each (n, g) group is one contiguous run of S = (C/G)*H*W floats
(65,536 at the benchmark). The work is split into 4096-element chunks, so even
G = 1 with huge groups keeps the whole GPU busy, and the statistics are computed
stably.

 1. chunk_stats  grid (chunks, N*G). A block holds its chunk in registers
                 (256 threads x 16, coalesced loads) and computes the chunk mean,
                 then M2 = sum (x - mean_c)^2, both with block reductions. This
                 two-pass form avoids the cancellation of sum/sum-of-squares.
                 Writes (count, mean, M2).
 2. group_stats  One block per group merges its chunks with Chan's formula:
                   mean = sum n_c mean_c / S,
                   M2   = sum (M2_c + n_c (mean_c - mean)^2),
                 then writes (mean, rstd = 1/sqrt(M2/S + eps)).
 3. normalize    Same grid as pass 1:
                 y = (x - mean) * rstd * gamma[c] + beta[c],
                 with the channel c = g*(C/G) + offset / (H*W). The scale and
                 shift are folded so each element costs one FMA.

Traffic: X is read twice and Y written once (about 192 MB at the benchmark), so
it's bandwidth-bound. Scratch is allocated once and reused.
*/

#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256
#define ITEMS   16
#define CHUNK   (THREADS * ITEMS)          // 4096
#define NWARPS  (THREADS / 32)

__device__ __forceinline__ float block_sum(float v)
{
    __shared__ float ws[NWARPS];
    __shared__ float r;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < NWARPS) ? ws[lane] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) r = v;
    }
    __syncthreads();
    const float out = r;
    __syncthreads();
    return out;
}

// ---------------------------------------------------------------- pass 1
__global__ void __launch_bounds__(THREADS)
chunk_stats(const float* __restrict__ X, long long S, int nCh, float3* __restrict__ cst)
{
    const int c = blockIdx.x, grp = blockIdx.y;
    const float* base = X + (size_t)grp * S;
    const long long start = (long long)c * CHUNK;
    const int cnt = (int)min((long long)CHUNK, S - start);

    float v[ITEMS];
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int i = j * THREADS + threadIdx.x;
        v[j] = (i < cnt) ? __ldg(base + start + i) : 0.f;
        s += v[j];
    }
    const float mean = block_sum(s) / (float)cnt;

    float m2 = 0.f;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int i = j * THREADS + threadIdx.x;
        if (i < cnt) { const float d = v[j] - mean; m2 = fmaf(d, d, m2); }
    }
    m2 = block_sum(m2);
    if (threadIdx.x == 0) cst[(size_t)grp * nCh + c] = make_float3((float)cnt, mean, m2);
}

// ---------------------------------------------------------------- pass 2
__global__ void __launch_bounds__(THREADS)
group_stats(const float3* __restrict__ cst, int nCh, long long S, float eps,
            float2* __restrict__ gst)
{
    const int grp = blockIdx.x;
    const float3* p = cst + (size_t)grp * nCh;
    const float invS = (float)(1.0 / (double)S);

    float a = 0.f;
    for (int c = threadIdx.x; c < nCh; c += THREADS) a = fmaf(p[c].x, p[c].y, a);
    const float mean = block_sum(a) * invS;

    float m2 = 0.f;
    for (int c = threadIdx.x; c < nCh; c += THREADS) {
        const float3 q = p[c];
        const float d = q.y - mean;
        m2 += q.z + q.x * d * d;
    }
    m2 = block_sum(m2);
    if (threadIdx.x == 0) gst[grp] = make_float2(mean, rsqrtf(m2 * invS + eps));
}

// ---------------------------------------------------------------- pass 3
__global__ void __launch_bounds__(THREADS)
normalize(const float* __restrict__ X, const float* __restrict__ gamma,
          const float* __restrict__ beta, const float2* __restrict__ gst,
          float* __restrict__ Y, long long S, int HW, int Cg, int G)
{
    const int c = blockIdx.x, grp = blockIdx.y;
    const float2 st = gst[grp];
    const int g = grp % G;
    const size_t base = (size_t)grp * S;
    const long long start = (long long)c * CHUNK;
    const int cnt = (int)min((long long)CHUNK, S - start);

    #pragma unroll 4
    for (int j = 0; j < ITEMS; ++j) {
        const int i = j * THREADS + threadIdx.x;
        if (i < cnt) {
            const int off = (int)start + i;               // S ≤ 16.7M, fits in int
            const int ch = g * Cg + off / HW;
            const float sc = st.y * __ldg(gamma + ch);
            const float sh = fmaf(-st.x, sc, __ldg(beta + ch));
            Y[base + off] = fmaf(__ldg(X + base + off), sc, sh);
        }
    }
}

// ---------------------------------------------------------------- host
static void*  g_scratch = nullptr;
static size_t g_bytes = 0;

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N, int C,
                      int H, int W, int G, float eps)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || G <= 0) return;
    const int Cg = C / G;
    const int HW = H * W;
    const long long S = (long long)Cg * HW;          // elements per group
    const int nCh = (int)((S + CHUNK - 1) / CHUNK);
    const int NG = N * G;

    const size_t need = (size_t)NG * nCh * sizeof(float3) + (size_t)NG * sizeof(float2) + 16;
    if (need > g_bytes) {
        if (g_scratch) cudaFree(g_scratch);
        cudaMalloc(&g_scratch, need);
        g_bytes = need;
    }
    float2* gst = reinterpret_cast<float2*>(g_scratch);                     // 8-byte aligned
    float3* cst = reinterpret_cast<float3*>(gst + NG);

    dim3 grid(nCh, NG);
    chunk_stats<<<grid, THREADS>>>(X, S, nCh, cst);
    group_stats<<<NG, THREADS>>>(cst, nCh, S, eps, gst);
    normalize<<<grid, THREADS>>>(X, gamma, beta, gst, Y, S, HW, Cg, G);
    cudaDeviceSynchronize();
}
