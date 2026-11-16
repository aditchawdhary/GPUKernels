/*
Min-P Sampling
For each row b of logits (B x V, float32):
  p = softmax(logits[b]);  keep tokens with p_i ≥ min_p * max_j p_j;  zero the rest;
  renormalize the kept probabilities to sum to 1.

Constraints
 - 1 ≤ B ≤ 256, 1 ≤ V ≤ 200,000, 0 ≤ min_p ≤ 1, logits in [-50, 50]
 - Performance is measured with B = 64, V = 128,000, min_p = 0.05
*/

/*
Key identity: with M = max_i x_i and e_i = exp(x_i - M),
  p_i = e_i / Z  and  p_max = 1 / Z,  so  p_i ≥ min_p * p_max  <=>  e_i ≥ min_p,
and after renormalizing, probs_i = e_i / S with S = sum of e_j over kept j.
The softmax normalizer Z cancels and is never computed. That leaves one max and
one masked sum per row, and it's numerically stable because the row max is
subtracted first.

Approach: three bandwidth-bound passes over 4096-element chunks, grid (chunks, B).
At the benchmark that's 32 chunks x 64 rows = 2048 blocks, so the whole GPU is busy
even though there are only 64 rows.
 1. chunk_max   per-chunk max  -> pmax[b][c]
 2. chunk_sum   M = max of the row's pmax; per-chunk sum of e_i where e_i ≥ min_p
                -> psum[b][c]
 3. write_out   M and S = sum of the row's psum; probs_i = e_i / S if kept, else
                exactly 0
Loads and stores are coalesced (consecutive threads take consecutive elements, so
any V works). The logits are read 3 times; the second and third reads largely come
from L2 (32 MB of input at the benchmark). The per-chunk scratch is a static device
array (256 rows x 49 chunks max), so there's no allocation.
*/

#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256
#define ITEMS   16
#define CHUNK   (THREADS * ITEMS)                       // 4096
#define NWARPS  (THREADS / 32)
#define MAX_B   256
#define MAX_CH  ((200000 + CHUNK - 1) / CHUNK)          // 49

__device__ float g_pmax[MAX_B * MAX_CH];
__device__ float g_psum[MAX_B * MAX_CH];

__device__ __forceinline__ float block_max(float v)
{
    __shared__ float ws[NWARPS];
    __shared__ float r;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < NWARPS) ? ws[lane] : -INFINITY;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (lane == 0) r = v;
    }
    __syncthreads();
    const float out = r;
    __syncthreads();                                    // ws / r reusable
    return out;
}

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

// row max from the per-chunk maxima (nCh ≤ 49, so one strided read per thread)
__device__ __forceinline__ float row_max(int b, int nCh)
{
    float m = -INFINITY;
    for (int c = threadIdx.x; c < nCh; c += THREADS) m = fmaxf(m, g_pmax[b * nCh + c]);
    return block_max(m);
}

// ---------------------------------------------------------------- pass 1
__global__ void __launch_bounds__(THREADS)
chunk_max(const float* __restrict__ x, int V, int nCh)
{
    const int c = blockIdx.x, b = blockIdx.y;
    const float* row = x + (size_t)b * V;
    const int start = c * CHUNK;
    float m = -INFINITY;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int i = start + j * THREADS + threadIdx.x;
        if (i < V) m = fmaxf(m, __ldg(row + i));
    }
    m = block_max(m);
    if (threadIdx.x == 0) g_pmax[b * nCh + c] = m;
}

// ---------------------------------------------------------------- pass 2
__global__ void __launch_bounds__(THREADS)
chunk_sum(const float* __restrict__ x, int V, int nCh, float min_p)
{
    const int c = blockIdx.x, b = blockIdx.y;
    const float M = row_max(b, nCh);
    const float* row = x + (size_t)b * V;
    const int start = c * CHUNK;
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int i = start + j * THREADS + threadIdx.x;
        if (i < V) {
            const float e = expf(__ldg(row + i) - M);
            if (e >= min_p) s += e;
        }
    }
    s = block_sum(s);
    if (threadIdx.x == 0) g_psum[b * nCh + c] = s;
}

// ---------------------------------------------------------------- pass 3
__global__ void __launch_bounds__(THREADS)
write_out(const float* __restrict__ x, float* __restrict__ probs, int V, int nCh, float min_p)
{
    const int c = blockIdx.x, b = blockIdx.y;
    const float M = row_max(b, nCh);
    float S = 0.f;
    for (int k = threadIdx.x; k < nCh; k += THREADS) S += g_psum[b * nCh + k];
    S = block_sum(S);                                   // ≥ 1: the max token is always kept

    const float* row = x + (size_t)b * V;
    float* out = probs + (size_t)b * V;
    const int start = c * CHUNK;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int i = start + j * THREADS + threadIdx.x;
        if (i < V) {
            const float e = expf(__ldg(row + i) - M);
            out[i] = (e >= min_p) ? e / S : 0.f;
        }
    }
}

// ---------------------------------------------------------------- host
// logits, probs are device pointers
extern "C" void solve(const float* logits, float* probs, float min_p, int B, int V)
{
    if (B <= 0 || V <= 0) return;
    const int nCh = (V + CHUNK - 1) / CHUNK;
    dim3 grid(nCh, B);
    chunk_max<<<grid, THREADS>>>(logits, V, nCh);
    chunk_sum<<<grid, THREADS>>>(logits, V, nCh, min_p);
    write_out<<<grid, THREADS>>>(logits, probs, V, nCh, min_p);
    cudaDeviceSynchronize();
}
