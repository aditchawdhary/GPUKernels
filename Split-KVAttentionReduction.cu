/*
Split-KV Attention Reduction (Flash-Decoding merge)
partial_out: (S, H, D), partial_lse: (S, H), output: (H, D), float32.
  m_h      = max_s lse[s, h]
  w[s, h]  = exp(lse[s, h] - m_h) / sum_s' exp(lse[s', h] - m_h)
  out[h,:] = sum_s w[s, h] * partial_out[s, h, :]

Constraints
 - 1 ≤ S ≤ 256, 1 ≤ H ≤ 128, 1 ≤ D ≤ 256
 - Performance is measured with S = 128, H = 64, D = 128
*/

/*
Approach: grid (H, ceil(D/32)), 256 threads arranged 32 (tx) x 8 (ty).
 1. Every block works out its head's merge weights. Thread s loads lse[s, h];
    block max, exp(lse - m), block sum, then w[s] in shared memory. S ≤ 256, so
    one value per thread. Recomputing this in each of the D/32 blocks per head
    costs nothing.
 2. tx picks the output column d = 32*blockIdx.y + tx, and ty strides over the
    splits (s = ty, ty+8, ...). Each warp load is 128 contiguous bytes of one
    partial_out row, and each thread has S/8 independent loads in flight
    (16 at the benchmark).
 3. The 8 ty partial sums per column are reduced through shared memory and
    written out.
At the benchmark: 64 x 4 = 256 blocks reading 4 MB once, so it's bandwidth- and
latency-bound and finishes in a few microseconds.
*/

#include <cuda_runtime.h>
#include <math.h>

#define TX 32
#define TY 8
#define THREADS (TX * TY)
#define MAX_S 256

__global__ void __launch_bounds__(THREADS)
lse_merge(const float* __restrict__ po, const float* __restrict__ lse, float* __restrict__ out,
          int S, int H, int D)
{
    __shared__ float w[MAX_S];
    __shared__ float red[TY][TX + 1];
    __shared__ float wsum[THREADS / 32];
    __shared__ float bval;

    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int tx = tid & (TX - 1), ty = tid >> 5;   // TX == 32: ty == warp id
    const int lane = tid & 31, warp = tid >> 5;

    // ---- 1. merge weights for head h
    const float l = (tid < S) ? __ldg(lse + (size_t)tid * H + h) : -INFINITY;
    float v = l;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if (lane == 0) wsum[warp] = v;
    __syncthreads();
    if (tid == 0) {
        float m = wsum[0];
        for (int i = 1; i < THREADS / 32; ++i) m = fmaxf(m, wsum[i]);
        bval = m;
    }
    __syncthreads();
    const float m = bval;

    const float e = (tid < S && l != -INFINITY) ? expf(l - m) : 0.f;
    v = e;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    __syncthreads();                                 // everyone has read bval / wsum
    if (lane == 0) wsum[warp] = v;
    __syncthreads();
    if (tid == 0) {
        float s = 0.f;
        for (int i = 0; i < THREADS / 32; ++i) s += wsum[i];
        bval = s;
    }
    __syncthreads();
    const float inv = (bval > 0.f) ? 1.f / bval : 0.f;
    if (tid < S) w[tid] = e * inv;
    __syncthreads();

    // ---- 2. weighted sum over splits for column d
    const int d = blockIdx.y * TX + tx;
    float acc = 0.f;
    if (d < D) {
        #pragma unroll 4
        for (int s = ty; s < S; s += TY)
            acc = fmaf(w[s], __ldg(po + ((size_t)s * H + h) * D + d), acc);
    }
    red[ty][tx] = acc;
    __syncthreads();

    // ---- 3. reduce the TY partials
    if (ty == 0 && d < D) {
        float r = 0.f;
        #pragma unroll
        for (int i = 0; i < TY; ++i) r += red[i][tx];
        out[(size_t)h * D + d] = r;
    }
}

// partial_out, partial_lse, output are device pointers
extern "C" void solve(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim)
{
    const int S = num_splits, H = num_heads, D = head_dim;
    if (S <= 0 || H <= 0 || D <= 0) return;
    dim3 grid(H, (D + TX - 1) / TX);
    lse_merge<<<grid, THREADS>>>(partial_out, partial_lse, output, S, H, D);
    cudaDeviceSynchronize();
}
