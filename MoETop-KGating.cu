#include <cuda_runtime.h>
#include <math.h>
#include <limits.h>

// MoE top-k gating: one warp per token row.
//   - E <= 256 logits live in registers (8 per lane).
//   - k rounds of warp-wide argmax (value desc, ties -> smaller index), each round
//     removes the winner, so the output is ordered by descending logit like torch.topk.
//   - softmax over the k selected values (max = first selected value).

#define WARPS 8
#define THREADS (WARPS * 32)
#define EMAX 256
#define PER_LANE (EMAX / 32)   // 8

// Is (v1, i1) better than (v2, i2)? INT_MAX index means "no candidate".
__device__ __forceinline__ bool better(float v1, int i1, float v2, int i2) {
    if (i1 == INT_MAX) return false;
    if (i2 == INT_MAX) return true;
    return (v1 > v2) || (v1 == v2 && i1 < i2);
}

__global__ void __launch_bounds__(THREADS)
topk_gating_kernel(const float* __restrict__ logits, float* __restrict__ topk_weights,
                   int* __restrict__ topk_indices, int M, int E, int k) {
    __shared__ float sv[WARPS][EMAX];

    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    const int row = blockIdx.x * WARPS + w;
    if (row >= M) return;                       // whole warp exits together

    const float* lr = logits + (size_t)row * E;
    float x[PER_LANE];
    unsigned avail = 0;                         // bit r set: element lane + 32r is selectable
    #pragma unroll
    for (int r = 0; r < PER_LANE; ++r) {
        int e = lane + 32 * r;
        x[r] = (e < E) ? __ldg(lr + e) : 0.f;
        if (e < E) avail |= 1u << r;
    }

    float* wout = topk_weights + (size_t)row * k;
    int*   iout = topk_indices + (size_t)row * k;

    for (int it = 0; it < k; ++it) {
        // Lane-local best among remaining elements.
        float bv = -INFINITY;
        int bi = INT_MAX;
        #pragma unroll
        for (int r = 0; r < PER_LANE; ++r) {
            int e = lane + 32 * r;
            if ((avail >> r) & 1u) {
                if (better(x[r], e, bv, bi)) { bv = x[r]; bi = e; }
            }
        }
        // Warp-wide argmax.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            int   oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        // All lanes agree on (bv, bi). Owner removes it; lane 0 records it.
        if ((bi & 31) == lane) avail &= ~(1u << (bi >> 5));
        if (lane == 0) {
            sv[w][it] = bv;
            iout[it] = bi;
        }
    }
    __syncwarp();

    // Softmax over the k selected logits; the first one is the maximum.
    const float mx = sv[w][0];
    float sum = 0.f;
    for (int j = lane; j < k; j += 32) sum += expf(sv[w][j] - mx);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    const float inv = 1.f / sum;
    for (int j = lane; j < k; j += 32) wout[j] = expf(sv[w][j] - mx) * inv;
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    const int blocks = (M + WARPS - 1) / WARPS;
    topk_gating_kernel<<<blocks, THREADS>>>(logits, topk_weights, topk_indices, M, E, k);
    cudaDeviceSynchronize();
}
