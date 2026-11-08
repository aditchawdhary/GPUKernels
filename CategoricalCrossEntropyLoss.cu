#include <cuda_runtime.h>
#include <float.h>

#define THREADS 256
#define WARPS   (THREADS / 32)

// Merge two (max, sum-of-exp) pairs
__device__ __forceinline__ void mergeMS(float& m, float& s, float m2, float s2)
{
    float mn = fmaxf(m, m2);
    s = s * __expf(m - mn) + s2 * __expf(m2 - mn);
    m = mn;
}

__global__ void cce_kernel(const float* __restrict__ logits,
                           const int* __restrict__ labels,
                           float* __restrict__ loss, int N, int C)
{
    __shared__ float warpLoss[WARPS];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row  = blockIdx.x * WARPS + warp;

    float rowLoss = 0.0f;
    if (row < N) {
        const float* z = logits + (size_t)row * C;

        // Online max / sum-of-exp over this lane's strided elements
        float m = -FLT_MAX, s = 0.0f;
        for (int k = lane; k < C; k += 32) {
            float x = __ldg(&z[k]);
            if (x > m) { s = s * __expf(m - x) + 1.0f; m = x; }
            else       { s += __expf(x - m); }
        }

        // Combine the 32 lanes' (m, s) pairs
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float m2 = __shfl_down_sync(0xffffffff, m, off);
            float s2 = __shfl_down_sync(0xffffffff, s, off);
            mergeMS(m, s, m2, s2);
        }

        if (lane == 0) {
            int y = __ldg(&labels[row]);
            float zy = (y >= 0 && y < C) ? __ldg(&z[y]) : 0.0f;
            rowLoss = m + __logf(s) - zy;          // log-sum-exp minus true logit
        }
    }

    if (lane == 0) warpLoss[warp] = rowLoss;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < WARPS) ? warpLoss[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) atomicAdd(loss, v / (float)N);
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C)
{
    cudaMemset(loss, 0, sizeof(float));
    int blocks = (N + WARPS - 1) / WARPS;
    cce_kernel<<<blocks, THREADS>>>(logits, true_labels, loss, N, C);
    cudaDeviceSynchronize();
}
