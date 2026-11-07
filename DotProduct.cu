/*
Dot Product

Implement a GPU program that computes the dot product of two vectors containing 32-bit floating point numbers. The dot product is the sum of the products of the corresponding elements of two vectors.

Mathematically, the dot product of two vectors A and B of length n is defined as:
A dot B = \sum_{i=0}_{n-1} Ai * Bi = A0 * B0 + A1 * B1 + A2 * B2 + .... + An * Bn

Implementation Requirements
 - Use only GPU native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output variable

Example 1:
Input:  A = [1.0, 2.0, 3.0, 4.0]
        B = [5.0, 6.0, 7.0, 8.0]
Output: result = 70.0  (1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0)

Example 2:
Input:  A = [0.5, 1.5, 2.5]
        B = [2.0, 3.0, 4.0]
Output: result = 15.5  (0.5*2.0 + 1.5*3.0 + 2.5*4.0)

Constraints
 - A and B have identical lengths
 - 1 ≤ N ≤ 100,000,000
 - Performance is measured with N = 4,194,304
*/


#include <cuda_runtime.h>

#define THREADS 256
#define WARPS   (THREADS / 32)

__inline__ __device__ float warpReduceSum(float v)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__global__ void dot_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ result, int N)
{
    __shared__ float warpSums[WARPS];

    const int tid    = threadIdx.x;
    const int gtid   = blockIdx.x * THREADS + tid;
    const int stride = gridDim.x * THREADS;

    // Vectorized main loop: 4 elements per load, two independent accumulators
    float acc0 = 0.0f, acc1 = 0.0f;
    const int n4 = N >> 2;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    for (int i = gtid; i < n4; i += stride) {
        float4 a = __ldg(&A4[i]);
        float4 b = __ldg(&B4[i]);
        acc0 = fmaf(a.x, b.x, acc0);
        acc1 = fmaf(a.y, b.y, acc1);
        acc0 = fmaf(a.z, b.z, acc0);
        acc1 = fmaf(a.w, b.w, acc1);
    }

    // Tail (N not divisible by 4)
    for (int i = (n4 << 2) + gtid; i < N; i += stride)
        acc0 = fmaf(A[i], B[i], acc0);

    float sum = acc0 + acc1;

    // Warp-level reduction
    sum = warpReduceSum(sum);
    if ((tid & 31) == 0) warpSums[tid >> 5] = sum;
    __syncthreads();

    // Block-level reduction in the first warp
    if (tid < 32) {
        sum = (tid < WARPS) ? warpSums[tid] : 0.0f;
        sum = warpReduceSum(sum);
        if (tid == 0) atomicAdd(result, sum);
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N)
{
    static int numSMs = 0;
    if (numSMs == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
    }

    cudaMemset(result, 0, sizeof(float));

    int maxBlocks = numSMs * 8;
    int needed    = (N / 4 + THREADS - 1) / THREADS;
    if (needed < 1) needed = 1;
    int blocks = needed < maxBlocks ? needed : maxBlocks;

    dot_kernel<<<blocks, THREADS>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
