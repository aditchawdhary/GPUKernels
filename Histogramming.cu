/*
Histogramming

Write a GPU program that computes the histogram of an array of 32-bit integers. 
The histogram should count the number of occurrences of each integer value in the range [0, num_bins). You are given an input array input of length N and the number of bins num_bins.

The result should be an array of integers of length num_bins, where each element represents the count of occurrences of its corresponding index in the input array.

Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the histogram array.

Examples
Input: input = [0, 1, 2, 1, 0],  N = 5, num_bins = 3
Output: [2, 2, 1]

Input: input = [3, 3, 3, 3], N = 4, num_bins = 5
Output: [0, 0, 0, 4, 0]

Constraints
 - 1 ≤ N ≤ 100,000,000
 - 0 ≤ input[i] < num_bins
 - 1 ≤ num_bins ≤ 1024
 - Performance is measured with N = 50,000,000, num_bins = 256

*/

#include <cuda_runtime.h>

#define THREADS 256
#define WARPS   (THREADS / 32)

__global__ void histogram_kernel(const int* __restrict__ input,
                                 int* __restrict__ histogram,
                                 int N, int num_bins)
{
    extern __shared__ int smem[];               // WARPS * num_bins ints
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    int* myHist = smem + warp * num_bins;

    // Zero all per-warp histograms
    for (int i = tid; i < WARPS * num_bins; i += THREADS)
        smem[i] = 0;
    __syncthreads();

    const int stride = gridDim.x * THREADS;
    const int gtid   = blockIdx.x * THREADS + tid;

    // Vectorized main loop: 4 ints per load
    const int n4 = N >> 2;
    const int4* in4 = reinterpret_cast<const int4*>(input);
    for (int i = gtid; i < n4; i += stride) {
        int4 v = __ldg(&in4[i]);
        atomicAdd(&myHist[v.x], 1);
        atomicAdd(&myHist[v.y], 1);
        atomicAdd(&myHist[v.z], 1);
        atomicAdd(&myHist[v.w], 1);
    }

    // Tail (N not divisible by 4)
    for (int i = (n4 << 2) + gtid; i < N; i += stride)
        atomicAdd(&myHist[input[i]], 1);

    __syncthreads();

    // Merge warp copies, then push block result to global memory
    for (int b = tid; b < num_bins; b += THREADS) {
        int sum = 0;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w)
            sum += smem[w * num_bins + b];
        if (sum) atomicAdd(&histogram[b], sum);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins)
{
    static int numSMs = 0;
    if (numSMs == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
    }

    cudaMemset(histogram, 0, num_bins * sizeof(int));

    size_t smemBytes = (size_t)WARPS * num_bins * sizeof(int);   // max 32 KB

    // Enough blocks to fill the GPU, but not more than the data needs
    int maxBlocks = numSMs * 8;
    int needed    = (N / 4 + THREADS - 1) / THREADS;
    int blocks    = needed < maxBlocks ? (needed > 0 ? needed : 1) : maxBlocks;

    histogram_kernel<<<blocks, THREADS, smemBytes>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
