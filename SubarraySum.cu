/*
Subarray Sum
Implement a program that computes the sum of a subarray of 32-bit integers. You are given an input array input of length N, and two indices S and E. S and E are inclusive, 0-based start and end indices — compute the sum of input[S..E].

Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output variable

Example 1:
Input: input = [1, 2, 1, 3, 4], S = 1, E = 3
Output: output = 6

Example 2:
Input: input = [1, 2, 3, 4], S = 0, E = 3
Output: output = 10

Constraints
 - 1 ≤ N ≤ 100,000,000
 - 1 ≤ input[i] ≤ 10
 - 0 ≤ S ≤ E ≤ N - 1
 - Performance is measured with N = 100,000,000

*/

#include <cuda_runtime.h>

// Sum of input[S..E] (inclusive).
// Max possible sum = 1e8 * 10 = 1e9 < 2^31, so int accumulation never overflows.
//
// The range [S, E+1) is split into
//   head: up to 3 ints before the first 16-byte boundary
//   body: aligned int4 loads, grid-stride, 4 int4 loads in flight per thread
//   tail: up to 3 ints after the last 16-byte boundary
// Each block reduces in registers -> warp shuffles -> shared memory, then does
// one atomicAdd, about 1k atomics in total.

#define THREADS 256
#define UNROLL  4

__global__ void __launch_bounds__(THREADS)
subarray_sum(const int* __restrict__ input, int* __restrict__ output, int S, int E)
{
    const int lo = S, hi = E + 1;                 // half-open range
    const int a  = min((lo + 3) & ~3, hi);        // first aligned index (clamped)
    const int b  = max(a, hi & ~3);               // last aligned boundary

    const int tid    = blockIdx.x * THREADS + threadIdx.x;
    const int stride = gridDim.x * THREADS;

    int sum = 0;

    // head / tail: at most 3 elements each
    if (tid < a - lo)  sum += __ldg(input + lo + tid);
    if (tid < hi - b)  sum += __ldg(input + b + tid);

    // aligned body
    const int4* in4 = reinterpret_cast<const int4*>(input);
    const int q0 = a >> 2, q1 = b >> 2;           // int4 range [q0, q1)

    int i = q0 + tid;
    for (; i + (UNROLL - 1) * stride < q1; i += UNROLL * stride) {
        int4 v[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = __ldcs(in4 + i + u * stride);
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) sum += (v[u].x + v[u].y) + (v[u].z + v[u].w);
    }
    for (; i < q1; i += stride) {
        int4 v = __ldcs(in4 + i);
        sum += (v.x + v.y) + (v.z + v.w);
    }

    // block reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);

    __shared__ int warpSum[THREADS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warpSum[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < THREADS / 32) ? warpSum[lane] : 0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0 && sum != 0) atomicAdd(output, sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E)
{
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, subarray_sum, THREADS, 0);

    // One resident wave of blocks, but no more than the work needs
    const int len4   = (E - S + 1 + 3) / 4 + 1;
    int blocks = sms * occ;
    const int need = (len4 + THREADS - 1) / THREADS;
    if (blocks > need) blocks = need;
    if (blocks < 1) blocks = 1;

    cudaMemsetAsync(output, 0, sizeof(int));
    subarray_sum<<<blocks, THREADS>>>(input, output, S, E);
    cudaDeviceSynchronize();
}
