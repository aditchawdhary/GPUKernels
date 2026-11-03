/*
Reverse Array
Implement a program that reverses an array of 32-bit floating point numbers in-place. The program should perform an in-place reversal of input.

Implementation Requirements
- Use only native features (external libraries are not permitted)
- The solve function signature must remain unchanged
- The final result must be stored back in input

Example 1:
Input: [1.0, 2.0, 3.0, 4.0]
Output: [4.0, 3.0, 2.0, 1.0]

Example 2:
Input: [1.5, 2.5, 3.5]
Output: [3.5, 2.5, 1.5]

Constraints
- 1 ≤ N ≤ 100,000,000
- Performance is measured with N = 25,000,000

*/

#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N/2)
    {
        float swap = input[idx];
        input[idx] = input[N - 1 - idx];
        input[N - 1 - idx] = swap;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
