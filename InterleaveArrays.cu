/*
Interleave Arrays

Write a GPU program that interleaves two arrays of 32-bit floating point numbers. Given two input arrays A and B, each of length N, produce an output array of length 2N where elements alternate between the two inputs: [A[0], B[0], A[1], B[1], A[2], B[2], ...]

A
a₀
a₁
a₂
a₃
B
b₀
b₁
b₂
b₃
output
a₀
b₀
a₁
b₁
a₂
b₂
a₃
b₃
Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output array
Example 1:
Input:  A = [1.0, 2.0, 3.0], B = [4.0, 5.0, 6.0]
Output: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
Example 2:
Input:  A = [10.0, 20.0], B = [30.0, 40.0]
Output: [10.0, 30.0, 20.0, 40.0]
Constraints
1 ≤ N ≤ 50,000,000
Performance is measured with N = 25,000,000
*/

#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
