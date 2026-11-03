/*
Matrix Copy

Implement a program that copies an NxN matrix of 32-bit floating point numbers from input array A to output array B on the GPU. 
The program should perform a direct element-wise copy so that B_i,j = A_i,j for all valid indices.

Implementation Requirements
- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in matrix B

Example 1:
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
Output: B = [[1.0, 2.0],
             [3.0, 4.0]]
Example 2:
Input:  A = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Output: B = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Constraints
- 1 ≤ N ≤ 4096
- All elements are 32-bit floating point numbers
- Performance is measured with N = 4,096
*/

#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int total)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total)
    {
        B[idx] = A[idx];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    cudaDeviceSynchronize();
}
