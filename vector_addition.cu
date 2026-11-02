/**
* Vector Addition
* Create a GPU program that performs element-wise addition of two vectors containing 32-bit floating point numbers. 
* The program should take two input vectors of equal length and produce a single output vector containing their sum.
* 
* Implementation Requirements
* External libraries are not permitted
* The solve function signature must remain unchanged
* The final result must be stored in vector C
* 
* 
* Example 1:
* Input:  A = [1.0, 2.0, 3.0, 4.0]
*         B = [5.0, 6.0, 7.0, 8.0]
* Output: C = [6.0, 8.0, 10.0, 12.0]
* 
* 
* Example 2:
* Input:  A = [1.5, 1.5, 1.5]
*         B = [2.3, 2.3, 2.3]
* Output: C = [3.8, 3.8, 3.8]
* 
* 
* Constraints
* Input vectors A and B have identical lengths
* 1 ≤ N ≤ 100,000,000
* Performance is measured with N = 25,000,000
*
*/

#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
  {
    C[i] = A[i] + B[i];
  }
}

// A, B, C are device pointers i.e. pointers to memory on the GPU
extern "C" void solve(const float* A, const float* B, float* C, int N)
{
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  cudaDeviceSynchronize();
}
