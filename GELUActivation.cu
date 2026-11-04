/*
GELU Activation

Write a GPU program that applies the Gaussian Error Linear Unit (GELU) activation element-wise to a vector of 32-bit floating point numbers. 
For each element x in the input vector X, compute the exact (erf-based) GELU and store the result in the output vector Y:
 
GELU(x) = x * phi(x) 
        = x/2 * (1 + erf(x / sqrt(2)))
GELU is the activation used in the feed-forward blocks of BERT, GPT-2 and ViT.

Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - The final result must be stored in vector Y
 - Use the exact erf form, not the tanh approximation

Example 1:
Input:  X = [0.0, 1.0, -1.0, 2.0]
Output: Y = [0.0, 0.8413, -0.1587, 1.9545]

Example 2:
Input:  X = [0.5, -0.5, 3.0, -3.0]
Output: Y = [0.3457, -0.1543, 2.9959, -0.0041]

Constraints
 - 1 ≤ N ≤ 100,000,000
 - Input values are finite 32-bit floating point numbers
 - Performance is measured with N = 50,000,000
*/

#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* X, float* Y, int N) {}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
