/*
ReLU
Implement a program that performs the Rectified Linear Unit (ReLU) activation function on a vector of 32-bit floating point numbers. 
The ReLU function sets all negative values to zero and leaves positive values unchanged:
ReLU(x) = max(0, x)

Implementation Requirements
- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in output

Example 1:
Input:  input = [-2.0, -1.0, 0.0, 1.0, 2.0]
Output: output = [0.0, 0.0, 0.0, 1.0, 2.0]

Example 2:
Input:  input = [-3.5, 0.0, 4.2]
Output: output = [0.0, 0.0, 4.2]

Constraints
- 1 ≤ N ≤ 100,000,000
- Performance is measured with N = 25,000,000
*/

#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        float value = 0.0f;
        if (input[idx] > 0)
        {
            value = input[idx];
        }
        output[idx] = value;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
