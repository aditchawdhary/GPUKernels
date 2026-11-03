/*
Leaky ReLU
Implement a program that performs the leaky ReLU activation function on a vector of floating-point numbers. The leaky ReLU function is defined as:
 
f(x) = x if x > 0
     = alpha * x if x <= 0

where alpha is a small positive constant (0.01 in this problem).

Implementation Requirements
- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in vector output
- Use alpha = 0.01 as the leaky coefficient

Example 1:
  Input:  x = [1.0, -2.0, 3.0, -4.0]
  Output: y = [1.0, -0.02, 3.0, -0.04]

Example 2:
  Input:  x = [-1.5, 0.0, 2.5, -3.0]
  Output: y = [-0.015, 0.0, 2.5, -0.03]

Constraints
- 1 ≤ N ≤ 100,000,000
- -1000.0 ≤ input[i] ≤ 1000.0
- Performance is measured with N = 50,000,000
*/

#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) 
{
    float alpha = 0.01;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N)
    {
        float val = input[idx];
        if (val < 0)
        {
            val = val * alpha;
        }
        output[idx] = val;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
