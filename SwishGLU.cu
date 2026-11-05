/*
Swish-Gated Linear Unit

Implement the Swish-Gated Linear Unit (SWiGLU) activation function forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the output using the elementwise formula. 
The input and output tensor must be of type float32.

SWiGLU is defined as:
 1. Split input x into two halves: x1 and x2 
 2. Compute SiLU on the first half: 
            SiLU(x1) = x1 * Sigma(x1), 
            Sigma(x1) = 1 / (1 + exp(-x))
 
 3. Compute the SWiGLU output:
            SWiGLU(x1, x2) = SiLU(x1) * x2

Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output tensor

Example 1:
Input:  [1.0, 2.0, 3.0, 4.0]  (N=4)
Output: [2.1931758, 7.0463767]

Example 2:
Input:  [0.5, 1.0]  (N=2)
Output: [0.31122968]

Constraints
 - 1 ≤ N ≤ 100,000
 - N is an even number
 - -100.0 ≤ input values ≤ 100.0
 - Performance is measured with N = 100,000

*/


#include <cuda_runtime.h>
#include <math.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < halfN)
    {
        float x1 = input[idx];
        float x2 = input[halfN + idx];

        output[idx] =  (x1 / (1.0f + exp(-x1))) * x2;

    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
