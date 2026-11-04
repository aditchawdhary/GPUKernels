/*
Value Clipping
Easy
Implement a GPU program that performs clipping on 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the output by clipping each element to a specified range [lo, hi]. The input and output tensor must be of type float32.

Clipping is defined as:

For each element x in the input tensor, "clip" the element so that it falls within the allowed range [lo, hi].
This operation ensures all values are within the specified range and is commonly used in ML for activation stabilization and pre-quantization.
Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output tensor
Example 1:
Input:  [1.5, -2.0, 3.0, 4.5], lo = 0.0, hi = 3.5
Output: [1.5, 0.0, 3.0, 3.5]
Example 2:
Input:  [-1.0, 2.0, 5.0], lo = -0.5, hi = 2.5
Output: [-0.5, 2.0, 2.5]
Constraints
1 ≤ N ≤ 100,000
-106 ≤ input[i] ≤ 106
lo ≤ hi
Performance is measured with N = 100,000
*/

#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
