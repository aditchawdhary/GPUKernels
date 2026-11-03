# 1D Convolution

Implement a program that performs a 1D convolution operation. Given an input array and a kernel (filter), compute the convolved output. 
The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

<img width="425" height="220" alt="image" src="https://github.com/user-attachments/assets/3d41e54a-7384-4b4c-bb78-044a824aea4d" />


The input consists of two arrays:

- input: A 1D array of 32-bit floating-point numbers.
- kernel: A 1D array of 32-bit floating-point numbers representing the convolution kernel.

The output should be written to the output array, which will have a size of input_size - kernel_size + 1.
The convolution operation is defined mathematically as:

<img width="409" height="71" alt="image" src="https://github.com/user-attachments/assets/a3fa3826-e03d-4711-8291-d4e1da6851a0" />

where i ranges from 0 to input_size - kernel_size

Implementation Requirements
- Use only native features (external libraries are not permitted)
- The solve function signature must remain unchanged
- The final result must be stored in the array output

```
Example 1:
Input: input = [1, 2, 3, 4, 5], kernel = [1, 0, -1]
Output: [-2, -2, -2]
```
```
Example 2:
Input: input = [2, 4, 6, 8], kernel = [0.5, 0.2]
Output: [1.8, 3.2, 4.6]
```

Constraints
- 1 ≤ input_size ≤ 1,500,000
- 1 ≤ kernel_size ≤ 2047
- kernel_size ≤ input_size
- Performance is measured with input_size = 1,500,000, kernel_size = 2,047

```cuda
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) 
{
    
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}

```
