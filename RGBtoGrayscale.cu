/*
RGB to Grayscale
Implement a GPU program that converts an RGB image to grayscale on the GPU. 
Given an input RGB image represented as a 1D array of 32-bit floating point values, compute the corresponding grayscale image using the standard RGB to grayscale conversion formula.

The conversion formula is: gray = 0.299 × R + 0.587 × G + 0.114 × B

The input array input contains height × width × 3 elements, where the RGB values for each pixel are stored consecutively (R, G, B, R, G, B, ...). The output array output should contain height × width grayscale values.

Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - The final result must be stored in the array output
 - Use the exact coefficients: 0.299 for red, 0.587 for green, 0.114 for blue

Example 1:
Input:  input = [255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 128.0, 128.0, 128.0], width=2, height=2
Output: output = [76.245, 149.685, 29.07, 128.0]

Example 2:
Input:  input = [100.0, 150.0, 200.0], width=1, height=1
Output: output = [140.75]

Constraints
 - 1 ≤ width ≤ 4096
 - 1 ≤ height ≤ 4096
 - width × height ≤ 4,194,304
 - All RGB values are in the range [0.0, 255.0]
 - Performance is measured with height = 2,048, width = 2,048
*/


#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
