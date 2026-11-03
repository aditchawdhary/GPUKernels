/*
Color Inversion

Write a program to invert the colors of an image. The image is represented as a 1D array of RGBA (Red, Green, Blue, Alpha) values, where each component is an 8-bit unsigned integer `(unsigned char)`.

Color inversion is performed by subtracting each color component (R, G, B) from 255. The Alpha component should remain unchanged.

The input array `image` will contain `width * height * 4` elements. The first 4 elements represent the RGBA values of the top-left pixel, the next 4 elements represent the pixel to its right, and so on.

Implementation Requirements
- Use only native features (external libraries are not permitted)
- The solve function signature must remain unchanged
- The final result must be stored in the array image

Example 1:
Input: image = [255, 0, 128, 255, 0, 255, 0, 255], width=1, height=2
Output: [0, 255, 127, 255, 255, 0, 255, 255]

Example 2:
Input: image = [10, 20, 30, 255, 100, 150, 200, 255], width=2, height=1
Output: [245, 235, 225, 255, 155, 105, 55, 255]

Constraints
- 1 ≤ width ≤ 4096
- 1 ≤ height ≤ 4096
- width * height ≤ 8,388,608.
- Performance is measured with height = 5,120, width = 4,096

*/

#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) 
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < width * height)
    {    
        reinterpret_cast<unsigned int*>(image)[tid] ^= 0x00FFFFFFu;
    }
}

// image is a device pointer (i.e. pointer to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}

/*
Optimized

#include <cuda_runtime.h>
#include <cstdint>

__global__ void invert_kernel(unsigned char* __restrict__ image, int width, int height) 
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < width * height / 4)
    {
        uint4* vec = reinterpret_cast<uint4*>(image);
        uint4 p = vec[tid];
        p.x ^= 0x00FFFFFFu;
        p.y ^= 0x00FFFFFFu;
        p.z ^= 0x00FFFFFFu;
        p.w ^= 0x00FFFFFFu;
        vec[tid] = p;
    } 
    else if (tid < (width * height)/4 + (width * height) - ((width * height / 4) * 4))
    {
        unsigned int* pixels = reinterpret_cast<unsigned int*>(image);
        pixels[(width * height / 4) * 4 + (tid - (width * height / 4))] ^= 0x00FFFFFFu;
    }
}

// image is a device pointer (i.e. pointer to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int numVec    = width * height / 4;
    int remainder = width * height - ((width * height / 4) * 4);

    int threadsPerBlock = 1024;
    int threadsNeeded   = numVec + remainder;
    int blocksPerGrid   = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
}

*/
