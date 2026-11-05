/*
Softmax
Write a program that computes the softmax function for an array of 32-bit floating-point numbers on a GPU. The softmax function is defined as follows:

For an input array x of length n, the softmax of x, denoted sigma(x), is an array of length n where the i-th element is:
sigma(x)_i = exp(x_i) / sum_j=1_n exp(xj)
 

Your solution should handle potential overflow issues by using the "max trick". Subtract the maximum value of the input array from each element before exponentiation.

Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the array output

Example 1:
Input: [1.0, 2.0, 3.0], N = 3
Output: [0.090, 0.244, 0.665] (approximately)

Example 2:
Input: [-10.0, -5.0, 0.0, 5.0, 10.0], N = 5
Output: [2.047e-09, 3.038e-07, 4.509e-05, 6.693e-03, 9.933e-01] (approximately)

Constraints
 - 1 ≤ N ≤ 500,000
 - Performance is measured with N = 500,000
*/


#include <cuda_runtime.h>
#include <math.h>
#define THREADS 256

__device__ float g_max;
__device__ float g_sum;

__device__ void atomicMaxFloat(float* addr, float val)
{
    int* addrAsInt = (int*)addr;
    int old = *addrAsInt, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(addrAsInt, assumed, __float_as_int(val));
    } while (assumed != old);
}

//Step 1: find the maximum value
__global__ void max_kernel(const float* input, int N) 
{
    __shared__ float sdata[THREADS];

    float m = -INFINITY;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
    {
        m = fmaxf(m, input[i]);
    }
    sdata[threadIdx.x] = m;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s = s/2)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicMaxFloat(&g_max, sdata[0]);
    }
}


// Step 2: sum of exp(x - max)
__global__ void sum_kernel(const float* input, int N)
{
    __shared__ float sdata[THREADS];

    float maxVal = g_max;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
    {
        sum += expf(input[i] - maxVal);
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s = s/2)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&g_sum, sdata[0]);
    }
}


// Step 3: write the normalized values
__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = expf(input[idx] - g_max) / g_sum;
    }
    
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int reduceBlocks = blocksPerGrid;
    if (blocksPerGrid > 1024)
    {
        reduceBlocks = 1024;
    }

    float negInf = -INFINITY, zero = 0.0f;
    cudaMemcpyToSymbol(g_max, &negInf, sizeof(float));
    cudaMemcpyToSymbol(g_sum, &zero, sizeof(float));

    max_kernel<<<reduceBlocks, THREADS>>>(input, N);
    sum_kernel<<<reduceBlocks, THREADS>>>(input, N);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

