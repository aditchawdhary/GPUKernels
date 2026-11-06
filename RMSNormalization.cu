#include <cuda_runtime.h>

#define THREADS 256

// Global accumulator for the sum of squares.
__device__ float g_sum_sq;

// Kernel 1: each block sums x^2 over its elements, then adds to g_sum_sq.
__global__ void sum_of_squares(const float* input, int N) {
    __shared__ float partial[THREADS];
    int tid = threadIdx.x;

    // Grid-stride loop: each thread accumulates several elements.
    float local = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x) {
        float x = input[i];
        local += x * x;
    }

    // Tree reduction within the block.
    partial[tid] = local;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    // One thread per block adds the block's total to the global sum.
    if (tid == 0) atomicAdd(&g_sum_sq, partial[0]);
}

// Kernel 2: y = gamma * x / rms + beta
__global__ void normalize(const float* input, float* output, float gamma, float beta,
                          int N, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float inv_rms = rsqrtf(g_sum_sq / N + eps);   // 1 / sqrt(mean(x^2) + eps)
        output[i] = gamma * input[i] * inv_rms + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    // Reset the accumulator to zero.
    float zero = 0.0f;
    cudaMemcpyToSymbol(g_sum_sq, &zero, sizeof(float));

    int blocks = (N + THREADS - 1) / THREADS;

    // Cap the reduction grid; the grid-stride loop covers the rest.
    int reduce_blocks = blocks < 256 ? blocks : 256;
    sum_of_squares<<<reduce_blocks, THREADS>>>(input, N);

    normalize<<<blocks, THREADS>>>(input, output, gamma, beta, N, eps);
}
