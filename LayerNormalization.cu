#include <cuda_runtime.h>

#define WARPS_PER_BLOCK 8

__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// One warp normalizes one row of length C.
__global__ void layernorm_kernel(const float* input, const float* weight, const float* bias,
                                 float* output, int N, int C, float eps) {
    int row  = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= N) return;

    const float* x = input  + (size_t)row * C;
    float*       y = output + (size_t)row * C;

    // Pass 1: mean
    float sum = 0.0f;
    for (int j = lane; j < C; j += 32) sum += x[j];
    float mean = warp_sum(sum) / C;

    // Pass 2: variance (two-pass is more accurate than sum of squares)
    float sq = 0.0f;
    for (int j = lane; j < C; j += 32) {
        float d = x[j] - mean;
        sq += d * d;
    }
    float inv_std = rsqrtf(warp_sum(sq) / C + eps);

    // Pass 3: normalize, scale, shift
    for (int j = lane; j < C; j += 32)
        y[j] = (x[j] - mean) * inv_std * weight[j] + bias[j];
}

// input, weight, bias, output are device pointers
extern "C" void solve(const float* input, const float* weight, const float* bias,
                      float* output, int N, int C, float eps) {
    int threads = WARPS_PER_BLOCK * 32;
    int blocks  = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    layernorm_kernel<<<blocks, threads>>>(input, weight, bias, output, N, C, eps);
}
