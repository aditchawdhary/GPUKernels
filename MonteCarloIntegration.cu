#include <cuda_runtime.h>
#include <stdint.h>

#define THREADS 256

__device__ __forceinline__ double block_sum_d(double v) {
    __shared__ double sh[THREADS / 32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < THREADS / 32) ? sh[threadIdx.x] : 0.0;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;   // valid in thread 0
}

__global__ void __launch_bounds__(THREADS)
partial_sum_kernel(const float* __restrict__ y, int n, double* __restrict__ partials) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Scalar head until 16-byte aligned, then float4 body, then scalar tail.
    int head = (int)(((16 - ((uintptr_t)y & 15)) & 15) >> 2);
    if (head > n) head = n;
    double s = 0.0;
    for (int i = tid; i < head; i += stride) s += (double)y[i];

    const float* a = y + head;
    const int m = n - head;
    const int m4 = m >> 2;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    for (int i = tid; i < m4; i += stride) {
        float4 v = __ldg(a4 + i);
        s += (double)v.x + (double)v.y + (double)v.z + (double)v.w;
    }
    for (int i = (m4 << 2) + tid; i < m; i += stride) s += (double)a[i];

    s = block_sum_d(s);
    if (threadIdx.x == 0) partials[blockIdx.x] = s;
}

__global__ void __launch_bounds__(THREADS)
final_kernel(const double* __restrict__ partials, int numPartials,
             float* __restrict__ result, float a, float b, int n) {
    double s = 0.0;
    for (int i = threadIdx.x; i < numPartials; i += THREADS) s += partials[i];
    s = block_sum_d(s);
    if (threadIdx.x == 0)
        *result = (float)(((double)b - (double)a) * s / (double)n);
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    if (n_samples <= 0) return;

    static int numSMs = 0;
    static double* partials = nullptr;
    static int cap = 0;
    if (!numSMs) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
        if (numSMs <= 0) numSMs = 80;
    }

    int blocks = numSMs * 8;
    int maxUseful = (n_samples + THREADS * 4 - 1) / (THREADS * 4);
    if (blocks > maxUseful) blocks = maxUseful;
    if (blocks < 1) blocks = 1;

    if (cap < blocks) {
        if (partials) cudaFree(partials);
        cap = numSMs * 8;
        cudaMalloc(&partials, (size_t)cap * sizeof(double));
    }

    partial_sum_kernel<<<blocks, THREADS>>>(y_samples, n_samples, partials);
    final_kernel<<<1, THREADS>>>(partials, blocks, result, a, b, n_samples);
    cudaDeviceSynchronize();
}
