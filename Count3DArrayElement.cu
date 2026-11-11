#include <cuda_runtime.h>
#include <stdint.h>

#define THREADS 256

__device__ __forceinline__ int block_sum_i(int v) {
    __shared__ int sh[THREADS / 32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < THREADS / 32) ? sh[threadIdx.x] : 0;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;   // valid in thread 0
}

__global__ void __launch_bounds__(THREADS)
count_kernel(const int* __restrict__ in, int n, int P, int* __restrict__ partials) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Scalar head until 16-byte aligned, int4 body, scalar tail.
    int head = (int)(((16 - ((uintptr_t)in & 15)) & 15) >> 2);
    if (head > n) head = n;
    int c = 0;
    for (int i = tid; i < head; i += stride) c += (in[i] == P);

    const int* a = in + head;
    const int m = n - head;
    const int m4 = m >> 2;
    const int4* a4 = reinterpret_cast<const int4*>(a);
    for (int i = tid; i < m4; i += stride) {
        int4 v = __ldg(a4 + i);
        c += (v.x == P) + (v.y == P) + (v.z == P) + (v.w == P);
    }
    for (int i = (m4 << 2) + tid; i < m; i += stride) c += (a[i] == P);

    c = block_sum_i(c);
    if (threadIdx.x == 0) partials[blockIdx.x] = c;
}

__global__ void __launch_bounds__(THREADS)
final_kernel(const int* __restrict__ partials, int numPartials, int* __restrict__ output) {
    int s = 0;
    for (int i = threadIdx.x; i < numPartials; i += THREADS) s += partials[i];
    s = block_sum_i(s);
    if (threadIdx.x == 0) *output = s;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {
    // Max 1000^3 = 1e9 < 2^31, so int is safe; compute in 64-bit to be careful.
    const int total = (int)((long long)N * M * K);
    if (total <= 0) return;

    static int numSMs = 0;
    static int* partials = nullptr;
    static int cap = 0;
    if (!numSMs) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
        if (numSMs <= 0) numSMs = 80;
    }

    int blocks = numSMs * 8;
    int maxUseful = (total + THREADS * 4 - 1) / (THREADS * 4);
    if (blocks > maxUseful) blocks = maxUseful;
    if (blocks < 1) blocks = 1;

    if (cap < blocks) {
        if (partials) cudaFree(partials);
        cap = numSMs * 8;
        cudaMalloc(&partials, (size_t)cap * sizeof(int));
    }

    count_kernel<<<blocks, THREADS>>>(input, total, P, partials);
    final_kernel<<<1, THREADS>>>(partials, blocks, output);
    cudaDeviceSynchronize();
}
