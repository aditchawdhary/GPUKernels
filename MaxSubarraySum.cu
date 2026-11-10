#include <cuda_runtime.h>
#include <limits.h>

#define T 1024

__device__ __forceinline__ int warp_incl_scan(int v) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    return v;
}

__device__ __forceinline__ int warp_max(int v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__global__ void __launch_bounds__(T)
max_window_kernel(const int* __restrict__ in, int* P, int* out, int N, int w) {
    __shared__ int warpSums[32];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;

    // 1) Each thread sums its contiguous chunk.
    const int chunk = (N + T - 1) / T;
    const int start = min(tid * chunk, N);
    const int end   = min(start + chunk, N);
    int s = 0;
    for (int j = start; j < end; ++j) s += in[j];

    // 2) Block-wide exclusive scan of chunk sums.
    int incl = warp_incl_scan(s);
    if (lane == 31) warpSums[wid] = incl;
    __syncthreads();
    if (wid == 0) {
        int ws = warpSums[lane];             // T/32 == 32 warps
        int wincl = warp_incl_scan(ws);
        warpSums[lane] = wincl - ws;         // exclusive warp offsets
    }
    __syncthreads();
    int run = warpSums[wid] + incl - s;      // exclusive prefix for this chunk

    // 3) Write prefix array P[0..N], P[j+1] = sum(in[0..j]).
    if (tid == 0) P[0] = 0;
    for (int j = start; j < end; ++j) { run += in[j]; P[j + 1] = run; }
    __syncthreads();

    // 4) Window sums P[i+w] - P[i], block max reduction.
    int best = INT_MIN;
    const int count = N - w + 1;
    for (int i = tid; i < count; i += T) best = max(best, P[i + w] - P[i]);

    best = warp_max(best);
    __syncthreads();                         // warpSums reuse
    if (lane == 0) warpSums[wid] = best;
    __syncthreads();
    if (wid == 0) {
        best = warp_max(warpSums[lane]);
        if (lane == 0) *out = best;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    static int* P = nullptr;
    static int cap = 0;
    if (cap < N + 1) {
        if (P) cudaFree(P);
        cap = N + 1;
        cudaMalloc(&P, (size_t)cap * sizeof(int));
    }
    max_window_kernel<<<1, T>>>(input, P, output, N, window_size);
    cudaDeviceSynchronize();
}
