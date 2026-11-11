#include <cuda_runtime.h>

#define THREADS 256
#define WARPS (THREADS / 32)

__device__ __forceinline__ int block_sum(int v) {
    __shared__ int sh[WARPS];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < WARPS) ? sh[threadIdx.x] : 0;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;   // valid in thread 0
}

// Each warp sums one (depth,row) segment input[d][r][S_COL..E_COL] at a time.
// Lanes stride across columns, so every row read is coalesced.
__global__ void __launch_bounds__(THREADS)
subarray_kernel(const int* __restrict__ in, int* __restrict__ partials,
                int M, int K, int S_DEP, int S_ROW, int nRows, int S_COL, int width,
                int numSegs) {
    const int lane = threadIdx.x & 31;
    const int warpGlobal = blockIdx.x * WARPS + (threadIdx.x >> 5);
    const int warpStride = gridDim.x * WARPS;

    int s = 0;
    for (int seg = warpGlobal; seg < numSegs; seg += warpStride) {
        int dd = S_DEP + seg / nRows;
        int rr = S_ROW + seg % nRows;
        const int* row = in + ((size_t)dd * M + rr) * K + S_COL;
        for (int c = lane; c < width; c += 32) s += __ldg(row + c);
    }

    s = block_sum(s);
    if (threadIdx.x == 0) partials[blockIdx.x] = s;
}

__global__ void __launch_bounds__(THREADS)
final_kernel(const int* __restrict__ partials, int n, int* __restrict__ output) {
    int s = 0;
    for (int i = threadIdx.x; i < n; i += THREADS) s += partials[i];
    s = block_sum(s);
    if (threadIdx.x == 0) *output = s;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP,
                      int S_ROW, int E_ROW, int S_COL, int E_COL) {
    const int nDeps   = E_DEP - S_DEP + 1;
    const int nRows   = E_ROW - S_ROW + 1;
    const int width   = E_COL - S_COL + 1;
    const int numSegs = nDeps * nRows;      // <= 250,000

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
    int maxUseful = (numSegs + WARPS - 1) / WARPS;
    if (blocks > maxUseful) blocks = maxUseful;
    if (blocks < 1) blocks = 1;

    if (cap < blocks) {
        if (partials) cudaFree(partials);
        cap = numSMs * 8;
        cudaMalloc(&partials, (size_t)cap * sizeof(int));
    }

    subarray_kernel<<<blocks, THREADS>>>(input, partials, M, K, S_DEP, S_ROW, nRows,
                                         S_COL, width, numSegs);
    final_kernel<<<1, THREADS>>>(partials, blocks, output);
    cudaDeviceSynchronize();
}
