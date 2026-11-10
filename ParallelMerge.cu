#include <cuda_runtime.h>

#define THREADS 256
#define IPT 16
#define TILE (THREADS * IPT)          // 4096 outputs per block
#define PAD(x) ((x) + ((x) >> 5))     // avoid bank conflicts on staged stores

// Number of elements taken from A among the first d outputs (A wins ties).
__device__ __forceinline__ int merge_path(const float* A, int M, const float* B, int N, int d) {
    int lo = max(0, d - N), hi = min(d, M);
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (A[mid] <= B[d - 1 - mid]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void partition_kernel(const float* __restrict__ A, int M,
                                 const float* __restrict__ B, int N,
                                 int* __restrict__ splits, int numTiles) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t > numTiles) return;
    int total = M + N;
    int d = min(t * TILE, total);
    splits[t] = merge_path(A, M, B, N, d);
}

__global__ void __launch_bounds__(THREADS)
merge_kernel(const float* __restrict__ A, int M,
             const float* __restrict__ B, int N,
             float* __restrict__ C, const int* __restrict__ splits) {
    __shared__ float sh[PAD(TILE) + 32];

    const int tile = blockIdx.x;
    const int total = M + N;
    const int d0 = tile * TILE;
    const int d1 = min(d0 + TILE, total);
    const int a0 = splits[tile], a1 = splits[tile + 1];
    const int b0 = d0 - a0,      b1 = d1 - a1;
    const int aLen = a1 - a0, bLen = b1 - b0;
    const int len = aLen + bLen;
    const int tid = threadIdx.x;

    // Load A slice then B slice contiguously into shared memory.
    for (int i = tid; i < len; i += THREADS)
        sh[i] = (i < aLen) ? A[a0 + i] : B[b0 + i - aLen];
    __syncthreads();

    const float* sA = sh;
    const float* sB = sh + aLen;

    // Per-thread merge path inside the tile, then sequential merge.
    float r[IPT];
    int dLocal = min(tid * IPT, len);
    int i = merge_path(sA, aLen, sB, bLen, dLocal);
    int j = dLocal - i;
    int cnt = min(IPT, len - dLocal);
    #pragma unroll
    for (int k = 0; k < IPT; ++k) {
        if (k < cnt) {
            bool takeA = (j >= bLen) || (i < aLen && sA[i] <= sB[j]);
            r[k] = takeA ? sA[i] : sB[j];
            i += takeA; j += !takeA;
        }
    }
    __syncthreads();

    // Stage through shared memory for coalesced global stores.
    #pragma unroll
    for (int k = 0; k < IPT; ++k)
        if (k < cnt) sh[PAD(tid * IPT + k)] = r[k];
    __syncthreads();
    for (int x = tid; x < len; x += THREADS)
        C[d0 + x] = sh[PAD(x)];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    int total = M + N;
    if (total <= 0) return;
    int numTiles = (total + TILE - 1) / TILE;

    static int* splits = nullptr;
    static int cap = 0;
    if (cap < numTiles + 1) {
        if (splits) cudaFree(splits);
        cap = numTiles + 1;
        cudaMalloc(&splits, (size_t)cap * sizeof(int));
    }

    partition_kernel<<<(numTiles + 1 + 255) / 256, 256>>>(A, M, B, N, splits, numTiles);
    merge_kernel<<<numTiles, THREADS>>>(A, M, B, N, C, splits);
    cudaDeviceSynchronize();
}
