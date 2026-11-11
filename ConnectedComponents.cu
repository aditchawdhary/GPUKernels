#include <cuda_runtime.h>

#define THREADS 256

// labels[] doubles as the union-find parent array.
// Invariant: a root is only ever linked to a SMALLER root, so the root of
// every tree is the minimum vertex ID in that tree.

__device__ __forceinline__ int find_root(int* parent, int x) {
    volatile int* p = parent;
    int px = p[x];
    while (px != x) {
        int ppx = p[px];
        if (ppx != px) p[x] = ppx;   // path halving (benign race: ppx is an ancestor)
        x = px;
        px = ppx;
    }
    return x;
}

__global__ void init_kernel(int* labels, int N) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
        labels[i] = i;
}

__global__ void union_kernel(const int* __restrict__ src, const int* __restrict__ dst,
                             int* labels, int M) {
    int stride = gridDim.x * blockDim.x;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < M; e += stride) {
        int u = __ldg(&src[e]);
        int v = __ldg(&dst[e]);
        if (u == v) continue;

        int a = find_root(labels, u);
        int b = find_root(labels, v);
        while (a != b) {
            if (a < b) { int t = a; a = b; b = t; }      // a = larger root
            int old = atomicCAS(&labels[a], a, b);        // link larger -> smaller
            if (old == a) break;                          // success
            a = find_root(labels, old);                   // a was linked meanwhile
            b = find_root(labels, b);
        }
    }
}

// Read-only root walk: in the final pass each thread writes only labels[i],
// so no stale path-halving write can overwrite a finished label.
__device__ __forceinline__ int find_root_ro(const int* parent, int x) {
    const volatile int* p = parent;
    int px = p[x];
    while (px != x) { x = px; px = p[x]; }
    return x;
}

__global__ void compress_kernel(int* labels, int N) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
        labels[i] = find_root_ro(labels, i);
}

// src, dst, labels are device pointers
extern "C" void solve(const int* src, const int* dst, int* labels, int N, int M) {
    int dev = 0, sms = 80;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int maxBlocks = sms * 8;

    int bN = (N + THREADS - 1) / THREADS;
    if (bN > maxBlocks) bN = maxBlocks;
    int bM = (M + THREADS - 1) / THREADS;
    if (bM > maxBlocks) bM = maxBlocks;
    if (bM < 1) bM = 1;

    init_kernel<<<bN, THREADS>>>(labels, N);
    if (M > 0) union_kernel<<<bM, THREADS>>>(src, dst, labels, M);
    compress_kernel<<<bN, THREADS>>>(labels, N);
    cudaDeviceSynchronize();
}
