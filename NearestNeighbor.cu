/*
## Nearest Neighbor

Implement a GPU program that, for N three-dimensional points stored on the device, fills indices[i] with the index j ≠ i of the point closest to points[i]. 
Comparing squared Euclidean distance is sufficient—you do not need to compute square-roots.

### Implementation Requirements
 - The solve function signature must remain unchanged
 - External libraries are not permitted
 - The final result must be stored in the indices array

#### Example 1:
```
Input:  points  = [(0,0,0), (1,0,0), (5,5,5)]
        indices = [-1, -1, -1]
        N       = 3
Output: indices = [1, 0, 1]   # 0⇆1 are nearest, 2 is closest to 1
```

#### Constraints
 - 1 ≤ N ≤ 100,000
 - Coordinates are 32-bit floats in the range [-1000, 1000]
 - Performance is measured with N = 10,000
*/


#include <cuda_runtime.h>

// For each point i: indices[i] = argmin_{j != i} |p_i - p_j|^2 (smallest j on ties).
//
// Brute force, O(N^2) — 1e8 pairs at N = 10k, which is tiny for a GPU; the
// challenge is filling the machine, since N = 10k is only ~40 blocks of queries.
// So the candidate range is also split across gridDim.y chunks:
//   block (x, y): 256 queries x one chunk of candidates, staged through smem.
// Each thread keeps its best (dist, j) for its chunk, then merges globally
// with a 64-bit atomicMin on the packed key (dist_bits << 32 | j). Distances
// are >= 0, so their float bit patterns order like the floats themselves, and
// equal distances fall back to the smaller index.

#define THREADS 256
#define TILE    256

__global__ void init_best(unsigned long long* best, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) best[i] = ~0ull;
}

__global__ void __launch_bounds__(THREADS)
nn_kernel(const float* __restrict__ pts, unsigned long long* __restrict__ best, int N, int chunk)
{
    __shared__ float sx[TILE], sy[TILE], sz[TILE];

    const int i  = blockIdx.x * THREADS + threadIdx.x;
    const int j0 = blockIdx.y * chunk;
    const int j1 = min(N, j0 + chunk);

    float qx = 0.f, qy = 0.f, qz = 0.f;
    if (i < N) { qx = pts[3 * i]; qy = pts[3 * i + 1]; qz = pts[3 * i + 2]; }

    float bd = __int_as_float(0x7f800000);   // +inf
    int   bj = -1;

    for (int t0 = j0; t0 < j1; t0 += TILE) {
        const int cnt = min(TILE, j1 - t0);
        __syncthreads();
        for (int k = threadIdx.x; k < cnt; k += THREADS) {
            const int j = t0 + k;
            sx[k] = pts[3 * j]; sy[k] = pts[3 * j + 1]; sz[k] = pts[3 * j + 2];
        }
        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < cnt; ++k) {
            const float dx = qx - sx[k], dy = qy - sy[k], dz = qz - sz[k];   // smem broadcast
            const float d  = dx * dx + dy * dy + dz * dz;
            const int   j  = t0 + k;
            if (d < bd && j != i) { bd = d; bj = j; }   // j ascending -> ties keep smaller j
        }
    }

    if (i < N && bj >= 0) {
        const unsigned long long key =
            ((unsigned long long)__float_as_uint(bd) << 32) | (unsigned)bj;
        atomicMin(best + i, key);
    }
}

__global__ void finalize(const unsigned long long* __restrict__ best, int* __restrict__ indices, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const unsigned long long k = best[i];
        indices[i] = (k == ~0ull) ? -1 : (int)(unsigned)(k & 0xffffffffu);   // -1 only when N == 1
    }
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N)
{
    static unsigned long long* best = nullptr;
    static int cap = 0;
    if (N > cap) {
        if (best) cudaFree(best);
        cudaMalloc(&best, (size_t)N * sizeof(unsigned long long));
        cap = N;
    }

    int dev = 0, sms = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    // Split candidates so there are ~8 blocks per SM, but keep chunks >= 512
    // points so the atomics and smem fills stay negligible.
    const int qblocks = (N + THREADS - 1) / THREADS;
    int S = (8 * sms + qblocks - 1) / qblocks;
    const int maxS = (N + 511) / 512;
    if (S > maxS) S = maxS;
    if (S < 1) S = 1;
    int chunk = (N + S - 1) / S;
    S = (N + chunk - 1) / chunk;

    const int eb = (N + 255) / 256;
    init_best<<<eb, 256>>>(best, N);
    nn_kernel<<<dim3(qblocks, S), THREADS>>>(points, best, N, chunk);
    finalize<<<eb, 256>>>(best, indices, N);
    cudaDeviceSynchronize();
}
