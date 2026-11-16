#include <cuda_runtime.h>

// K-means (2-D) with exactly max_iterations rounds of
//   labels[i] = argmin_c (x_i - cx_c)^2 + (y_i - cy_c)^2      (first minimum wins)
//   centroid_c = mean of its points, only if the cluster is non-empty.
// Labels returned are those of the last round (assigned before the last update), as in the
// reference. Distances use explicit round-to-nearest mul/add (no FMA contraction) so they are
// bit-identical to the reference's dx**2 + dy**2 and near-ties resolve the same way.
// Cluster sums are accumulated in double, so the means match the reference to ~1e-7.
//
// Fast path (k <= 8 and n <= 65536; the benchmark is n = 10000, k = 5): one persistent block
// runs every iteration. Each thread keeps per-cluster sums in registers, then a block reduction
// updates the centroids in shared memory. One launch instead of two per iteration.
// General path: per iteration, an assign kernel (centroids in shared memory, shared-memory
// accumulators flushed to global with atomics) and a small update kernel.

#define FAST_T 512
#define FAST_K 8
#define FAST_N 65536
#define GEN_T 256

__device__ __forceinline__ float dist2(float x, float y, float cx, float cy) {
    float dx = __fsub_rn(x, cx), dy = __fsub_rn(y, cy);
    return __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
}

__global__ void __launch_bounds__(FAST_T)
kmeans_fast(const float* __restrict__ X, const float* __restrict__ Y, int* __restrict__ labels,
            const float* __restrict__ icx, const float* __restrict__ icy,
            float* __restrict__ fcx, float* __restrict__ fcy, int n, int k, int iters) {
    __shared__ float cx[FAST_K], cy[FAST_K];
    __shared__ double red_x[FAST_T / 32][FAST_K], red_y[FAST_T / 32][FAST_K];
    __shared__ int red_c[FAST_T / 32][FAST_K];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    if (tid < k) { cx[tid] = icx[tid]; cy[tid] = icy[tid]; }
    __syncthreads();

    for (int it = 0; it < iters; it++) {
        double sx[FAST_K], sy[FAST_K];
        int cnt[FAST_K];
#pragma unroll
        for (int c = 0; c < FAST_K; c++) { sx[c] = 0.0; sy[c] = 0.0; cnt[c] = 0; }

        for (int i = tid; i < n; i += FAST_T) {
            const float x = X[i], y = Y[i];
            int best = 0;
            float bd = dist2(x, y, cx[0], cy[0]);
            for (int c = 1; c < k; c++) {
                float d = dist2(x, y, cx[c], cy[c]);
                if (d < bd) { bd = d; best = c; }
            }
            if (it == iters - 1) labels[i] = best;
#pragma unroll
            for (int c = 0; c < FAST_K; c++)
                if (best == c) { sx[c] += (double)x; sy[c] += (double)y; cnt[c]++; }
        }

        // block reduction of the k (sum_x, sum_y, count) triples
#pragma unroll
        for (int c = 0; c < FAST_K; c++) {
            double a = sx[c], b = sy[c];
            int m = cnt[c];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                a += __shfl_xor_sync(0xffffffffu, a, o);
                b += __shfl_xor_sync(0xffffffffu, b, o);
                m += __shfl_xor_sync(0xffffffffu, m, o);
            }
            if (lane == 0) { red_x[warp][c] = a; red_y[warp][c] = b; red_c[warp][c] = m; }
        }
        __syncthreads();
        if (tid < k) {
            double a = 0.0, b = 0.0;
            int m = 0;
            for (int w = 0; w < FAST_T / 32; w++) { a += red_x[w][tid]; b += red_y[w][tid]; m += red_c[w][tid]; }
            if (m > 0) { cx[tid] = (float)(a / m); cy[tid] = (float)(b / m); }
        }
        __syncthreads();
    }
    if (tid < k) { fcx[tid] = cx[tid]; fcy[tid] = cy[tid]; }
}

// ---------------------------------------------------------------- general path
__global__ void __launch_bounds__(GEN_T)
kmeans_assign(const float* __restrict__ X, const float* __restrict__ Y, int* __restrict__ labels,
              const float* __restrict__ fcx, const float* __restrict__ fcy,
              double* __restrict__ gsx, double* __restrict__ gsy, int* __restrict__ gcnt,
              int n, int k) {
    extern __shared__ __align__(8) unsigned char smem[];
    double* ssx = (double*)smem;                 // [k]
    double* ssy = ssx + k;                       // [k]
    float* scx = (float*)(ssy + k);              // [k]
    float* scy = scx + k;                        // [k]
    int* scnt = (int*)(scy + k);                 // [k]

    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        scx[c] = fcx[c]; scy[c] = fcy[c];
        ssx[c] = 0.0; ssy[c] = 0.0; scnt[c] = 0;
    }
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float x = X[i], y = Y[i];
        int best = 0;
        float bd = dist2(x, y, scx[0], scy[0]);
        for (int c = 1; c < k; c++) {
            float d = dist2(x, y, scx[c], scy[c]);
            if (d < bd) { bd = d; best = c; }
        }
        labels[i] = best;
        atomicAdd(&ssx[best], (double)x);
        atomicAdd(&ssy[best], (double)y);
        atomicAdd(&scnt[best], 1);
    }
    __syncthreads();

    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        if (scnt[c]) {
            atomicAdd(&gsx[c], ssx[c]);
            atomicAdd(&gsy[c], ssy[c]);
            atomicAdd(&gcnt[c], scnt[c]);
        }
    }
}

// apply the means and reset the accumulators for the next round
__global__ void kmeans_update(float* __restrict__ fcx, float* __restrict__ fcy,
                              double* __restrict__ gsx, double* __restrict__ gsy,
                              int* __restrict__ gcnt, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    int m = gcnt[c];
    if (m > 0) { fcx[c] = (float)(gsx[c] / m); fcy[c] = (float)(gsy[c] / m); }
    gsx[c] = 0.0; gsy[c] = 0.0; gcnt[c] = 0;
}

static char* g_buf = nullptr;
static size_t g_bytes = 0;

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {
    const int n = sample_size;
    if (k <= FAST_K && n <= FAST_N) {
        kmeans_fast<<<1, FAST_T>>>(data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
                                   final_centroid_x, final_centroid_y, n, k, max_iterations);
        return;
    }

    size_t need = (size_t)k * (2 * sizeof(double) + sizeof(int));
    if (need > g_bytes) {
        if (g_buf) cudaFree(g_buf);
        cudaMalloc(&g_buf, need);
        g_bytes = need;
    }
    double* gsx = (double*)g_buf;
    double* gsy = gsx + k;
    int* gcnt = (int*)(gsy + k);
    cudaMemsetAsync(g_buf, 0, need);
    cudaMemcpyAsync(final_centroid_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(final_centroid_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);

    int blocks = (n + GEN_T - 1) / GEN_T;
    if (blocks > 1024) blocks = 1024;
    size_t smem = (size_t)k * (2 * sizeof(double) + 2 * sizeof(float) + sizeof(int));
    for (int it = 0; it < max_iterations; it++) {
        kmeans_assign<<<blocks, GEN_T, smem>>>(data_x, data_y, labels, final_centroid_x,
                                               final_centroid_y, gsx, gsy, gcnt, n, k);
        kmeans_update<<<(k + 255) / 256, 256>>>(final_centroid_x, final_centroid_y, gsx, gsy, gcnt, k);
    }
}
