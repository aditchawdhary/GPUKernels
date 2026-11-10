#include <cuda_runtime.h>

// mse = (1/N) * sum_i (predictions[i] - targets[i])^2
//
// Memory-bound: 400 MB read at N = 50M.
// Kernel 1: one resident wave of blocks, grid-stride float4 loads (4 loads of
//           each array in flight per thread), per-thread float sums (~200
//           terms each), warp-shuffle + smem block reduce -> one partial/block.
// Kernel 2: a single block sums the ~1k partials in double and divides by N.
// Deterministic (no float atomics), and the double final sum keeps the large
// total (up to ~4e14) accurate.

#define THREADS 256
#define UNROLL  4

__device__ __forceinline__ float sq4(float4 p, float4 t)
{
    const float a = p.x - t.x, b = p.y - t.y, c = p.z - t.z, d = p.w - t.w;
    return fmaf(a, a, fmaf(b, b, fmaf(c, c, d * d)));
}

__global__ void __launch_bounds__(THREADS)
mse_partial(const float* __restrict__ pred, const float* __restrict__ tgt,
            float* __restrict__ partial, int N)
{
    const int tid    = blockIdx.x * THREADS + threadIdx.x;
    const int stride = gridDim.x * THREADS;
    const int N4     = N >> 2;
    const float4* p4 = reinterpret_cast<const float4*>(pred);
    const float4* t4 = reinterpret_cast<const float4*>(tgt);

    float sum = 0.f;
    int i = tid;
    for (; i + (UNROLL - 1) * stride < N4; i += UNROLL * stride) {
        float4 p[UNROLL], t[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            p[u] = __ldcs(p4 + i + u * stride);
            t[u] = __ldcs(t4 + i + u * stride);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) sum += sq4(p[u], t[u]);
    }
    for (; i < N4; i += stride) sum += sq4(__ldcs(p4 + i), __ldcs(t4 + i));

    // scalar tail (N % 4 elements)
    if (tid < (N & 3)) {
        const int k = (N4 << 2) + tid;
        const float d = pred[k] - tgt[k];
        sum = fmaf(d, d, sum);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);

    __shared__ float ws[THREADS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < THREADS / 32) ? ws[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0) partial[blockIdx.x] = sum;
    }
}

__global__ void __launch_bounds__(1024)
mse_final(const float* __restrict__ partial, int G, float* __restrict__ mse, int N)
{
    double s = 0.0;
    for (int i = threadIdx.x; i < G; i += 1024) s += (double)partial[i];

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);

    __shared__ double ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) ws[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = ws[lane];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
        if (lane == 0) *mse = (float)(s / (double)N);
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N)
{
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mse_partial, THREADS, 0);

    int G = sms * occ;                                   // one resident wave
    const int need = ((N >> 2) + THREADS - 1) / THREADS; // no more blocks than work
    if (G > need) G = need;
    if (G < 1) G = 1;

    static float* partial = nullptr;
    static int cap = 0;
    if (G > cap) {
        if (partial) cudaFree(partial);
        cudaMalloc(&partial, G * sizeof(float));
        cap = G;
    }

    mse_partial<<<G, THREADS>>>(predictions, targets, partial, N);
    mse_final<<<1, 1024>>>(partial, G, mse, N);
    cudaDeviceSynchronize();
}
