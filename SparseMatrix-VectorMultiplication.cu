#include <cuda_runtime.h>

// y = A * x, with A given as a dense M x N row-major array (60-70% zeros).
//
// Because A arrives dense, every element must be read anyway to know whether
// it is zero, so a CSR conversion can't save bandwidth — the fastest thing is
// a bandwidth-bound dense GEMV that streams A exactly once:
//   one 256-thread block per row, float4 streaming loads of A (4 in flight per
//   thread), x read through the read-only cache (40 KB, L1/L2 resident),
//   warp-shuffle + smem reduction, one store per row.

#define THREADS 256
#define UNROLL  4

template <bool VEC4>
__global__ void __launch_bounds__(THREADS)
gemv_rows(const float* __restrict__ A, const float* __restrict__ x,
          float* __restrict__ y, int M, int N)
{
    const int row = blockIdx.x;
    const float* a = A + (size_t)row * N;
    float sum = 0.f;

    if (VEC4) {
        // N % 4 == 0 -> every row start and x are 16-byte aligned
        const float4* a4 = reinterpret_cast<const float4*>(a);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const int N4 = N >> 2;
        int i = threadIdx.x;
        for (; i + (UNROLL - 1) * THREADS < N4; i += UNROLL * THREADS) {
            float4 av[UNROLL], xv[UNROLL];
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                av[u] = __ldcs(a4 + i + u * THREADS);
                xv[u] = __ldg (x4 + i + u * THREADS);
            }
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u)
                sum = fmaf(av[u].x, xv[u].x, fmaf(av[u].y, xv[u].y,
                      fmaf(av[u].z, xv[u].z, fmaf(av[u].w, xv[u].w, sum))));
        }
        for (; i < N4; i += THREADS) {
            const float4 av = __ldcs(a4 + i), xv = __ldg(x4 + i);
            sum = fmaf(av.x, xv.x, fmaf(av.y, xv.y, fmaf(av.z, xv.z, fmaf(av.w, xv.w, sum))));
        }
    } else {
        for (int k = threadIdx.x; k < N; k += THREADS)
            sum = fmaf(__ldcs(a + k), __ldg(x + k), sum);
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
        if (lane == 0) y[row] = sum;
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz)
{
    (void)nnz;   // not needed: A is dense storage
    if ((N & 3) == 0) gemv_rows<true ><<<M, THREADS>>>(A, x, y, M, N);
    else              gemv_rows<false><<<M, THREADS>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
