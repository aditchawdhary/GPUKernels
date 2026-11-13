#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// FP16 batched GEMM with FP32 accumulation on tensor cores (WMMA, sm_70+).
//   C[b] = A[b] (M x K) * B[b] (K x N), row-major, results rounded to half.
// Block tile 64x64, K-step 32, 4 warps; each warp owns a 32x32 sub-tile (2x2 WMMA 16x16x16).
// Tiles are staged through shared memory with zero padding, so any M, N, K work.

#define BM 64
#define BN 64
#define BK 32
#define THREADS 128
#define APAD 8      // keep WMMA ldm a multiple of 8 halves and reduce bank conflicts
#define BPAD 8
#define CPAD 4

__device__ __forceinline__ void load_A_tile(half (*As)[BK + APAD], const half* __restrict__ A,
                                            int M, int K, int m0, int k0, bool vecOK) {
    // 64 x 32 halves = 256 chunks of 8; 2 chunks per thread.
    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int e = threadIdx.x + l * THREADS;
        int r = e >> 2, c8 = (e & 3) * 8;
        int gr = m0 + r, gk = k0 + c8;
        half* dst = &As[r][c8];
        if (vecOK && gr < M && gk + 8 <= K) {
            *reinterpret_cast<uint4*>(dst) =
                __ldg(reinterpret_cast<const uint4*>(A + (size_t)gr * K + gk));
        } else {
            #pragma unroll
            for (int u = 0; u < 8; ++u)
                dst[u] = (gr < M && gk + u < K) ? A[(size_t)gr * K + gk + u] : __float2half(0.f);
        }
    }
}

__device__ __forceinline__ void load_B_tile(half (*Bs)[BN + BPAD], const half* __restrict__ B,
                                            int K, int N, int k0, int n0, bool vecOK) {
    // 32 x 64 halves = 256 chunks of 8; 2 chunks per thread.
    #pragma unroll
    for (int l = 0; l < 2; ++l) {
        int e = threadIdx.x + l * THREADS;
        int r = e >> 3, c8 = (e & 7) * 8;
        int gk = k0 + r, gc = n0 + c8;
        half* dst = &Bs[r][c8];
        if (vecOK && gk < K && gc + 8 <= N) {
            *reinterpret_cast<uint4*>(dst) =
                __ldg(reinterpret_cast<const uint4*>(B + (size_t)gk * N + gc));
        } else {
            #pragma unroll
            for (int u = 0; u < 8; ++u)
                dst[u] = (gk < K && gc + u < N) ? B[(size_t)gk * N + gc + u] : __float2half(0.f);
        }
    }
}

__global__ void __launch_bounds__(THREADS)
hgemm_batched_wmma(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
                   int M, int N, int K) {
    __shared__ __align__(32) half  As[BM][BK + APAD];
    __shared__ __align__(32) half  Bs[BK][BN + BPAD];
    __shared__ __align__(32) float Cs[BM][BN + CPAD];

    const int b = blockIdx.z;
    A += (size_t)b * M * K;
    B += (size_t)b * K * N;
    C += (size_t)b * M * N;

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int warp = threadIdx.x >> 5;
    const int wr = (warp >> 1) * 32;     // warp's row offset in the block tile
    const int wc = (warp & 1) * 32;      // warp's col offset

    // 16-byte vector loads need every row start 16B-aligned.
    const bool vecA = (K & 7) == 0;
    const bool vecB = (N & 7) == 0;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        load_A_tile(As, A, M, K, m0, k0, vecA);
        load_B_tile(Bs, B, K, N, k0, n0, vecB);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[wr + 16 * i][kk], BK + APAD);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[kk][wc + 16 * j], BN + BPAD);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    // FP32 accumulators -> shared -> rounded half with bounds checks.
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&Cs[wr + 16 * i][wc + 16 * j], acc[i][j], BN + CPAD,
                                    wmma::mem_row_major);
    __syncthreads();

    for (int e = threadIdx.x; e < BM * BN; e += THREADS) {
        int r = e / BN, c = e % BN;
        int gr = m0 + r, gc = n0 + c;
        if (gr < M && gc < N) C[(size_t)gr * N + gc] = __float2half_rn(Cs[r][c]);
    }
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    hgemm_batched_wmma<<<grid, THREADS>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
