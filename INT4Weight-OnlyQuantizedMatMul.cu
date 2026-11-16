/*
INT4 Weight-Only Quantized MatMul (W4A16)
y[M x N] = x[M x K] * W^T, where W[N x K] is dequantized on the fly from packed INT4:
  byte w_q[n, i] : high nibble -> w[n, 2i], low nibble -> w[n, 2i+1]
  W[n, k] = (nibble - 8) * scales[n, k / group_size]
x, scales, y are float16; w_q is uint8; all row-major.

Constraints
 - 1 ≤ M, N, K ≤ 8,192;  K divisible by 2 and by group_size
 - group_size ∈ {2, 4, 8, 16, 32, 64, 128}
 - Performance is measured with M = N = K = 4,096, group_size = 128
*/

/*
Approach: one fused tensor-core GEMM. The weights are dequantized in the
global -> shared load, so no full fp16 W is ever written to memory.

 - Block tile 128 (M) x 128 (N) x 32 (K), 256 threads = 8 warps, each warp owns a
   64 x 32 sub-tile = 4 x 2 WMMA 16x16x16 fragments with fp32 accumulators.
 - Per K-tile, each thread fetches
     x : 16 halves (2 x 16 B when K % 8 == 0, else half2 loads)
     w : 8 packed bytes = 16 INT4 weights of one row n (one 8 B load when
         K % 16 == 0, else byte loads), unpacked and multiplied by the
         group scale in fp16 (one scale load when group_size ≥ 16).
   Loads for tile k+1 are issued into registers before the MMAs on tile k run
   (register double buffering), so global latency hides behind tensor-core work.
 - Shared layout: As[m][k] (row_major A) and Bs[n][k], which is exactly
   col_major B = W^T, so no transpose is needed. Rows are padded to 40 halves
   to spread banks.
 - Epilogue: each warp stages one 16x16 fp32 fragment at a time in its own
   shared scratch, then converts it to fp16 and does a bounds-checked store, so
   any M, N works.

Out-of-range rows, columns and k are zero-filled, so partial tiles are exact.
Requires sm_70+ (WMMA).
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 32
#define LDS (BK + 8)            // padded smem row length (halves), multiple of 8
#define THREADS 256

struct Prefetch {
    uint4 x[2];                 // 16 halves of x (row r, k = kb + c*16 .. +15)
    uint4 w[2];                 // 16 dequantized halves of W (row n, same k range)
};

__device__ __forceinline__ void fetch(Prefetch& p,
                                      const half* __restrict__ x,
                                      const uint8_t* __restrict__ wq,
                                      const half* __restrict__ scales,
                                      int M, int N, int K, int gs,
                                      int bm, int bn, int kb,
                                      bool vecX, bool vecW)
{
    const int tid = threadIdx.x;
    const int r   = tid >> 1;           // 0..127  tile row (m for x, n for w)
    const int c   = tid & 1;            // which 16-wide half of the 32-wide K tile
    const int k0  = kb + c * 16;

    // ---------------- x
    {
        const int m = bm + r;
        half h[16];
        if (vecX) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int k = k0 + j * 8;
                if (m < M && k < K)
                    p.x[j] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)m * K + k));
                else
                    p.x[j] = make_uint4(0, 0, 0, 0);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int k = k0 + 2 * j;   // K is even, so a half2 is all-in or all-out
                half2 v = __floats2half2_rn(0.f, 0.f);
                if (m < M && k < K)
                    v = __ldg(reinterpret_cast<const half2*>(x + (size_t)m * K + k));
                h[2 * j] = __low2half(v);
                h[2 * j + 1] = __high2half(v);
            }
            p.x[0] = *reinterpret_cast<uint4*>(&h[0]);
            p.x[1] = *reinterpret_cast<uint4*>(&h[8]);
        }
    }

    // ---------------- w (packed int4 -> fp16)
    {
        const int n = bn + r;
        const int Kb = K >> 1;          // bytes per weight row
        const int G  = K / gs;          // groups per row
        uint8_t b[8];
        if (vecW) {                     // K % 16 == 0: 8-byte chunk is all-in or all-out
            uint2 v = make_uint2(0, 0);
            if (n < N && k0 < K)
                v = __ldg(reinterpret_cast<const uint2*>(wq + (size_t)n * Kb + (k0 >> 1)));
            *reinterpret_cast<uint2*>(b) = v;
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int k = k0 + 2 * j;
                b[j] = (n < N && k < K) ? __ldg(wq + (size_t)n * Kb + (k >> 1)) : (uint8_t)0x88;
            }
        }

        half h[16];
        const half* srow = scales + (size_t)min(n, N - 1) * G;
        if (gs >= 16) {                 // k0 is a multiple of 16, so one group covers all 16
            const half s = (k0 < K) ? __ldg(srow + k0 / gs) : __float2half(0.f);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                h[2 * j]     = __hmul(__int2half_rn((int)(b[j] >> 4)  - 8), s);
                h[2 * j + 1] = __hmul(__int2half_rn((int)(b[j] & 15) - 8), s);
            }
        } else {                        // gs is even: both nibbles of a byte share a group
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int k = k0 + 2 * j;
                const half s = (k < K) ? __ldg(srow + k / gs) : __float2half(0.f);
                h[2 * j]     = __hmul(__int2half_rn((int)(b[j] >> 4)  - 8), s);
                h[2 * j + 1] = __hmul(__int2half_rn((int)(b[j] & 15) - 8), s);
            }
        }
        if (n >= N) {                   // padding rows contribute nothing
            #pragma unroll
            for (int j = 0; j < 16; ++j) h[j] = __float2half(0.f);
        }
        p.w[0] = *reinterpret_cast<uint4*>(&h[0]);
        p.w[1] = *reinterpret_cast<uint4*>(&h[8]);
    }
}

__device__ __forceinline__ void commit(const Prefetch& p, half (*As)[LDS], half (*Bs)[LDS])
{
    const int r = threadIdx.x >> 1;
    const int c = (threadIdx.x & 1) * 16;
    *reinterpret_cast<uint4*>(&As[r][c])     = p.x[0];
    *reinterpret_cast<uint4*>(&As[r][c + 8]) = p.x[1];
    *reinterpret_cast<uint4*>(&Bs[r][c])     = p.w[0];
    *reinterpret_cast<uint4*>(&Bs[r][c + 8]) = p.w[1];
}

__global__ void __launch_bounds__(THREADS)
w4a16_gemm(const half* __restrict__ x, const uint8_t* __restrict__ wq,
           const half* __restrict__ scales, half* __restrict__ y,
           int M, int N, int K, int gs, bool vecX, bool vecW)
{
    __shared__ __align__(16) half As[BM][LDS];
    __shared__ __align__(16) half Bs[BN][LDS];
    __shared__ __align__(16) float Cs[THREADS / 32][16 * 16];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = warp >> 2;           // 0..1  -> 64-row slab
    const int wn = warp & 3;            // 0..3  -> 32-col slab
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int KT = (K + BK - 1) / BK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    Prefetch p;
    fetch(p, x, wq, scales, M, N, K, gs, bm, bn, 0, vecX, vecW);
    commit(p, As, Bs);
    __syncthreads();

    for (int kt = 0; kt < KT; ++kt) {
        const bool more = (kt + 1 < KT);
        if (more) fetch(p, x, wq, scales, M, N, K, gs, bm, bn, (kt + 1) * BK, vecX, vecW);

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(af[i], &As[wm * 64 + i * 16][kk], LDS);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[wn * 32 + j * 16][kk], LDS);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }

        if (more) {
            __syncthreads();            // everyone finished reading this tile
            commit(p, As, Bs);
            __syncthreads();            // next tile visible
        }
    }

    // ---------------- epilogue: fp32 fragment -> per-warp smem -> fp16 global
    float* cs = Cs[warp];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(cs, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            const int row0 = bm + wm * 64 + i * 16;
            const int col0 = bn + wn * 32 + j * 16;
            #pragma unroll
            for (int e = lane; e < 256; e += 32) {
                const int rr = row0 + (e >> 4);
                const int cc = col0 + (e & 15);
                if (rr < M && cc < N) y[(size_t)rr * N + cc] = __float2half(cs[e]);
            }
            __syncwarp();
        }
    }
}

// x, w_q, scales, y are device pointers
extern "C" void solve(const half* x, const uint8_t* w_q, const half* scales, half* y,
                      int M, int N, int K, int group_size)
{
    if (M <= 0 || N <= 0 || K <= 0) return;
    const bool vecX = (K % 8) == 0;     // 16 B aligned x rows / chunks
    const bool vecW = (K % 16) == 0;    // 8 B aligned packed-weight rows / chunks

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    w4a16_gemm<<<grid, THREADS>>>(x, w_q, scales, y, M, N, K, group_size, vecX, vecW);
    cudaDeviceSynchronize();
}
