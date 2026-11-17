/*
FP8 Block-Scaled GEMM (DeepSeek-V3 / DeepGEMM style)
x_q: (M, K) E4M3, x_scales: (M, K/128) f32; w_q: (N, K) E4M3, w_scales: (N, K/128) f32
  x[m,k] = e4m3(x_q[m,k]) * x_scales[m, k/128],  w[n,k] = e4m3(w_q[n,k]) * w_scales[n, k/128]
  y[m,n] = sum_k x[m,k] * w[n,k]   (stored as float16)

Constraints
 - 1 ≤ M, N ≤ 16384; 128 ≤ K ≤ 16384, K % 128 == 0; scales in [0.01, 4]
 - Performance is measured with M = 2048, N = 4096, K = 4096
*/

/*
Approach: tensor cores on exactly decoded FP8, with per-block scale promotion.

 - Decode: every E4M3 value is exactly representable in fp16. Shifting the 7
   magnitude bits left by 7 lands them in fp16's exponent/mantissa fields with
   bias 15 instead of 7, and multiplying by 2^8 fixes the bias. That's exact,
   subnormals included (m * 2^-17 * 256 = m * 2^-9). Two bytes become one half2
   with a handful of integer ops and one __hmul2.
 - MMA: mma.sync.m16n8k8 f16 x f16 -> f32 (sm_75+). Products of fp16 values are
   exact in fp32.
 - Scales (DeepGEMM-style promotion): for each 128-wide K block, the raw products
   go into a temporary fp32 accumulator. At the block's end,
   acc += tmp * x_scale[row, kb] * w_scale[col, kb], using the known m16n8
   accumulator layout (rows lane/4 and lane/4 + 8, cols 2*(lane%4) + {0, 1}).
   The scales are applied once per 128 K, in fp32, exactly as defined.
 - Tiling: block 128 x 128, K-step 32, 256 threads = 8 warps (2 x 4), warp tile
   64 x 32 = 4 x 4 MMAs per k8. Each thread loads 16 raw bytes of x and of w per
   K-step (uint4), prefetched into registers one step ahead. They're decoded
   into double-buffered shared memory As/Bs[128][40] halves (40 KB static).
   The padding makes the 32-bit fragment loads conflict-free.
 - Edges: rows ≥ M or columns ≥ N are zero-filled and never stored. K is a
   multiple of 128, so every K-step and scale block is full.
At the benchmark: 32 x 16 = 512 blocks, 68.7 GFLOP on tensor cores.
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BM 128
#define BN 128
#define BK 32
#define LDS (BK + 8)                 // halves per smem row (80 B)
#define THREADS 256
#define QBLK 128                     // scale block along K
#define KT_PER_QBLK (QBLK / BK)      // 4

// 4 E4M3 bytes -> 4 fp16 values (two half2), exact.
__device__ __forceinline__ void e4m3x4_to_half(uint32_t w, uint32_t& lo, uint32_t& hi)
{
    const uint32_t a = ((w & 0x00000080u) << 8) | ((w & 0x0000007Fu) << 7) |
                       ((w & 0x00008000u) << 16) | ((w & 0x00007F00u) << 15);
    const uint32_t v = w >> 16;
    const uint32_t b = ((v & 0x00000080u) << 8) | ((v & 0x0000007Fu) << 7) |
                       ((v & 0x00008000u) << 16) | ((v & 0x00007F00u) << 15);
    const __half2 s = __float2half2_rn(256.0f);
    __half2 ha = __hmul2(*reinterpret_cast<const __half2*>(&a), s);
    __half2 hb = __hmul2(*reinterpret_cast<const __half2*>(&b), s);
    lo = *reinterpret_cast<uint32_t*>(&ha);
    hi = *reinterpret_cast<uint32_t*>(&hb);
}

// 16 raw bytes -> 16 halves written to dst (32 B, 16 B aligned)
__device__ __forceinline__ void decode_store(uint4 raw, __half* dst)
{
    uint4 o0, o1;
    e4m3x4_to_half(raw.x, o0.x, o0.y);
    e4m3x4_to_half(raw.y, o0.z, o0.w);
    e4m3x4_to_half(raw.z, o1.x, o1.y);
    e4m3x4_to_half(raw.w, o1.z, o1.w);
    reinterpret_cast<uint4*>(dst)[0] = o0;
    reinterpret_cast<uint4*>(dst)[1] = o1;
}

__device__ __forceinline__ void mma_16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

__global__ void __launch_bounds__(THREADS, 1)
fp8_block_gemm(const uint8_t* __restrict__ xq, const float* __restrict__ xs,
               const uint8_t* __restrict__ wq, const float* __restrict__ ws,
               __half* __restrict__ y, int M, int N, int K)
{
    __shared__ __align__(16) __half As[2][BM][LDS];
    __shared__ __align__(16) __half Bs[2][BN][LDS];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;          // 2 x 4 warps
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int KB = K / QBLK;                          // scale blocks per row
    const int KT = K / BK;

    // load assignment: row = tid/2, 16-byte half of the 32-byte K-step
    const int lr = tid >> 1, lp = (tid & 1) * 16;
    const bool aOk = (m0 + lr) < M, bOk = (n0 + lr) < N;
    const uint8_t* aSrc = xq + (size_t)(m0 + lr) * K + lp;
    const uint8_t* bSrc = wq + (size_t)(n0 + lr) * K + lp;
    const uint4 zero4 = make_uint4(0, 0, 0, 0);

    float acc[4][4][4], tmp[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) { acc[i][j][e] = 0.f; tmp[i][j][e] = 0.f; }

    // first K-step
    uint4 ra = aOk ? __ldg(reinterpret_cast<const uint4*>(aSrc)) : zero4;
    uint4 rb = bOk ? __ldg(reinterpret_cast<const uint4*>(bSrc)) : zero4;
    decode_store(ra, &As[0][lr][lp]);
    decode_store(rb, &Bs[0][lr][lp]);
    __syncthreads();

    const int fr = lane >> 2;            // fragment row (A) / col (B) within 8
    const int fk = (lane & 3) * 2;       // fragment k pair

    for (int kt = 0; kt < KT; ++kt) {
        const int cur = kt & 1;
        const bool more = kt + 1 < KT;
        if (more) {
            const size_t off = (size_t)(kt + 1) * BK;
            ra = aOk ? __ldg(reinterpret_cast<const uint4*>(aSrc + off)) : zero4;
            rb = bOk ? __ldg(reinterpret_cast<const uint4*>(bSrc + off)) : zero4;
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            uint32_t af[4][2], bf[4];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int r = wm * 64 + mi * 16 + fr;
                af[mi][0] = *reinterpret_cast<const uint32_t*>(&As[cur][r][kk + fk]);
                af[mi][1] = *reinterpret_cast<const uint32_t*>(&As[cur][r + 8][kk + fk]);
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int c = wn * 32 + ni * 8 + fr;
                bf[ni] = *reinterpret_cast<const uint32_t*>(&Bs[cur][c][kk + fk]);
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma_16816(tmp[mi][ni], af[mi][0], af[mi][1], bf[ni]);
        }

        if (more) {                                   // other buffer is idle
            decode_store(ra, &As[cur ^ 1][lr][lp]);
            decode_store(rb, &Bs[cur ^ 1][lr][lp]);
        }

        // ---- end of a 128-wide scale block: promote tmp into acc with scales
        if ((kt + 1) % KT_PER_QBLK == 0) {
            const int kb = kt / KT_PER_QBLK;
            float sx[4][2], sw[4][2];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = m0 + wm * 64 + mi * 16 + fr + h * 8;
                    sx[mi][h] = (r < M) ? __ldg(xs + (size_t)r * KB + kb) : 0.f;
                }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int c = n0 + wn * 32 + ni * 8 + fk + h;
                    sw[ni][h] = (c < N) ? __ldg(ws + (size_t)c * KB + kb) : 0.f;
                }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    // c0,c1: row fr, cols fk, fk+1 ; c2,c3: row fr+8
                    acc[mi][ni][0] = fmaf(tmp[mi][ni][0], sx[mi][0] * sw[ni][0], acc[mi][ni][0]);
                    acc[mi][ni][1] = fmaf(tmp[mi][ni][1], sx[mi][0] * sw[ni][1], acc[mi][ni][1]);
                    acc[mi][ni][2] = fmaf(tmp[mi][ni][2], sx[mi][1] * sw[ni][0], acc[mi][ni][2]);
                    acc[mi][ni][3] = fmaf(tmp[mi][ni][3], sx[mi][1] * sw[ni][1], acc[mi][ni][3]);
                    tmp[mi][ni][0] = tmp[mi][ni][1] = tmp[mi][ni][2] = tmp[mi][ni][3] = 0.f;
                }
        }
        __syncthreads();
    }

    // ---- store fp16
    const bool evenN = (N & 1) == 0;
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m0 + wm * 64 + mi * 16 + fr + h * 8;
            if (r >= M) continue;
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int c = n0 + wn * 32 + ni * 8 + fk;
                const float v0 = acc[mi][ni][2 * h], v1 = acc[mi][ni][2 * h + 1];
                __half* dst = y + (size_t)r * N + c;
                if (evenN && c + 1 < N) {
                    *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(v0, v1);
                } else {
                    if (c < N)     dst[0] = __float2half_rn(v0);
                    if (c + 1 < N) dst[1] = __float2half_rn(v1);
                }
            }
        }
}

// x_q, x_scales, w_q, w_scales, y are device pointers
extern "C" void solve(const uint8_t* x_q, const float* x_scales, const uint8_t* w_q,
                      const float* w_scales, __half* y, int M, int N, int K)
{
    if (M <= 0 || N <= 0 || K <= 0) return;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    fp8_block_gemm<<<grid, THREADS>>>(x_q, x_scales, w_q, w_scales, y, M, N, K);
    cudaDeviceSynchronize();
}
