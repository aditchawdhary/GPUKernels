/*
NVFP4 GEMM:  y[m, n] = alpha * sum_k x[m, k] * w[n, k]      (y stored as float16)
  x[m, k] = fp4(x_q nibble) * e4m3(x_scales[m, k/16])
  w[n, k] = fp4(w_q nibble) * e4m3(w_scales[n, k/16])
  Packing: high nibble = even k, low nibble = odd k.

Approach: tensor cores on exactly dequantized FP16.
 - Exactness: an E2M1 value has <= 2 significant bits and an E4M3 scale <= 4, so
   their product has <= 6 significant bits, magnitude <= 6 * 448 = 2688 and
   >= 0.5 * 2^-9 = 2^-10 (a normal fp16). So fp4 * scale is exactly
   representable in fp16, and __hmul2 of the two exact operands rounds nothing.
   Products of fp16 values are exact in fp32, so the MMA result equals the
   definition up to fp32 summation order. Scales are folded in at decode time,
   which avoids per-block accumulator promotion.
 - Decode (8 values from one 32-bit word): the fp16 bit pattern of every E2M1
   magnitude is 0xXX00 with XX in {00,38,3C,3E,40,42,44,46}. __byte_perm looks
   up four high bytes at once from that 8-byte table; a second __byte_perm
   interleaves the even/odd values into half2 words; sign bits are OR'ed in.
 - MMA: mma.sync.m16n8k16 f16 x f16 -> f32 (sm_80+).
 - Tiling: block 128 x 128, K-step 32 (= two 16-value scale blocks), 256 threads
   = 8 warps (2 x 4), warp tile 64 x 32 = 4 x 4 MMAs per k16. Each thread loads
   exactly one 16-value block of x (8 bytes + 1 scale byte) and one of w per
   K-step. Raw bytes for step k+1 are prefetched into registers while step k
   computes, then decoded into double-buffered smem As/Bs[128][40] halves.
   The 40-half row pitch (20 words) makes the 32-bit fragment loads
   bank-conflict-free.
 - Edges: rows >= M / cols >= N and a trailing half K-step (K % 32 == 16) are
   zero-filled; out-of-range outputs are never stored.
At the benchmark (M=2048, N=18432, K=3072): 16 x 144 = 2304 blocks, 232 GFLOP.
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BM 128
#define BN 128
#define BK 32
#define PITCH (BK + 8)       // halves per smem row
#define THREADS 256

// fp16 high bytes of E2M1 magnitudes 0..7: 0, 0.5, 1, 1.5, 2, 3, 4, 6
#define LUT_LO 0x3E3C3800u
#define LUT_HI 0x46444240u

__device__ __forceinline__ float e4m3_to_float(uint32_t b)
{
    const uint32_t e = (b >> 3) & 0xF, m = b & 7;
    float v = (e == 0) ? (float)m * 0.001953125f                       // m * 2^-9
                       : __uint_as_float(((e + 120u) << 23) | (m << 20)); // (1 + m/8) * 2^(e-7)
    return (b & 0x80) ? -v : v;
}

// Pack 4 byte-lane values (each 0..7) into 4 selector nibbles for __byte_perm.
__device__ __forceinline__ uint32_t lanes_to_nibbles(uint32_t x)
{
    x = (x | (x >> 4)) & 0x00FF00FFu;
    x = (x | (x >> 8)) & 0x0000FFFFu;
    return x;
}

// Decode 8 FP4 values (one 32-bit word = 4 packed bytes, value order
// hi(b0), lo(b0), hi(b1), lo(b1), ...) into 4 half2 words scaled by s2.
__device__ __forceinline__ void decode8(uint32_t w, __half2 s2, uint32_t (&out)[4])
{
    const uint32_t selE = lanes_to_nibbles((w >> 4) & 0x07070707u);   // even-index magnitudes
    const uint32_t selO = lanes_to_nibbles(w & 0x07070707u);          // odd-index magnitudes
    const uint32_t hbE = __byte_perm(LUT_LO, LUT_HI, selE);
    const uint32_t hbO = __byte_perm(LUT_LO, LUT_HI, selO);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        // byte1 = even value's high byte, byte3 = odd value's high byte
        uint32_t h = __byte_perm(hbE, hbO, (uint32_t)(((4 + k) << 12) | (k << 4))) & 0xFF00FF00u;
        h |= ((w >> (8 * k + 7)) & 1u) << 15;                          // even sign
        h |= ((w >> (8 * k + 3)) & 1u) << 31;                          // odd sign
        __half2 v = *reinterpret_cast<__half2*>(&h);
        v = __hmul2(v, s2);                                            // exact (see header)
        out[k] = *reinterpret_cast<uint32_t*>(&v);
    }
}

__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct Raw { uint2 q; uint32_t s; };

// Thread t loads row (t >> 1), 16-value block (t & 1) of the K-step.
__device__ __forceinline__ Raw load_raw(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sc,
                                        int rows, int row, int kb, int KB, int rowBytes)
{
    Raw r; r.q = make_uint2(0u, 0u); r.s = 0u;
    if (row < rows && kb < KB) {
        r.q = __ldg(reinterpret_cast<const uint2*>(q + (size_t)row * rowBytes + (size_t)kb * 8));
        r.s = __ldg(sc + (size_t)row * KB + kb);
    }
    return r;
}

__device__ __forceinline__ void store_decoded(const Raw& r, __half (*dst)[PITCH], int lr, int half)
{
    const __half2 s2 = __float2half2_rn(e4m3_to_float(r.s));   // E4M3 is exact in fp16
    uint32_t o0[4], o1[4];
    decode8(r.q.x, s2, o0);
    decode8(r.q.y, s2, o1);
    uint4* p = reinterpret_cast<uint4*>(&dst[lr][half * 16]);
    p[0] = make_uint4(o0[0], o0[1], o0[2], o0[3]);
    p[1] = make_uint4(o1[0], o1[1], o1[2], o1[3]);
}

__global__ void __launch_bounds__(THREADS)
nvfp4_gemm_kernel(const uint8_t* __restrict__ x_q, const uint8_t* __restrict__ x_scales,
                  const uint8_t* __restrict__ w_q, const uint8_t* __restrict__ w_scales,
                  __half* __restrict__ y, int M, int N, int K, float alpha)
{
    __shared__ __align__(16) __half As[2][BM][PITCH];
    __shared__ __align__(16) __half Bs[2][BN][PITCH];

    const int tid  = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;          // 2 x 4 warps
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM;                    // consecutive blocks share a W tile
    const int n0 = blockIdx.y * BN;

    const int KB = K >> 4;                             // 16-value blocks per row
    const int rowBytes = K >> 1;
    const int nSteps = (K + BK - 1) / BK;
    const int lr = tid >> 1, hb = tid & 1;

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

    Raw rx = load_raw(x_q, x_scales, M, m0 + lr, hb, KB, rowBytes);
    Raw rw = load_raw(w_q, w_scales, N, n0 + lr, hb, KB, rowBytes);
    store_decoded(rx, As[0], lr, hb);
    store_decoded(rw, Bs[0], lr, hb);
    __syncthreads();

    for (int s = 0; s < nSteps; ++s) {
        const int cur = s & 1;
        if (s + 1 < nSteps) {
            const int kb = (s + 1) * 2 + hb;
            rx = load_raw(x_q, x_scales, M, m0 + lr, kb, KB, rowBytes);
            rw = load_raw(w_q, w_scales, N, n0 + lr, kb, KB, rowBytes);
        }

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[4][4], bf[4][2];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int r = wm * 64 + mi * 16 + g;
                af[mi][0] = *reinterpret_cast<const uint32_t*>(&As[cur][r    ][kk + 2 * t4]);
                af[mi][1] = *reinterpret_cast<const uint32_t*>(&As[cur][r + 8][kk + 2 * t4]);
                af[mi][2] = *reinterpret_cast<const uint32_t*>(&As[cur][r    ][kk + 2 * t4 + 8]);
                af[mi][3] = *reinterpret_cast<const uint32_t*>(&As[cur][r + 8][kk + 2 * t4 + 8]);
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int c = wn * 32 + ni * 8 + g;
                bf[ni][0] = *reinterpret_cast<const uint32_t*>(&Bs[cur][c][kk + 2 * t4]);
                bf[ni][1] = *reinterpret_cast<const uint32_t*>(&Bs[cur][c][kk + 2 * t4 + 8]);
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma_16816(acc[mi][ni], af[mi], bf[ni]);
        }

        if (s + 1 < nSteps) {
            store_decoded(rx, As[cur ^ 1], lr, hb);
            store_decoded(rw, Bs[cur ^ 1], lr, hb);
        }
        __syncthreads();
    }

    // ---- epilogue: alpha * acc -> fp16 ----
    const bool pairOK = (N % 2) == 0;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int col = n0 + wn * 32 + ni * 8 + 2 * t4;
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int row = m0 + wm * 64 + mi * 16 + g + hr * 8;
                if (row >= M) continue;
                const float v0 = alpha * acc[mi][ni][2 * hr];
                const float v1 = alpha * acc[mi][ni][2 * hr + 1];
                __half* dst = y + (size_t)row * N + col;
                if (pairOK && col + 1 < N) {
                    *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(v0, v1);
                } else {
                    if (col     < N) dst[0] = __float2half_rn(v0);
                    if (col + 1 < N) dst[1] = __float2half_rn(v1);
                }
            }
        }
    }
}

// x_q, x_scales, w_q, w_scales, y are device pointers
extern "C" void solve(const uint8_t* x_q, const uint8_t* x_scales, const uint8_t* w_q,
                      const uint8_t* w_scales, __half* y, int M, int N, int K, float alpha)
{
    if (M <= 0 || N <= 0 || K <= 0) return;
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    nvfp4_gemm_kernel<<<grid, THREADS>>>(x_q, x_scales, w_q, w_scales, y, M, N, K, alpha);
    cudaDeviceSynchronize();
}
