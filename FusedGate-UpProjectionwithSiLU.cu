#include <cuda_runtime.h>
#include <math.h>

// Fused gate-up projection + SiLU (SwiGLU MLP front half), full FP32:
//   g = x @ W[:, i],  u = x @ W[:, I + i],  out[m, i] = silu(g) * u = g / (1 + e^-g) * u
//
// One FP32 SIMT GEMM whose epilogue does the SiLU-and-mul in registers, so the
// (M x 2I) intermediate never reaches global memory.
//  - Block tile: 128 rows x 64 output channels. Its B tile is 128 columns wide:
//    smem columns 0..63 = gate columns i0..i0+63, columns 64..127 = the matching
//    up columns I+i0..I+i0+63.
//  - 256 threads in a 16 x 16 grid, each owning an 8 x 8 accumulator: rows
//    {ty*4+r, 64+ty*4+r} and smem columns {tx*4+j, 64+tx*4+j}. The second
//    column group is the up partner of the first, so each thread holds g and u
//    for the same 8 x 4 (row, channel) outputs.
//  - BK = 8, double-buffered smem with register prefetch; A is stored transposed
//    (As[k][m]) so both operands are read as float4 broadcasts / conflict-free rows.
//  - Vectorized float4 global loads when the layout allows (H % 4 == 0 for x;
//    I % 4 == 0 for W), scalar guarded loads otherwise. All edges are guarded.
// Consecutive blocks share a W column tile (blockIdx.x walks M), so each 2 MB W
// tile is read from DRAM once and reused from L2 by all 32 row-tiles.

#define BM 128
#define BN 64                 // output channels per block (B tile is 2*BN wide)
#define BK 8
#define THREADS 256

template <bool VEC>
__device__ __forceinline__ void load_A(const float* __restrict__ x, int M, int H, int m0, int k0,
                                       float (&ra)[4], int tid)
{
    // 128 rows x 8 k: thread -> row tid/2, k (tid&1)*4 .. +3
    const int r = tid >> 1, kk = (tid & 1) * 4;
    const int gr = m0 + r, gk = k0 + kk;
    ra[0] = ra[1] = ra[2] = ra[3] = 0.f;
    if (gr < M) {
        const float* p = x + (size_t)gr * H + gk;
        if (VEC && gk + 3 < H) {
            const float4 t = __ldg(reinterpret_cast<const float4*>(p));
            ra[0] = t.x; ra[1] = t.y; ra[2] = t.z; ra[3] = t.w;
        } else {
#pragma unroll
            for (int u = 0; u < 4; ++u) if (gk + u < H) ra[u] = __ldg(p + u);
        }
    }
}

template <bool VEC>
__device__ __forceinline__ void load_B(const float* __restrict__ W, int H, int I, int i0, int k0,
                                       float (&rb)[4], int tid)
{
    // 8 k x 128 smem columns: thread -> k tid/32, smem cols (tid&31)*4 .. +3
    const int kk = tid >> 5, c = (tid & 31) * 4;
    const int gk = k0 + kk;
    const int ch = i0 + (c & (BN - 1));                 // channel of first element
    const int gc = (c < BN) ? ch : I + ch;              // global column in W
    const int chEnd = I;                                // channels must be < I
    rb[0] = rb[1] = rb[2] = rb[3] = 0.f;
    if (gk < H) {
        const float* p = W + (size_t)gk * (2 * (size_t)I) + gc;
        if (VEC && ch + 3 < chEnd) {
            const float4 t = __ldg(reinterpret_cast<const float4*>(p));
            rb[0] = t.x; rb[1] = t.y; rb[2] = t.z; rb[3] = t.w;
        } else {
#pragma unroll
            for (int u = 0; u < 4; ++u) if (ch + u < chEnd) rb[u] = __ldg(p + u);
        }
    }
}

__device__ __forceinline__ void store_A(float (*As)[BM + 4], const float (&ra)[4], int tid)
{
    const int r = tid >> 1, kk = (tid & 1) * 4;
#pragma unroll
    for (int u = 0; u < 4; ++u) As[kk + u][r] = ra[u];
}

__device__ __forceinline__ void store_B(float (*Bs)[2 * BN + 4], const float (&rb)[4], int tid)
{
    const int kk = tid >> 5, c = (tid & 31) * 4;
    *reinterpret_cast<float4*>(&Bs[kk][c]) = make_float4(rb[0], rb[1], rb[2], rb[3]);
}

template <bool VA, bool VB>
__global__ void __launch_bounds__(THREADS, 2)
gate_up_silu_kernel(const float* __restrict__ x, const float* __restrict__ W,
                    float* __restrict__ out, int M, int H, int I)
{
    __shared__ __align__(16) float As[2][BK][BM + 4];
    __shared__ __align__(16) float Bs[2][BK][2 * BN + 4];

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.x * BM;
    const int i0 = blockIdx.y * BN;

    float acc[8][8];
#pragma unroll
    for (int a = 0; a < 8; ++a)
#pragma unroll
        for (int b = 0; b < 8; ++b) acc[a][b] = 0.f;

    float ra[4], rb[4];
    load_A<VA>(x, M, H, m0, 0, ra, tid);
    load_B<VB>(W, H, I, i0, 0, rb, tid);
    store_A(As[0], ra, tid);
    store_B(Bs[0], rb, tid);
    __syncthreads();

    const int nk = (H + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        const int cur = t & 1;
        if (t + 1 < nk) {                                // prefetch next K slice into registers
            load_A<VA>(x, M, H, m0, (t + 1) * BK, ra, tid);
            load_B<VB>(W, H, I, i0, (t + 1) * BK, rb, tid);
        }
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[cur][k][64 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4]);        // gate
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[cur][k][BN + tx * 4]);   // up
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int p = 0; p < 8; ++p)
#pragma unroll
                for (int q = 0; q < 8; ++q) acc[p][q] = fmaf(a[p], b[q], acc[p][q]);
        }
        if (t + 1 < nk) {
            store_A(As[cur ^ 1], ra, tid);
            store_B(Bs[cur ^ 1], rb, tid);
        }
        __syncthreads();
    }

    // ---- epilogue: out = silu(gate) * up ----
    const int ch0 = i0 + tx * 4;
    const bool vecOut = ((I & 3) == 0) && (ch0 + 3 < I);
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int row = m0 + (p < 4 ? ty * 4 + p : 64 + ty * 4 + (p - 4));
        if (row >= M) continue;
        float r[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float g = acc[p][j], u = acc[p][4 + j];
            r[j] = g / (1.f + expf(-g)) * u;
        }
        float* dst = out + (size_t)row * I + ch0;
        if (vecOut) {
            *reinterpret_cast<float4*>(dst) = make_float4(r[0], r[1], r[2], r[3]);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) if (ch0 + j < I) dst[j] = r[j];
        }
    }
}

// x, W_gate_up, output are device pointers
extern "C" void solve(const float* x, const float* W_gate_up, float* output, int M, int H, int I) {
    dim3 grid((M + BM - 1) / BM, (I + BN - 1) / BN);
    const bool va = (H % 4 == 0) && ((reinterpret_cast<size_t>(x) & 15) == 0);
    const bool vb = (I % 4 == 0) && ((reinterpret_cast<size_t>(W_gate_up) & 15) == 0);
    if (va && vb)  gate_up_silu_kernel<true,  true ><<<grid, THREADS>>>(x, W_gate_up, output, M, H, I);
    else if (va)   gate_up_silu_kernel<true,  false><<<grid, THREADS>>>(x, W_gate_up, output, M, H, I);
    else if (vb)   gate_up_silu_kernel<false, true ><<<grid, THREADS>>>(x, W_gate_up, output, M, H, I);
    else           gate_up_silu_kernel<false, false><<<grid, THREADS>>>(x, W_gate_up, output, M, H, I);
    cudaDeviceSynchronize();
}
