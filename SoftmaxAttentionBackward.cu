/*
Softmax Attention Backward
Q, dO: (M, d); K, V: (N, d); float32. Forward (no mask): S = Q K^T / sqrt(d),
P = softmax_row(S), O = P V. Compute
  dV = P^T dO
  dP = dO V^T,  dS = P * (dP - rowsum(dP * P)),  dS /= sqrt(d)
  dQ = dS K,    dK = dS^T Q
(Spec: leetgpu-challenges #111, tolerance 1e-4.)
Constraints: M, N ≤ 100,000, d ≤ 128. Performance is measured with M = 8192,
N = 4096, d = 128.
*/

/*
Approach: FlashAttention-2 style backward in FP32. P is never materialized (it
can be 100k x 100k); it's recomputed tile by tile from per-row log-sum-exp.

  0. Identity: rowsum(dP * P)_i = sum_j P_ij (dO_i . V_j) = dO_i . O_i =: D_i,
     so dS = P * (dP - D_i) needs only one number per row.
  1. fwd_stats   per 32-query tile: flash forward over all keys (online softmax,
                 O accumulated in registers), then writes
                 LSE2_i = m_i + log2(l_i) (log2 domain) and D_i = dO_i . O_i / l_i.
  2. bwd_kv      per (32-key tile, query split): for each 32-query tile,
                   S^T = K Q^T, dP^T = V dO^T  (two 32x32 tiles over d-chunks),
                   P^T = exp2(S^T*c - LSE2_q), dS^T = P^T (dP^T - D_q),
                   dV += P^T dO,  dK += dS^T Q.
  3. bwd_q       per (32-query tile, key split): S and dP for each key tile, then
                 dS, dQ += dS K.
  Splits keep about 512 blocks or more in flight (at the benchmark: 4 query
  splits for the 128 key tiles, 2 key splits for the 256 query tiles). Each split
  writes a partial result to scratch, and a sum kernel adds them in a fixed order,
  so the result is deterministic. With one split, kernels write the outputs
  directly. 1/sqrt(d) is applied as partials are written.

Tile machinery (shared by all three kernels): 32 x 32 tiles, 128 threads =
16 (tx) x 8 (ty). A thread owns 4 tile rows (4ty..4ty+3) and tile columns
tx, tx + 16, plus 4 output columns per 64-wide d-chunk (4tx..4tx+3). d ≤ 128
means NCH ∈ {1, 2} chunks, a template parameter.
  row operand  [d][r]  swizzled in float4 units  -> one broadcast float4 = 4 rows
  col operand  [d][c]  XOR-swizzled              -> conflict-free scalars
  prob matrix  [c][r]  swizzled in float4 units  (reduction index first)
  row-major    [r][64] float4 reads
Shared memory is 24 KB per block. All accumulation is FP32; exp2 has
scale*log2(e) folded into the scores.
FLOPs at the benchmark: 9 GEMM-sized passes of 8.6 GFLOP each ≈ 77 GFLOP.
*/

#include <cuda_runtime.h>
#include <math.h>

#define TS 32
#define DC 64
#define THREADS 128

__device__ __forceinline__ int col_idx(int d, int c) { return d * TS + (c ^ (d & 31)); }
__device__ __forceinline__ int t32_idx(int a, int r) { return a * 32 + ((((r >> 2) ^ (a & 7))) << 2) + (r & 3); }

// ---- shared-memory staging helpers (32 rows/cols x 64-wide d-chunk)
__device__ __forceinline__ void load_rowop(float* dst, const float* __restrict__ src, int row0, int nrows,
                                           int D, int db, int dw, float scale)
{
    for (int e = threadIdx.x; e < TS * DC; e += THREADS) {
        const int r = e >> 6, dd = e & 63, row = row0 + r;
        dst[t32_idx(dd, r)] = (row < nrows && dd < dw) ? __ldg(src + (size_t)row * D + db + dd) * scale : 0.f;
    }
}
__device__ __forceinline__ void load_colop(float* dst, const float* __restrict__ src, int row0, int nrows,
                                           int D, int db, int dw)
{
    for (int e = threadIdx.x; e < TS * DC; e += THREADS) {
        const int r = e >> 6, dd = e & 63, row = row0 + r;
        dst[col_idx(dd, r)] = (row < nrows && dd < dw) ? __ldg(src + (size_t)row * D + db + dd) : 0.f;
    }
}
__device__ __forceinline__ void load_rowmajor(float* dst, const float* __restrict__ src, int row0, int nrows,
                                              int D, int db, int dw)
{
    for (int e = threadIdx.x; e < TS * DC; e += THREADS) {
        const int r = e >> 6, dd = e & 63, row = row0 + r;
        dst[r * DC + dd] = (row < nrows && dd < dw) ? __ldg(src + (size_t)row * D + db + dd) : 0.f;
    }
}
// s[a][b] += sum_dd rowop[dd][4ty+a] * colop[dd][tx+16b]
__device__ __forceinline__ void tile_mma(float (&s)[4][2], const float* A, const float* B, int dw)
{
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    #pragma unroll 4
    for (int dd = 0; dd < dw; ++dd) {
        const float4 a = *reinterpret_cast<const float4*>(&A[t32_idx(dd, ty * 4)]);
        const float* br = &B[dd * TS];
        const int sw = dd & 31;
        const float b0 = br[tx ^ sw], b1 = br[(tx + 16) ^ sw];
        s[0][0] = fmaf(a.x, b0, s[0][0]); s[0][1] = fmaf(a.x, b1, s[0][1]);
        s[1][0] = fmaf(a.y, b0, s[1][0]); s[1][1] = fmaf(a.y, b1, s[1][1]);
        s[2][0] = fmaf(a.z, b0, s[2][0]); s[2][1] = fmaf(a.z, b1, s[2][1]);
        s[3][0] = fmaf(a.w, b0, s[3][0]); s[3][1] = fmaf(a.w, b1, s[3][1]);
    }
}
// acc[a][0..3] += sum_c Pm[c][4ty+a] * R[c][4tx .. 4tx+3]
__device__ __forceinline__ void accum(float (&acc)[4][4], const float* Pm, const float* R)
{
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    #pragma unroll 8
    for (int c = 0; c < TS; ++c) {
        const float4 p = *reinterpret_cast<const float4*>(&Pm[t32_idx(c, ty * 4)]);
        const float4 v = *reinterpret_cast<const float4*>(&R[c * DC + tx * 4]);
        const float pa[4] = {p.x, p.y, p.z, p.w};
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            acc[a][0] = fmaf(pa[a], v.x, acc[a][0]); acc[a][1] = fmaf(pa[a], v.y, acc[a][1]);
            acc[a][2] = fmaf(pa[a], v.z, acc[a][2]); acc[a][3] = fmaf(pa[a], v.w, acc[a][3]);
        }
    }
}
template <int NCH>
__device__ __forceinline__ void store_rows(float* __restrict__ dst, const float (&acc)[NCH][4][4],
                                           int row0, int nrows, int D, float scale)
{
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int row = row0 + ty * 4 + a;
        if (row >= nrows) continue;
        #pragma unroll
        for (int c = 0; c < NCH; ++c)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = c * DC + tx * 4 + j;
                if (col < D) dst[(size_t)row * D + col] = acc[c][a][j] * scale;
            }
    }
}

// ======================================================================== 1. forward stats
template <int NCH>
__global__ void __launch_bounds__(THREADS)
fwd_stats(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
          const float* __restrict__ dO, float* __restrict__ LSE, float* __restrict__ Dv,
          int M, int N, int D, float c2)
{
    __shared__ __align__(16) float A[DC * TS], B[DC * TS], Pm[TS * TS];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int q0 = blockIdx.x * TS;

    float o[NCH][4][4], m[4], l[4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        m[a] = -INFINITY; l[a] = 0.f;
        #pragma unroll
        for (int c = 0; c < NCH; ++c)
            #pragma unroll
            for (int j = 0; j < 4; ++j) o[c][a][j] = 0.f;
    }
    if (NCH == 1) load_rowop(A, Q, q0, M, D, 0, min(DC, D), c2);

    const int nKT = (N + TS - 1) / TS;
    for (int kt = 0; kt < nKT; ++kt) {
        const int k0 = kt * TS;
        float s[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();
            if (NCH > 1) load_rowop(A, Q, q0, M, D, db, dw, c2);
            load_colop(B, K, k0, N, D, db, dw);
            __syncthreads();
            tile_mma(s, A, B, dw);
        }
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            float mx = -INFINITY;
            #pragma unroll
            for (int b = 0; b < 2; ++b) {
                if (k0 + tx + 16 * b >= N) s[a][b] = -INFINITY;
                mx = fmaxf(mx, s[a][b]);
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            const float mNew = fmaxf(m[a], mx);          // finite: tile has ≥ 1 valid key
            const float alpha = exp2f(m[a] - mNew);
            float sum = 0.f;
            #pragma unroll
            for (int b = 0; b < 2; ++b) { s[a][b] = exp2f(s[a][b] - mNew); sum += s[a][b]; }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            m[a] = mNew;
            l[a] = l[a] * alpha + sum;
            #pragma unroll
            for (int c = 0; c < NCH; ++c)
                #pragma unroll
                for (int j = 0; j < 4; ++j) o[c][a][j] *= alpha;
            #pragma unroll
            for (int b = 0; b < 2; ++b) Pm[t32_idx(tx + 16 * b, ty * 4 + a)] = s[a][b];
        }
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();
            load_rowmajor(B, V, k0, N, D, db, dw);
            __syncthreads();
            accum(o[c], Pm, B);
        }
    }

    // LSE (log2 domain) and D_i = dO_i . O_i
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int q = q0 + ty * 4 + a;
        const float inv = 1.f / l[a];
        float dsum = 0.f;
        if (q < M) {
            #pragma unroll
            for (int c = 0; c < NCH; ++c)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = c * DC + tx * 4 + j;
                    if (col < D) dsum = fmaf(o[c][a][j] * inv, __ldg(dO + (size_t)q * D + col), dsum);
                }
        }
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) dsum += __shfl_xor_sync(0xffffffff, dsum, off);
        if (tx == 0 && q < M) { LSE[q] = m[a] + log2f(l[a]); Dv[q] = dsum; }
    }
}

// ======================================================================== 2. dK, dV
template <int NCH>
__global__ void __launch_bounds__(THREADS)
bwd_kv(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
       const float* __restrict__ dO, const float* __restrict__ LSE, const float* __restrict__ Dv,
       float* __restrict__ dK, float* __restrict__ dV, size_t plane,
       int M, int N, int D, float c2, float invs, int qtPerSplit)
{
    __shared__ __align__(16) float A[DC * TS], B[DC * TS], Pm[TS * TS], dSm[TS * TS];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int k0 = blockIdx.x * TS;
    const int nQT = (M + TS - 1) / TS;
    const int qtA = blockIdx.y * qtPerSplit, qtB = min(qtA + qtPerSplit, nQT);

    float gv[NCH][4][4], gk[NCH][4][4];
    #pragma unroll
    for (int c = 0; c < NCH; ++c)
        #pragma unroll
        for (int a = 0; a < 4; ++a)
            #pragma unroll
            for (int j = 0; j < 4; ++j) { gv[c][a][j] = 0.f; gk[c][a][j] = 0.f; }

    for (int qt = qtA; qt < qtB; ++qt) {
        const int q0 = qt * TS;
        float s[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        float p[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {                   // S^T = K Q^T (scaled)
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();
            load_rowop(A, K, k0, N, D, db, dw, c2);
            load_colop(B, Q, q0, M, D, db, dw);
            __syncthreads();
            tile_mma(s, A, B, dw);
        }
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {                   // dP^T = V dO^T
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();
            load_rowop(A, V, k0, N, D, db, dw, 1.f);
            load_colop(B, dO, q0, M, D, db, dw);
            __syncthreads();
            tile_mma(p, A, B, dw);
        }
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int q = q0 + tx + 16 * b;
            const bool ok = q < M;
            const float lse = ok ? __ldg(LSE + q) : 0.f, dq = ok ? __ldg(Dv + q) : 0.f;
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float P = ok ? exp2f(s[a][b] - lse) : 0.f;
                Pm[t32_idx(tx + 16 * b, ty * 4 + a)]  = P;
                dSm[t32_idx(tx + 16 * b, ty * 4 + a)] = P * (p[a][b] - dq);
            }
        }
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {                   // dV += P^T dO, dK += dS^T Q
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();
            load_rowmajor(A, dO, q0, M, D, db, dw);
            load_rowmajor(B, Q, q0, M, D, db, dw);
            __syncthreads();
            accum(gv[c], Pm, A);
            accum(gk[c], dSm, B);
        }
    }
    store_rows<NCH>(dV + blockIdx.y * plane, gv, k0, N, D, 1.f);
    store_rows<NCH>(dK + blockIdx.y * plane, gk, k0, N, D, invs);
}

// ======================================================================== 3. dQ
template <int NCH>
__global__ void __launch_bounds__(THREADS)
bwd_q(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
      const float* __restrict__ dO, const float* __restrict__ LSE, const float* __restrict__ Dv,
      float* __restrict__ dQ, size_t plane, int M, int N, int D, float c2, float invs, int ktPerSplit)
{
    __shared__ __align__(16) float A[DC * TS], B[DC * TS], dSm[TS * TS];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int q0 = blockIdx.x * TS;
    const int nKT = (N + TS - 1) / TS;
    const int ktA = blockIdx.y * ktPerSplit, ktB = min(ktA + ktPerSplit, nKT);

    float lse[4], dd[4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int q = q0 + ty * 4 + a;
        lse[a] = (q < M) ? __ldg(LSE + q) : 0.f;
        dd[a]  = (q < M) ? __ldg(Dv + q) : 0.f;
    }
    float g[NCH][4][4];
    #pragma unroll
    for (int c = 0; c < NCH; ++c)
        #pragma unroll
        for (int a = 0; a < 4; ++a)
            #pragma unroll
            for (int j = 0; j < 4; ++j) g[c][a][j] = 0.f;

    for (int kt = ktA; kt < ktB; ++kt) {
        const int k0 = kt * TS;
        float s[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        float p[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {                   // S = Q K^T (scaled)
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();
            load_rowop(A, Q, q0, M, D, db, dw, c2);
            load_colop(B, K, k0, N, D, db, dw);
            __syncthreads();
            tile_mma(s, A, B, dw);
        }
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {                   // dP = dO V^T
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();
            load_rowop(A, dO, q0, M, D, db, dw, 1.f);
            load_colop(B, V, k0, N, D, db, dw);
            __syncthreads();
            tile_mma(p, A, B, dw);
        }
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
            const bool ok = (k0 + tx + 16 * b) < N;
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float P = ok ? exp2f(s[a][b] - lse[a]) : 0.f;
                dSm[t32_idx(tx + 16 * b, ty * 4 + a)] = P * (p[a][b] - dd[a]);
            }
        }
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {                   // dQ += dS K
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();
            load_rowmajor(B, K, k0, N, D, db, dw);
            __syncthreads();
            accum(g[c], dSm, B);
        }
    }
    store_rows<NCH>(dQ + blockIdx.y * plane, g, q0, M, D, invs);
}

// ======================================================================== split reduction
__global__ void sum_planes(const float* __restrict__ part, float* __restrict__ out, size_t n, int S)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int s = 0; s < S; ++s) acc += part[(size_t)s * n + i];
    out[i] = acc;
}

// ======================================================================== host
static float* g_ws = nullptr;
static size_t g_cap = 0;

template <int NCH>
static void run(const float* Q, const float* K, const float* V, const float* dO,
                float* dQ, float* dK, float* dV, int M, int N, int D)
{
    const float c2 = (1.0f / sqrtf((float)D)) * 1.4426950408889634f;   // scale * log2(e)
    const float invs = 1.0f / sqrtf((float)D);
    const int nQT = (M + TS - 1) / TS, nKT = (N + TS - 1) / TS;

    int SA = (512 + nKT - 1) / nKT; SA = max(1, min(SA, nQT));
    const int qtPer = (nQT + SA - 1) / SA; SA = (nQT + qtPer - 1) / qtPer;
    int SB = (512 + nQT - 1) / nQT; SB = max(1, min(SB, nKT));
    const int ktPer = (nKT + SB - 1) / SB; SB = (nKT + ktPer - 1) / ktPer;

    const size_t nKV = (size_t)N * D, nQ = (size_t)M * D;
    const size_t need = 2 * (size_t)M + (SA > 1 ? 2 * SA * nKV : 0) + (SB > 1 ? SB * nQ : 0);
    if (need > g_cap) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, need * sizeof(float));
        g_cap = need;
    }
    float* LSE = g_ws;
    float* Dv  = LSE + M;
    float* pK  = Dv + M;
    float* pV  = pK + (SA > 1 ? SA * nKV : 0);
    float* pQ  = pV + (SA > 1 ? SA * nKV : 0);

    fwd_stats<NCH><<<nQT, THREADS>>>(Q, K, V, dO, LSE, Dv, M, N, D, c2);

    if (SA > 1) {
        bwd_kv<NCH><<<dim3(nKT, SA), THREADS>>>(Q, K, V, dO, LSE, Dv, pK, pV, nKV, M, N, D, c2, invs, qtPer);
        sum_planes<<<(unsigned)((nKV + 255) / 256), 256>>>(pK, dK, nKV, SA);
        sum_planes<<<(unsigned)((nKV + 255) / 256), 256>>>(pV, dV, nKV, SA);
    } else {
        bwd_kv<NCH><<<dim3(nKT, 1), THREADS>>>(Q, K, V, dO, LSE, Dv, dK, dV, 0, M, N, D, c2, invs, qtPer);
    }

    if (SB > 1) {
        bwd_q<NCH><<<dim3(nQT, SB), THREADS>>>(Q, K, V, dO, LSE, Dv, pQ, nQ, M, N, D, c2, invs, ktPer);
        sum_planes<<<(unsigned)((nQ + 255) / 256), 256>>>(pQ, dQ, nQ, SB);
    } else {
        bwd_q<NCH><<<dim3(nQT, 1), THREADS>>>(Q, K, V, dO, LSE, Dv, dQ, 0, M, N, D, c2, invs, ktPer);
    }
}

// Q, K, V, dO, dQ, dK, dV are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* dO, float* dQ,
                      float* dK, float* dV, int M, int N, int d)
{
    if (M <= 0 || N <= 0 || d <= 0) return;
    if (d <= 64) run<1>(Q, K, V, dO, dQ, dK, dV, M, N, d);
    else         run<2>(Q, K, V, dO, dQ, dK, dV, M, N, d);
    cudaDeviceSynchronize();
}
