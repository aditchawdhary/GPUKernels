/*
Multi-Head Cross-Attention
Q, output: (M, H, D); K, V: (N, H, D); float32, row-major.
  out[m, h] = softmax_n( Q[m, h] . K[n, h] / sqrt(D) ) @ V[:, h]      (no mask)
(Spec: leetgpu-challenges #26, tolerance 1e-4. M, N ≤ 4096, H ≤ 64, D ≤ 256.
Performance at M = 1024, N = 2048, H = 16, D = 128.)
*/

/*
Approach: an FP32 flash-attention kernel (online softmax; the M x N score matrix is
never materialized). One block per (32-query tile, head); at the benchmark that's
32 x 16 = 512 blocks.

 - Layout: token t, head h starts at (t*H + h)*D, so each tile is a strided gather
   of D-contiguous rows. Loads are coalesced along d.
 - Tiles: 32 queries x 32 keys, 128 threads arranged 16 (tx) x 8 (ty). A thread owns
   4 query rows: S on key columns tx and tx + 16, and O on columns 4tx..4tx+3 of
   each 64-wide d-chunk. NCH = ceil(D / 64) is a template parameter (1/2/4), so
   D ≤ 256 works.
 - Shared memory (20 KB): Q chunk [d][q] (loaded once when NCH = 1, otherwise per
   chunk), K chunk transposed with an XOR swizzle (conflict-free), or V chunk
   row-major [k][d] (float4 reads), and the probabilities [k][q] swizzled in
   float4 units.
 - exp2 with scale*log2(e) folded into Q. Keys past N are masked, so any N works,
   including N = 1 (out = V) and all-zero inputs (uniform 1/N).
FLOPs at the benchmark: 4*M*N*D*H ≈ 17 GFLOP of FP32.
*/

#include <cuda_runtime.h>
#include <math.h>

#define BQ 32
#define BKT 32
#define DC 64
#define THREADS 128

__device__ __forceinline__ int kc_idx(int d, int k) { return d * BKT + (k ^ (d & 31)); }
__device__ __forceinline__ int t32_idx(int a, int q) { return a * 32 + ((((q >> 2) ^ (a & 7))) << 2) + (q & 3); }

template <int NCH>
__global__ void __launch_bounds__(THREADS)
mhca(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
     float* __restrict__ out, int M, int N, int H, int D, float qscale)
{
    __shared__ __align__(16) float Qc[DC * BQ];
    __shared__ __align__(16) float KV[DC * BKT];
    __shared__ __align__(16) float Pt[BKT * BQ];

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int q0 = blockIdx.x * BQ, h = blockIdx.y;
    const size_t rs = (size_t)H * D;                 // token stride
    const size_t hc = (size_t)h * D;                 // head offset

    float o[NCH][4][4], m[4], l[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY; l[i] = 0.f;
        #pragma unroll
        for (int c = 0; c < NCH; ++c)
            #pragma unroll
            for (int j = 0; j < 4; ++j) o[c][i][j] = 0.f;
    }

    auto load_q = [&](int c) {
        const int db = c * DC, dw = min(DC, D - db);
        for (int e = tid; e < BQ * DC; e += THREADS) {
            const int r = e / DC, dd = e % DC, q = q0 + r;
            Qc[t32_idx(dd, r)] = (q < M && dd < dw) ? __ldg(Q + q * rs + hc + db + dd) * qscale : 0.f;
        }
    };
    if (NCH == 1) load_q(0);                         // visible after the first barrier

    const int nKT = (N + BKT - 1) / BKT;
    for (int kt = 0; kt < nKT; ++kt) {
        const int k0 = kt * BKT;

        float s[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();
            if (NCH > 1) load_q(c);
            for (int e = tid; e < BKT * DC; e += THREADS) {
                const int r = e / DC, dd = e % DC, k = k0 + r;
                KV[kc_idx(dd, r)] = (k < N && dd < dw) ? __ldg(K + k * rs + hc + db + dd) : 0.f;
            }
            __syncthreads();
            #pragma unroll 4
            for (int dd = 0; dd < dw; ++dd) {
                const float4 qv = *reinterpret_cast<const float4*>(&Qc[t32_idx(dd, ty * 4)]);
                const float* kr = &KV[dd * BKT];
                const int sw = dd & 31;
                const float k0v = kr[tx ^ sw], k1v = kr[(tx + 16) ^ sw];
                s[0][0] = fmaf(qv.x, k0v, s[0][0]); s[0][1] = fmaf(qv.x, k1v, s[0][1]);
                s[1][0] = fmaf(qv.y, k0v, s[1][0]); s[1][1] = fmaf(qv.y, k1v, s[1][1]);
                s[2][0] = fmaf(qv.z, k0v, s[2][0]); s[2][1] = fmaf(qv.z, k1v, s[2][1]);
                s[3][0] = fmaf(qv.w, k0v, s[3][0]); s[3][1] = fmaf(qv.w, k1v, s[3][1]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float mx = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (k0 + tx + 16 * j >= N) s[i][j] = -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            const float mNew = fmaxf(m[i], mx);      // finite: every tile has a valid key
            const float alpha = exp2f(m[i] - mNew);
            float sum = 0.f;
            #pragma unroll
            for (int j = 0; j < 2; ++j) { s[i][j] = exp2f(s[i][j] - mNew); sum += s[i][j]; }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            m[i] = mNew;
            l[i] = l[i] * alpha + sum;
            #pragma unroll
            for (int c = 0; c < NCH; ++c)
                #pragma unroll
                for (int j = 0; j < 4; ++j) o[c][i][j] *= alpha;
            #pragma unroll
            for (int j = 0; j < 2; ++j) Pt[t32_idx(tx + 16 * j, ty * 4 + i)] = s[i][j];
        }

        #pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();                         // K reads done, Pt complete
            for (int e = tid; e < BKT * DC; e += THREADS) {
                const int r = e / DC, dd = e % DC, k = k0 + r;
                KV[r * DC + dd] = (k < N && dd < dw) ? __ldg(V + k * rs + hc + db + dd) : 0.f;
            }
            __syncthreads();
            #pragma unroll 8
            for (int k = 0; k < BKT; ++k) {
                const float4 pv = *reinterpret_cast<const float4*>(&Pt[t32_idx(k, ty * 4)]);
                const float4 vv = *reinterpret_cast<const float4*>(&KV[k * DC + tx * 4]);
                const float pa[4] = {pv.x, pv.y, pv.z, pv.w};
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o[c][i][0] = fmaf(pa[i], vv.x, o[c][i][0]);
                    o[c][i][1] = fmaf(pa[i], vv.y, o[c][i][1]);
                    o[c][i][2] = fmaf(pa[i], vv.z, o[c][i][2]);
                    o[c][i][3] = fmaf(pa[i], vv.w, o[c][i][3]);
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = q0 + ty * 4 + i;
        if (q >= M) continue;
        const float inv = 1.f / l[i];
        float* orow = out + q * rs + hc;
        #pragma unroll
        for (int c = 0; c < NCH; ++c)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = c * DC + tx * 4 + j;
                if (col < D) orow[col] = o[c][i][j] * inv;
            }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int H, int D)
{
    if (M <= 0 || N <= 0 || H <= 0 || D <= 0) return;
    const float qscale = (1.0f / sqrtf((float)D)) * 1.4426950408889634f;   // * log2(e)
    dim3 grid((M + BQ - 1) / BQ, H);
    if      (D <= 64)  mhca<1><<<grid, THREADS>>>(Q, K, V, output, M, N, H, D, qscale);
    else if (D <= 128) mhca<2><<<grid, THREADS>>>(Q, K, V, output, M, N, H, D, qscale);
    else               mhca<4><<<grid, THREADS>>>(Q, K, V, output, M, N, H, D, qscale);
    cudaDeviceSynchronize();
}
