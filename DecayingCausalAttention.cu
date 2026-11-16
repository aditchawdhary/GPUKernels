/*
Decaying Causal Attention (RetNet parallel form)
Q, K, V, output: (N = seq_len, D = d_model) float32, row-major.
  output[n] = sum_{m <= n} gamma^(n - m) * (Q[n] . K[m]) * V[m]
No softmax and no normalization.

Constraints
 - 1 ≤ N ≤ 8192, 1 ≤ D ≤ 256, 0 < gamma ≤ 1
 - Performance is measured with N = 4096, D = 64
*/

/*
Approach: a tiled "linear attention" kernel. It is flash-attention style, but with
no softmax, so there is no running max or rescaling: O += (S .* decay) V.

 - Tiles: 32 queries x 32 keys, 128 threads arranged 16 (tx) x 8 (ty). Each thread
   owns 4 query rows. It computes S on key columns tx and tx + 16, and accumulates
   O on columns c*64 + 4tx .. +3 for each 64-wide d-chunk c (up to 4 chunks, so
   D ≤ 256 needs 64 accumulators).
 - Shared memory (44 KB static, launches on any GPU with no opt-in):
     Qs[d][q]   the whole Q tile, transposed, loaded once per block: float4 broadcast reads
     KVc        one 64-wide d-chunk at a time, either Kc[d][k] (transposed,
                XOR-swizzled, conflict-free) or Vc[k][d] (row-major, float4 reads)
     Pt[k][q]   decayed scores, transposed, swizzled in float4 units
 - Decay: w = S * exp2((n - m) * log2(gamma)), and w = 0 for m > n. log2(gamma)
   is computed in double on the host.
 - Exact skip for gamma < 1: a key tile is skipped when every weight in it is
   below 2^-160, which is under the smallest float denormal, so those weights
   would round to 0 anyway. Each query tile only visits key tiles from that cutoff
   up to the diagonal. At gamma = 1 nothing is skipped (full causal).
 - Causal: query tile qt visits key tiles up to qt, and the diagonal tile applies
   the m ≤ n mask.
 - Split-K: since there's no softmax, partial outputs over disjoint key ranges just
   add. Each query tile's key range is cut into slices of SPLIT = 8 key tiles
   (256 keys), each slice is its own block (grid = query tiles x slices), and
   blocks atomicAdd into the zeroed output. At N = 4096 that's about 1100 working
   blocks instead of 128, and the causal triangle is load-balanced. Float atomics
   make the summation order non-deterministic, so the last bits can vary run to run.
*/

#include <cuda_runtime.h>
#include <math.h>

#define BQ 32
#define BKT 32
#define DC 64                     // d-chunk width
#define NCH 4                     // max chunks (D ≤ 256)
#define DMAX (DC * NCH)
#define THREADS 128               // 16 x 8
#define SPLIT 8                   // key tiles per block (split-K)

__device__ __forceinline__ int kc_idx(int d, int k) { return d * BKT + (k ^ (d & 31)); }
__device__ __forceinline__ int pt_idx(int k, int q) { return k * BQ + ((((q >> 2) ^ (k & 7))) << 2) + (q & 3); }

__global__ void __launch_bounds__(THREADS)
retention_fwd(const float* __restrict__ Q, const float* __restrict__ K,
              const float* __restrict__ V, float* __restrict__ out,
              int N, int D, float q_scale, float lg2g, int cut)
{
    __shared__ __align__(16) float Qs[DMAX * BQ];      // 32 KB
    __shared__ __align__(16) float KVc[DC * BKT];      //  8 KB (Kc[64][32] or Vc[32][64])
    __shared__ __align__(16) float Pt[BKT * BQ];       //  4 KB

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int qt = blockIdx.x;
    const int q0 = qt * BQ;
    const int nch = (D + DC - 1) / DC;

    // first key tile that can contribute anything non-zero, then this block's slice
    const int num = q0 - cut - (BKT - 1);
    const int kt0 = (num > 0) ? num / BKT : 0;
    const int ktA = kt0 + blockIdx.y * SPLIT;
    if (ktA > qt) return;                                  // whole block exits together
    const int ktB = min(qt, ktA + SPLIT - 1);

    // ---- Q tile -> Qs[d][q]
    for (int e = tid; e < BQ * D; e += THREADS) {
        const int r = e / D, d = e - r * D;
        const int q = q0 + r;
        Qs[d * BQ + r] = (q < N) ? __ldg(Q + (size_t)q * D + d) * q_scale : 0.f;
    }

    float o[NCH][4][4];
    #pragma unroll
    for (int c = 0; c < NCH; ++c)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) o[c][i][j] = 0.f;

    for (int kt = ktA; kt <= ktB; ++kt) {
        const int k0 = kt * BKT;

        // ---- S = Q K^T, accumulated over d-chunks
        float s[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        for (int c = 0; c < nch; ++c) {
            const int dBase = c * DC;
            const int dw = min(DC, D - dBase);
            __syncthreads();                                   // KVc / Pt free
            for (int e = tid; e < BKT * dw; e += THREADS) {
                const int r = e / dw, dd = e - r * dw;
                const int k = k0 + r;
                KVc[kc_idx(dd, r)] = (k < N) ? __ldg(K + (size_t)k * D + dBase + dd) : 0.f;
            }
            __syncthreads();
            #pragma unroll 4
            for (int dd = 0; dd < dw; ++dd) {
                const float4 qv = *reinterpret_cast<const float4*>(&Qs[(dBase + dd) * BQ + ty * 4]);
                const float* kr = &KVc[dd * BKT];
                const int sw = dd & 31;
                const float k0v = kr[tx ^ sw], k1v = kr[(tx + 16) ^ sw];
                s[0][0] = fmaf(qv.x, k0v, s[0][0]); s[0][1] = fmaf(qv.x, k1v, s[0][1]);
                s[1][0] = fmaf(qv.y, k0v, s[1][0]); s[1][1] = fmaf(qv.y, k1v, s[1][1]);
                s[2][0] = fmaf(qv.z, k0v, s[2][0]); s[2][1] = fmaf(qv.z, k1v, s[2][1]);
                s[3][0] = fmaf(qv.w, k0v, s[3][0]); s[3][1] = fmaf(qv.w, k1v, s[3][1]);
            }
        }

        // ---- decay + causal mask -> Pt
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int n = q0 + ty * 4 + i;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int m = k0 + tx + 16 * j;
                float w = 0.f;
                if (m <= n && m < N) w = s[i][j] * exp2f((float)(n - m) * lg2g);
                Pt[pt_idx(tx + 16 * j, ty * 4 + i)] = w;
            }
        }

        // ---- O += P V, per d-chunk
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {
            if (c < nch) {
                const int dBase = c * DC;
                const int dw = min(DC, D - dBase);
                __syncthreads();                               // Kc reads done; Pt ready
                for (int e = tid; e < BKT * DC; e += THREADS) {
                    const int r = e >> 6, dd = e & (DC - 1);
                    const int k = k0 + r;
                    KVc[r * DC + dd] = (k < N && dd < dw) ? __ldg(V + (size_t)k * D + dBase + dd) : 0.f;
                }
                __syncthreads();
                #pragma unroll 8
                for (int k = 0; k < BKT; ++k) {
                    const float4 pv = *reinterpret_cast<const float4*>(&Pt[pt_idx(k, ty * 4)]);
                    const float4 vv = *reinterpret_cast<const float4*>(&KVc[k * DC + tx * 4]);
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
    }

    // ---- accumulate this slice into the (zeroed) output
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int n = q0 + ty * 4 + i;
        if (n >= N) continue;
        float* orow = out + (size_t)n * D;
        #pragma unroll
        for (int c = 0; c < NCH; ++c)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = c * DC + tx * 4 + j;
                if (d < D) atomicAdd(orow + d, o[c][i][j]);
            }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma)
{
    const int N = seq_len, D = d_model;
    if (N <= 0 || D <= 0) return;

    const double lg = log2((double)gamma);           // ≤ 0
    // distance beyond which gamma^dist < 2^-160 (rounds to 0 in float)
    int cut = 1 << 30;
    if (lg < 0.0) {
        const double c = ceil(160.0 / -lg);
        if (c < (double)(1 << 30)) cut = (int)c;
    }

    const int nQT = (N + BQ - 1) / BQ;
    const int nSplit = (nQT + SPLIT - 1) / SPLIT;          // enough for the longest range
    cudaMemsetAsync(output, 0, (size_t)N * D * sizeof(float));
    retention_fwd<<<dim3(nQT, nSplit), THREADS>>>(Q, K, V, output, N, D,  1.0f / sqrtf((float)D), (float)lg, cut);
    cudaDeviceSynchronize();
}
