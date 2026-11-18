/*
Softcapped Sliding Window Attention (Gemma-2 / Mistral style, GQA)
Q, output: (Hq, S, D); K, V: (Hkv, S, D); float32. Query head h uses KV head h / (Hq/Hkv).
Spec from the reference implementation (leetgpu-challenges #152), tolerance 1e-5:
  s_ij = (q_i . k_j) / sqrt(D);   c_ij = softcap * tanh(s_ij / softcap)
  query i sees key j iff j <= i and i - j < window_size
  out_i = softmax_j(c_ij) @ V
Constraints: Hq, Hkv ≤ 32, S ≤ 4096, D ≤ 128, window ≤ 8192, 0 < softcap ≤ 50.
Performance at Hq = 16, Hkv = 4, S = 4096, D = 64, window = 1024, softcap = 10.
*/

/*
Approach: FP32 flash attention that visits only the key tiles inside each query tile's
causal window.

 - Window: query tile [q0, q0+31] needs keys [max(0, q0 - W + 1), min(S-1, q0 + 31)].
   That's about W/32 + 1 key tiles (33 at the benchmark) instead of q0/32. The
   element mask j <= i && i - j < W (strict, so window = 1 means the token sees only
   itself) && j < S is applied on the edge tiles.
 - Precision (the tolerance is 1e-5): scores follow the reference's operation
   order exactly. The raw dot product is formed first, then * (1/sqrt(D)), then
   / softcap, tanhf, then * softcap. The softmax uses natural-base expf with the
   running max subtracted. Everything is accumulated in FP32. Capped scores lie in
   [-softcap, softcap], so the weights are well-conditioned.
 - Tiling (same machinery as the other attention kernels): one block per (32-query
   tile, query head), 128 threads arranged 16 x 8. A thread owns 4 query rows, and
   NCH = ceil(D/64) ∈ {1, 2} is a template parameter. Shared memory is 20 KB:
   Q chunk [d][q] (loaded once when NCH = 1), K chunk XOR-swizzled or V chunk
   row-major, and P [k][q].
 - Query tiles are launched in reverse order, since early tiles have shorter windows.
At the benchmark: 128 x 16 = 2048 blocks, about 17 GFLOP.
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
swa_softcap(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
            float* __restrict__ out, int group, int S, int D, int W, float scale, float cap)
{
    __shared__ __align__(16) float Qc[DC * BQ];
    __shared__ __align__(16) float KV[DC * BKT];
    __shared__ __align__(16) float Pt[BKT * BQ];

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int nQT = (S + BQ - 1) / BQ;
    const int qt = nQT - 1 - blockIdx.x;
    const int h = blockIdx.y, g = h / group;
    const int q0 = qt * BQ;
    const float* Qh = Q + (size_t)h * S * D;
    const float* Kg = K + (size_t)g * S * D;
    const float* Vg = V + (size_t)g * S * D;

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
            Qc[t32_idx(dd, r)] = (q < S && dd < dw) ? __ldg(Qh + (size_t)q * D + db + dd) : 0.f;
        }
    };
    if (NCH == 1) load_q(0);                         // visible after the first barrier

    const int kmin = max(0, q0 - W + 1);
    const int kmax = min(S - 1, q0 + BQ - 1);
    for (int kt = kmin / BKT; kt <= kmax / BKT; ++kt) {
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
                KV[kc_idx(dd, r)] = (k < S && dd < dw) ? __ldg(Kg + (size_t)k * D + db + dd) : 0.f;
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
            const int qi = q0 + ty * 4 + i;
            float mx = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int k = k0 + tx + 16 * j;
                const bool ok = (k <= qi) && (qi - k < W) && (k < S);
                s[i][j] = ok ? cap * tanhf(__fdiv_rn(s[i][j] * scale, cap)) : -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            const float mNew = fmaxf(m[i], mx);
            float alpha = 1.f, sum = 0.f;
            if (mNew == -INFINITY) {                 // nothing visible yet for this row
                s[i][0] = s[i][1] = 0.f;
            } else {
                alpha = expf(m[i] - mNew);           // m = -inf -> 0
                #pragma unroll
                for (int j = 0; j < 2; ++j) { s[i][j] = expf(s[i][j] - mNew); sum += s[i][j]; }
            }
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
                KV[r * DC + dd] = (k < S && dd < dw) ? __ldg(Vg + (size_t)k * D + db + dd) : 0.f;
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
        if (q >= S) continue;
        const float inv = 1.f / l[i];                // the diagonal key is always visible
        float* orow = out + ((size_t)h * S + q) * D;
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
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim, int window_size,
                      float softcap)
{
    const int Hq = num_q_heads, Hkv = num_kv_heads, S = seq_len, D = head_dim, W = window_size;
    if (Hq <= 0 || Hkv <= 0 || S <= 0 || D <= 0 || W <= 0) return;
    const int group = Hq / Hkv;
    const float scale = (float)(1.0 / sqrt((double)D));
    dim3 grid((S + BQ - 1) / BQ, Hq);
    if (D <= 64) swa_softcap<1><<<grid, THREADS>>>(Q, K, V, output, group, S, D, W, scale, softcap);
    else         swa_softcap<2><<<grid, THREADS>>>(Q, K, V, output, group, S, D, W, scale, softcap);
    cudaDeviceSynchronize();
}
