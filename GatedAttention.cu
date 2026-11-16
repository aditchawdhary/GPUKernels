/*
Gated Attention (Qiu et al., 2025)
Q, K, V, X: (M, d_model); W_g: (d_model, d_model); h heads, d_k = d_model / h.
For head j (column block [j*d_k, (j+1)*d_k)):
  A_j = softmax( Q_j K_j^T / sqrt(d_k) + causal_mask ) V_j
  G_j = (X W_g)[:, block j]
  output[:, block j] = sigmoid(G_j) (.) A_j

Constraints
 - 1 ≤ M ≤ 4096, d_model ≤ 512 (divisible by h), h ≤ 8, all float32
 - Performance is measured with M = 4096, d_model = 512, h = 8 (d_k = 64)
*/

/*
Approach: one fused kernel. Each block does causal flash attention for one
(query tile, head) and then computes that tile's gate logits in its epilogue, so
X W_g is never written to memory.

 - Grid: (query tiles of 32, heads). Query tiles are launched in reverse order so
   the long causal rows start first. At the benchmark: 128 x 8 = 1024 blocks.
 - Block: 128 threads arranged 16 (tx) x 8 (ty). A thread owns 4 query rows. It
   computes S on key columns tx and tx + 16, and O (and G) on columns
   c*64 + 4tx .. +3 of each 64-wide chunk c of the head (NCH = ceil(d_k/64) chunks,
   templated on 1/2/4/8, so d_k up to 512 works).
 - Attention: key tiles of 32 up to the diagonal, with an FP32 online softmax
   (exp2, with scale*log2e folded into Q) and the causal mask on the diagonal tile.
   Q and K are staged per 64-wide d-chunk (Q only once when NCH = 1), and V per
   chunk. Shared memory:
     Qc[d][q]  transposed Q chunk (also reused for X^T in the epilogue)
     KVc       K chunk Kc[d][k] (transposed, XOR-swizzled), V chunk Vc[k][d],
               or the W_g chunk Wc[k][c] in the epilogue
     Pt[k][q]  probabilities, transposed and swizzled in float4 units
   That's 28 KB static, which launches anywhere with no opt-in.
 - Gate epilogue: for each output chunk c, G = X[q0:q0+32, :] @ W_g[:, cols of c],
   accumulated over d_model in 64-wide k-steps from staged X^T and W_g tiles, using
   the same per-thread register tile as O. Then out = O / l * sigmoid(G).
 - Everything accumulates in FP32. Gate cost is 2*M*d_model^2 FLOPs (about 2 GFLOP),
   small next to the attention's roughly 17 GFLOP.
*/

#include <cuda_runtime.h>
#include <math.h>

#define BQ 32
#define BKT 32
#define DC 64
#define THREADS 128

// Kc[d][k]: d*32 + (k ^ (d & 31))
__device__ __forceinline__ int kc_idx(int d, int k) { return d * BKT + (k ^ (d & 31)); }
// [row-major 32-wide, float4 groups swizzled by (row & 7)] used for Qc/Xt[d][q] and Pt[k][q]
__device__ __forceinline__ int t32_idx(int a, int q) { return a * 32 + ((((q >> 2) ^ (a & 7))) << 2) + (q & 3); }

template <int NCH>
__global__ void __launch_bounds__(THREADS)
gated_attn(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
           const float* __restrict__ X, const float* __restrict__ Wg, float* __restrict__ out,
           int M, int dm, int dk, float qscale)
{
    __shared__ __align__(16) float Qc[DC * BQ];       //  8 KB  (Qc / Xt)
    __shared__ __align__(16) float KVc[DC * DC];      // 16 KB  (Kc / Vc / Wc)
    __shared__ __align__(16) float Pt[BKT * BQ];      //  4 KB

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int nQT = (M + BQ - 1) / BQ;
    const int qt = nQT - 1 - blockIdx.x;              // heavy tiles first
    const int head = blockIdx.y;
    const int q0 = qt * BQ;
    const int hc = head * dk;                         // first column of this head

    float o[NCH][4][4];
    float m[4], l[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY; l[i] = 0.f;
        #pragma unroll
        for (int c = 0; c < NCH; ++c)
            #pragma unroll
            for (int j = 0; j < 4; ++j) o[c][i][j] = 0.f;
    }

    auto load_q_chunk = [&](int c) {                  // Qc[dd][r] = Q[q0+r][hc + c*64 + dd] * qscale
        const int db = c * DC, dw = min(DC, dk - db);
        for (int e = tid; e < BQ * DC; e += THREADS) {
            const int r = e / DC, dd = e % DC;
            const int q = q0 + r;
            Qc[t32_idx(dd, r)] = (q < M && dd < dw) ? __ldg(Q + (size_t)q * dm + hc + db + dd) * qscale : 0.f;
        }
    };

    if (NCH == 1) load_q_chunk(0);                    // made visible by the first barrier below

    // ================================================================ attention
    for (int kt = 0; kt <= qt; ++kt) {
        const int k0 = kt * BKT;

        float s[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int db = c * DC, dw = min(DC, dk - db);
            if (dw <= 0) break;
            __syncthreads();                          // KVc / Qc / Pt free
            if (NCH > 1) load_q_chunk(c);
            for (int e = tid; e < BKT * DC; e += THREADS) {
                const int r = e / DC, dd = e % DC;
                const int k = k0 + r;
                KVc[kc_idx(dd, r)] = (k < M && dd < dw) ? __ldg(K + (size_t)k * dm + hc + db + dd) : 0.f;
            }
            __syncthreads();
            #pragma unroll 4
            for (int dd = 0; dd < dw; ++dd) {
                const float4 qv = *reinterpret_cast<const float4*>(&Qc[t32_idx(dd, ty * 4)]);
                const float* kr = &KVc[dd * BKT];
                const int sw = dd & 31;
                const float k0v = kr[tx ^ sw], k1v = kr[(tx + 16) ^ sw];
                s[0][0] = fmaf(qv.x, k0v, s[0][0]); s[0][1] = fmaf(qv.x, k1v, s[0][1]);
                s[1][0] = fmaf(qv.y, k0v, s[1][0]); s[1][1] = fmaf(qv.y, k1v, s[1][1]);
                s[2][0] = fmaf(qv.z, k0v, s[2][0]); s[2][1] = fmaf(qv.z, k1v, s[2][1]);
                s[3][0] = fmaf(qv.w, k0v, s[3][0]); s[3][1] = fmaf(qv.w, k1v, s[3][1]);
            }
        }

        // ---- causal mask + online softmax -> Pt
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int qi = q0 + ty * 4 + i;
            float mx = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int k = k0 + tx + 16 * j;
                if (!(k <= qi && k < M)) s[i][j] = -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));

            const float mNew = fmaxf(m[i], mx);
            float alpha = 1.f, sum = 0.f;
            if (mNew == -INFINITY) {                  // row has nothing visible (only rows ≥ M)
                s[i][0] = s[i][1] = 0.f;
            } else {
                alpha = exp2f(m[i] - mNew);
                #pragma unroll
                for (int j = 0; j < 2; ++j) { s[i][j] = exp2f(s[i][j] - mNew); sum += s[i][j]; }
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

        // ---- O += P V, per chunk
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int db = c * DC, dw = min(DC, dk - db);
            if (dw <= 0) break;
            __syncthreads();                          // Kc reads done; Pt complete
            for (int e = tid; e < BKT * DC; e += THREADS) {
                const int r = e / DC, dd = e % DC;
                const int k = k0 + r;
                KVc[r * DC + dd] = (k < M && dd < dw) ? __ldg(V + (size_t)k * dm + hc + db + dd) : 0.f;
            }
            __syncthreads();
            #pragma unroll 8
            for (int k = 0; k < BKT; ++k) {
                const float4 pv = *reinterpret_cast<const float4*>(&Pt[t32_idx(k, ty * 4)]);
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

    // ================================================================ gate epilogue
    #pragma unroll
    for (int c = 0; c < NCH; ++c) {
        const int cb = c * DC, cw = min(DC, dk - cb);   // output columns hc+cb .. +cw
        if (cw <= 0) break;

        float g[4][4] = {{0.f}};
        for (int kb = 0; kb < dm; kb += DC) {
            const int kw = min(DC, dm - kb);
            __syncthreads();                          // Qc / KVc free
            for (int e = tid; e < BQ * DC; e += THREADS) {   // Xt[kk][r] = X[q0+r][kb+kk]
                const int r = e / DC, kk = e % DC;
                const int q = q0 + r;
                Qc[t32_idx(kk, r)] = (q < M && kk < kw) ? __ldg(X + (size_t)q * dm + kb + kk) : 0.f;
            }
            for (int e = tid; e < DC * DC; e += THREADS) {   // Wc[kk][col] = Wg[kb+kk][hc+cb+col]
                const int kk = e / DC, col = e % DC;
                KVc[kk * DC + col] = (kk < kw && col < cw) ? __ldg(Wg + (size_t)(kb + kk) * dm + hc + cb + col) : 0.f;
            }
            __syncthreads();
            #pragma unroll 8
            for (int kk = 0; kk < kw; ++kk) {
                const float4 xv = *reinterpret_cast<const float4*>(&Qc[t32_idx(kk, ty * 4)]);
                const float4 wv = *reinterpret_cast<const float4*>(&KVc[kk * DC + tx * 4]);
                const float xa[4] = {xv.x, xv.y, xv.z, xv.w};
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    g[i][0] = fmaf(xa[i], wv.x, g[i][0]);
                    g[i][1] = fmaf(xa[i], wv.y, g[i][1]);
                    g[i][2] = fmaf(xa[i], wv.z, g[i][2]);
                    g[i][3] = fmaf(xa[i], wv.w, g[i][3]);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = q0 + ty * 4 + i;
            if (q >= M) continue;
            const float inv = 1.f / l[i];             // l ≥ 1 for real rows (diagonal is visible)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = cb + tx * 4 + j;
                if (col < dk) {
                    const float gate = 1.f / (1.f + expf(-g[i][j]));
                    out[(size_t)q * dm + hc + col] = o[c][i][j] * inv * gate;
                }
            }
        }
    }
}

// Q, K, V, X, W_g, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* X,
                      const float* W_g, float* output, int M, int d_model, int h)
{
    if (M <= 0 || d_model <= 0 || h <= 0) return;
    const int dk = d_model / h;
    const float qscale = (1.0f / sqrtf((float)dk)) * 1.4426950408889634f;   // * log2(e)
    const int nQT = (M + BQ - 1) / BQ;
    dim3 grid(nQT, h);

    if      (dk <= 64)  gated_attn<1><<<grid, THREADS>>>(Q, K, V, X, W_g, output, M, d_model, dk, qscale);
    else if (dk <= 128) gated_attn<2><<<grid, THREADS>>>(Q, K, V, X, W_g, output, M, d_model, dk, qscale);
    else if (dk <= 256) gated_attn<4><<<grid, THREADS>>>(Q, K, V, X, W_g, output, M, d_model, dk, qscale);
    else                gated_attn<8><<<grid, THREADS>>>(Q, K, V, X, W_g, output, M, d_model, dk, qscale);
    cudaDeviceSynchronize();
}
