/*
Variable-Length Causal Attention (packed / "varlen" batch)
Q, K, V, output: (T, d) float32; cu_seqlens: (S + 1) int32, sequence s = rows
[cu[s], cu[s+1]). A query attends only to keys of its own sequence at positions
≤ its own (causal), with scale 1/sqrt(d). (Spec: leetgpu-challenges #102,
tolerance 1e-4.)

Constraints: S ≤ 256, each length ≤ 2048, T ≤ 16384, d ≤ 256.
Performance is measured with T = 8192, d = 64, S = 32 (lengths 128-384).
*/

/*
Approach: an FP32 flash-attention kernel with a device-side varlen tile schedule.
It never builds the T x T score matrix, and it never touches keys from other
sequences or keys above the diagonal.

 - Schedule: sequence s has ceil(len_s / 32) query tiles. Each block builds the
   tile prefix over cu_seqlens in shared memory and maps blockIdx.x to
   (sequence, query tile). The grid is the upper bound ceil(T/32) + S, and spare
   blocks exit at once, so cu_seqlens never has to be copied to the host.
 - Work per block: query tile qt of a sequence visits key tiles 0..qt of the same
   sequence (tiles start at the sequence start). The diagonal tile applies
   k_local ≤ q_local, and the last tile also applies k < len.
 - Kernel body (same structure as the gated-attention kernel): 32 queries x 32
   keys, 128 threads arranged 16 x 8. Each thread owns 4 rows and 4 output
   columns per 64-wide d-chunk. NCH = ceil(d / 64) is a template parameter
   (1/2/4), so d up to 256 works. FP32 online softmax with exp2 and
   scale*log2(e) folded into Q.
   Shared memory (20 KB): Qc[d][q] (loaded once when NCH = 1), KV holding the
   K chunk transposed with an XOR swizzle or the V chunk row-major, and the
   probabilities Pt[k][q] swizzled.
At the benchmark: about 270 query tiles. Causal work is about
sum(len^2)/2 * d * 4 ≈ 0.6 GFLOP, so it finishes in well under a millisecond.
*/

#include <cuda_runtime.h>
#include <math.h>

#define BQ 32
#define BKT 32
#define DC 64
#define THREADS 128
#define MAX_S 256

__device__ __forceinline__ int kc_idx(int d, int k) { return d * BKT + (k ^ (d & 31)); }
__device__ __forceinline__ int t32_idx(int a, int q) { return a * 32 + ((((q >> 2) ^ (a & 7))) << 2) + (q & 3); }

template <int NCH>
__global__ void __launch_bounds__(THREADS)
varlen_attn(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
            const int* __restrict__ cu, float* __restrict__ out, int D, int S, float qscale)
{
    __shared__ __align__(16) float Qc[DC * BQ];
    __shared__ __align__(16) float KV[DC * BKT];
    __shared__ __align__(16) float Pt[BKT * BQ];
    __shared__ int tileStart[MAX_S + 1];
    __shared__ int info[3];                      // seq start row, seq length, query tile

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;

    // ---- schedule: blockIdx.x -> (sequence, query tile)
    if (tid == 0) {
        int acc = 0;
        for (int s = 0; s < S; ++s) {
            tileStart[s] = acc;
            acc += (cu[s + 1] - cu[s] + BQ - 1) / BQ;
        }
        tileStart[S] = acc;
        const int t = blockIdx.x;
        info[2] = -1;
        if (t < acc) {
            int s = 0;
            while (tileStart[s + 1] <= t) ++s;     // skips empty sequences too
            info[0] = cu[s];
            info[1] = cu[s + 1] - cu[s];
            info[2] = t - tileStart[s];
        }
    }
    __syncthreads();
    const int qt = info[2];
    if (qt < 0) return;                            // spare block (uniform)
    const int base = info[0], L = info[1];
    const int q0 = qt * BQ;                        // local query offset

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
            Qc[t32_idx(dd, r)] = (q < L && dd < dw) ? __ldg(Q + (size_t)(base + q) * D + db + dd) * qscale : 0.f;
        }
    };
    if (NCH == 1) load_q(0);                       // visible after the first barrier below

    for (int kt = 0; kt <= qt; ++kt) {
        const int k0 = kt * BKT;                   // local key offset

        float s[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();
            if (NCH > 1) load_q(c);
            for (int e = tid; e < BKT * DC; e += THREADS) {
                const int r = e / DC, dd = e % DC, k = k0 + r;
                KV[kc_idx(dd, r)] = (k < L && dd < dw) ? __ldg(K + (size_t)(base + k) * D + db + dd) : 0.f;
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
                if (!(k <= qi && k < L)) s[i][j] = -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            const float mNew = fmaxf(m[i], mx);
            float alpha = 1.f, sum = 0.f;
            if (mNew == -INFINITY) {
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

        #pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int db = c * DC, dw = min(DC, D - db);
            if (dw <= 0) break;
            __syncthreads();                       // K reads done, Pt complete
            for (int e = tid; e < BKT * DC; e += THREADS) {
                const int r = e / DC, dd = e % DC, k = k0 + r;
                KV[r * DC + dd] = (k < L && dd < dw) ? __ldg(V + (size_t)(base + k) * D + db + dd) : 0.f;
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
        if (q >= L) continue;
        const float inv = 1.f / l[i];              // diagonal key always visible -> l > 0
        float* orow = out + (size_t)(base + q) * D;
        #pragma unroll
        for (int c = 0; c < NCH; ++c)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = c * DC + tx * 4 + j;
                if (col < D) orow[col] = o[c][i][j] * inv;
            }
    }
}

// Q, K, V, cu_seqlens, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const int* cu_seqlens,
                      float* output, int T, int d, int S)
{
    if (T <= 0 || d <= 0 || S <= 0) return;
    const float qscale = (1.0f / sqrtf((float)d)) * 1.4426950408889634f;   // * log2(e)
    const int grid = (T + BQ - 1) / BQ + S;            // ≥ total query tiles
    if      (d <= 64)  varlen_attn<1><<<grid, THREADS>>>(Q, K, V, cu_seqlens, output, d, S, qscale);
    else if (d <= 128) varlen_attn<2><<<grid, THREADS>>>(Q, K, V, cu_seqlens, output, d, S, qscale);
    else               varlen_attn<4><<<grid, THREADS>>>(Q, K, V, cu_seqlens, output, d, S, qscale);
    cudaDeviceSynchronize();
}
