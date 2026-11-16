/*
Shared-Prefix Cascade Attention
Q, K_suffix, V_suffix, output : (B, H, S, D)      K_prefix, V_prefix : (H, P, D)
Suffix query i of sequence b, head h attends to all P prefix keys of head h and
causally to suffix keys 0..i of its own sequence. scale = 1/sqrt(D). float32.

Constraints
 - 1 ≤ B ≤ 64, 1 ≤ H ≤ 64, 0 ≤ P ≤ 8192, 1 ≤ S ≤ 1024, 1 ≤ D ≤ 128
 - tolerance 1e-4 abs/rel
 - Performance is measured with B = 16, H = 32, P = 4096, S = 256, D = 128
*/

/*
Approach: an FP32 flash-attention kernel that streams the prefix and then the
suffix through a single online softmax, with blocks ordered so the shared prefix
is read about once per head.

 - Merge: the prefix pass and the causal suffix pass are one key stream run
   through the same running (max, sum, O) state. This is exactly the cascade
   merge of the two partial softmaxes, done in registers, with no partial
   outputs written to memory.
 - Prefix reuse: block id = (h, b, query tile), with h slowest. Blocks that run
   at the same time mostly share a head, so each head's K/V prefix (4 MB at
   P = 4096, D = 128) is pulled from DRAM roughly once and then served from L2
   to all B sequences. This avoids re-reading the prefix once per sequence.
 - Tiles: 32 queries x 32 keys, 128 threads arranged 16 (tx) x 8 (ty). Each
   thread owns 4 query rows and computes S on key columns tx and tx + 16, then O
   on columns 4tx..4tx+3 and 64+4tx..64+4tx+3.
     Qs[d][q]  transposed: float4 broadcast loads of a thread's 4 rows
     Ks[d][k]  transposed, XOR-swizzled: conflict-free transposing stores and loads
     Vs[k][d]  row-major: float4 loads
     Pt[k][q]  transposed probabilities, XOR-swizzled in float4 units: float4 loads
   K and V share one buffer (K -> S -> P, then V -> PV). That keeps shared
   memory at 36 KB of static allocation, under the 48 KB default limit, so it
   launches on any GPU (T4 included) with no opt-in.
 - Softmax uses exp2 with scale*log2(e) folded into Q. Row max and row sum are
   reduced over the 16 lanes that share a row with __shfl_xor. Rows with every
   key masked so far are guarded against -inf - -inf.
 - Causal: query tile q0 only visits suffix key tiles below q0 + 32. The mask
   applies j ≤ i and j < S.
 - Everything is FP32, not TF32, to meet the 1e-4 tolerance.
*/

#include <cuda_runtime.h>
#include <math.h>

#define BQ 32
#define BKT 32
#define DMAX 128
#define THREADS 128           // 16 x 8
#define QS BQ                 // Qs / Pt row stride
#define KS BKT                // Ks row stride (XOR-swizzled)
#define VS DMAX               // Vs row stride
#define KV_FLOATS (BKT * VS)  // == DMAX * KS = 4096
// total static smem: (128*32 + 4096 + 32*32) * 4 B = 36 KB

// Ks[d][k] stored at d*32 + (k ^ (d & 31)): conflict-free for both the
// transposing store (fixed k, consecutive d) and the reads (fixed d, consecutive k).
__device__ __forceinline__ int ks_idx(int d, int k) { return d * KS + (k ^ (d & 31)); }
// Pt[k][q] stored with its float4 column group XORed by (k & 7): float4 reads
// stay contiguous, and the transposing stores are at most 2-way conflicted.
__device__ __forceinline__ int pt_idx(int k, int q) { return k * QS + ((((q >> 2) ^ (k & 7))) << 2) + (q & 3); }

__global__ void __launch_bounds__(THREADS)
cascade_attn(const float* __restrict__ Q,
             const float* __restrict__ Kp, const float* __restrict__ Vp,
             const float* __restrict__ Ks_g, const float* __restrict__ Vs_g,
             float* __restrict__ out,
             int B, int H, int P, int S, int D, float qscale)
{
    __shared__ __align__(16) float Qs[DMAX * QS];    // [DMAX][QS]
    __shared__ __align__(16) float KV[KV_FLOATS];    // Ks [DMAX][KS]  or  Vs [BKT][VS]
    __shared__ __align__(16) float Pt[BKT * QS];     // [BKT][QS]

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int nQT = (S + BQ - 1) / BQ;

    int bid = blockIdx.x;
    const int qt = bid % nQT;  bid /= nQT;
    const int b  = bid % B;    bid /= B;
    const int h  = bid;
    const int q0 = qt * BQ;

    const size_t seqBase = ((size_t)b * H + h) * S;   // row offset into (B,H,S,*)
    const float* Qg = Q    + seqBase * D;
    const float* Kp_h = Kp + (size_t)h * P * D;
    const float* Vp_h = Vp + (size_t)h * P * D;
    const float* Ks_h = Ks_g + seqBase * D;
    const float* Vs_h = Vs_g + seqBase * D;

    // ---- load Q tile (transposed, pre-scaled by scale*log2e), zero for rows ≥ S
    for (int e = tid; e < BQ * D; e += THREADS) {
        const int r = e / D, d = e - r * D;
        const int q = q0 + r;
        Qs[d * QS + r] = (q < S) ? Qg[(size_t)q * D + d] * qscale : 0.f;
    }

    float o[4][8];
    float m[4], l[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY; l[i] = 0.f;
        #pragma unroll
        for (int c = 0; c < 8; ++c) o[i][c] = 0.f;
    }

    const int nP   = (P + BKT - 1) / BKT;
    const int sufKeys = min(S, q0 + BQ);               // causal upper bound for this tile
    const int nSuf = (sufKeys + BKT - 1) / BKT;
    const int nT   = nP + nSuf;

    for (int t = 0; t < nT; ++t) {
        const bool pre = t < nP;
        const int  k0  = pre ? t * BKT : (t - nP) * BKT;
        const int  kLim = pre ? P : S;
        const float* Kg = pre ? Kp_h : Ks_h;
        const float* Vg = pre ? Vp_h : Vs_h;

        __syncthreads();                               // previous PV done with KV / Pt
        // ---- K tile -> Ks[d][k]
        for (int e = tid; e < BKT * D; e += THREADS) {
            const int r = e / D, d = e - r * D;
            const int k = k0 + r;
            KV[ks_idx(d, r)] = (k < kLim) ? __ldg(Kg + (size_t)k * D + d) : 0.f;
        }
        __syncthreads();

        // ---- S = Q K^T  (4 rows x 2 cols per thread)
        float s[4][2];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j) s[i][j] = 0.f;

        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            const float4 qv = *reinterpret_cast<const float4*>(&Qs[d * QS + ty * 4]);
            const float* kr = &KV[d * KS];
            const int sw = d & 31;
            const float kv0 = kr[tx ^ sw], kv1 = kr[(tx + 16) ^ sw];
            const float qa[4] = {qv.x, qv.y, qv.z, qv.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[i][0] = fmaf(qa[i], kv0, s[i][0]);
                s[i][1] = fmaf(qa[i], kv1, s[i][1]);
            }
        }

        // ---- mask + online softmax update
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int qi = q0 + ty * 4 + i;
            float mx = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int k = k0 + tx + 16 * j;
                const bool valid = pre ? (k < P) : (k < S && k <= qi);
                if (!valid) s[i][j] = -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));

            const float mNew = fmaxf(m[i], mx);
            float alpha, sum = 0.f;
            if (mNew == -INFINITY) {                   // nothing visible yet
                alpha = 1.f;
                #pragma unroll
                for (int j = 0; j < 2; ++j) s[i][j] = 0.f;
            } else {
                alpha = exp2f(m[i] - mNew);            // m = -inf -> 0
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    s[i][j] = exp2f(s[i][j] - mNew);   // masked -> 0
                    sum += s[i][j];
                }
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                sum += __shfl_xor_sync(0xffffffff, sum, off);

            m[i] = mNew;
            l[i] = l[i] * alpha + sum;
            #pragma unroll
            for (int c = 0; c < 8; ++c) o[i][c] *= alpha;

            #pragma unroll
            for (int j = 0; j < 2; ++j) Pt[pt_idx(tx + 16 * j, ty * 4 + i)] = s[i][j];
        }
        __syncthreads();                               // Ks no longer needed; Pt ready

        // ---- V tile -> Vs[k][d]  (zero rows past kLim and cols ≥ D)
        for (int e = tid; e < BKT * DMAX; e += THREADS) {
            const int r = e >> 7, d = e & (DMAX - 1);
            const int k = k0 + r;
            KV[r * VS + d] = (k < kLim && d < D) ? __ldg(Vg + (size_t)k * D + d) : 0.f;
        }
        __syncthreads();

        // ---- O += P V
        #pragma unroll 4
        for (int k = 0; k < BKT; ++k) {
            const float4 pv = *reinterpret_cast<const float4*>(&Pt[pt_idx(k, ty * 4)]);
            const float4 v0 = *reinterpret_cast<const float4*>(&KV[k * VS + tx * 4]);
            const float4 v1 = *reinterpret_cast<const float4*>(&KV[k * VS + 64 + tx * 4]);
            const float pa[4] = {pv.x, pv.y, pv.z, pv.w};
            const float va[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int c = 0; c < 8; ++c) o[i][c] = fmaf(pa[i], va[c], o[i][c]);
        }
    }

    // ---- normalize and store
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = q0 + ty * 4 + i;
        if (q >= S) continue;
        const float inv = (l[i] > 0.f) ? 1.f / l[i] : 0.f;
        float* orow = out + (seqBase + q) * D;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int d = (c < 4) ? (tx * 4 + c) : (64 + tx * 4 + (c - 4));
            if (d < D) orow[d] = o[i][c] * inv;
        }
    }
}

// Q, K_prefix, V_prefix, K_suffix, V_suffix, output are device pointers
extern "C" void solve(const float* Q, const float* K_prefix, const float* V_prefix,
                      const float* K_suffix, const float* V_suffix, float* output, int batch_size,
                      int num_heads, int prefix_len, int suffix_len, int head_dim)
{
    const int B = batch_size, H = num_heads, P = prefix_len, S = suffix_len, D = head_dim;
    if (B <= 0 || H <= 0 || S <= 0 || D <= 0) return;

    const float qscale = (1.0f / sqrtf((float)D)) * 1.4426950408889634f;   // * log2(e)
    const int nQT = (S + BQ - 1) / BQ;
    const unsigned grid = (unsigned)(H * B * nQT);     // h slowest -> prefix L2 reuse

    cascade_attn<<<grid, THREADS>>>(Q, K_prefix, V_prefix, K_suffix, V_suffix, output,
                                    B, H, P, S, D, qscale);
    cudaDeviceSynchronize();
}
