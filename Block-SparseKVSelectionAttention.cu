/*
Block-Sparse KV Selection Attention (NSA / Quest-style decode)
Q (H, D); K, V (H, S, D); output (H, D); float32. Spec from the reference implementation
(leetgpu-challenges #119), tolerance 1e-4:
  blocks of block_size positions (the tail block may be partial)
  block_score[h,b] = Q[h] . mean_{p in b} K[h,p]        (raw dot product, no scale)
  keep the min(num_selected, NB) best blocks: a higher score wins, ties go to the lower
  block index
  out[h] = softmax over kept positions of (Q[h] . K[h,p]) / sqrt(D), applied to V
Constraints: H ≤ 64, S ≤ 8192, D ≤ 128, block_size ≤ 256.
Performance at H = 32, S = 8192, D = 128, block_size = 64, num_selected = 32.
*/

/*
Approach (4 launches, FP32):
 1. block_scores  grid (NB, H), 128 threads = dims. Thread d sums K over the block's
                  positions (coalesced across d), divides by the real count (the tail
                  block averages only existing positions), multiplies by Q[h, d], and
                  a block reduction gives the score. K is read once, in full.
 2. rank_select   grid (blocks of 256, H). rank_b = #{b' : s_b' > s_b, or s_b' == s_b
                  and b' < b}, exactly the reference's deterministic order. The
                  scores are staged through shared memory in chunks. Kept blocks have
                  unique ranks 0..k-1, so list[h][rank] = b builds the compact
                  selected-block list with no extra scan.
 3. sparse_attn   flash decoding over only the selected blocks: grid (splits, H),
                  128 threads = 4 warps. Each warp takes keys of its split in turn.
                  Lanes stride over d (D ≤ 128, so ≤ 4 dims per lane), a warp-shuffle
                  sum gives the score, followed by an online softmax and o += p * v in
                  registers. The 4 warps merge through shared memory and write a
                  partial (m, l, o[D]).
 4. combine       one block per head merges the splits by log-sum-exp and normalizes.
 Traffic at the benchmark: K read once for scoring (134 MB) plus only the 25% of K
 and V in the kept blocks (67 MB), so it's bandwidth-bound, on the order of 100 µs.
 Ties (zero query or zero keys) select the lowest block indices, matching the
 reference.
*/

#include <cuda_runtime.h>
#include <math.h>

#define DT 128                        // threads for per-dim kernels (D ≤ 128)
#define RT 256                        // rank_select threads
#define AT 128                        // sparse_attn threads (4 warps)
#define AW (AT / 32)

// ======================================================================== 1. block scores
__global__ void __launch_bounds__(DT)
block_scores(const float* __restrict__ Q, const float* __restrict__ K, float* __restrict__ score,
             int S, int D, int bs, int NB)
{
    __shared__ float ws[DT / 32];
    const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int p0 = b * bs, p1 = min(p0 + bs, S);
    float v = 0.f;
    if (d < D) {
        const float* kp = K + ((size_t)h * S + p0) * D + d;
        float s = 0.f;
        for (int p = p0; p < p1; ++p, kp += D) s += __ldg(kp);
        v = (s / (float)(p1 - p0)) * __ldg(Q + (size_t)h * D + d);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if ((d & 31) == 0) ws[d >> 5] = v;
    __syncthreads();
    if (d == 0) score[(size_t)h * NB + b] = ws[0] + ws[1] + ws[2] + ws[3];
}

// ======================================================================== 2. rank / select
__global__ void __launch_bounds__(RT)
rank_select(const float* __restrict__ score, int* __restrict__ list, int NB, int keep)
{
    __shared__ float sc[RT];
    const int h = blockIdx.y;
    const int b = blockIdx.x * RT + threadIdx.x;
    const float* s = score + (size_t)h * NB;
    const float mine = (b < NB) ? s[b] : 0.f;
    int rank = 0;
    for (int base = 0; base < NB; base += RT) {
        const int j = base + threadIdx.x;
        sc[threadIdx.x] = (j < NB) ? s[j] : 0.f;
        __syncthreads();
        const int n = min(RT, NB - base);
        for (int t = 0; t < n; ++t) {
            const float o = sc[t];
            rank += (o > mine) || (o == mine && base + t < b);
        }
        __syncthreads();
    }
    if (b < NB && rank < keep) list[(size_t)h * NB + rank] = b;
}

// ======================================================================== 3. sparse attention (partials)
__global__ void __launch_bounds__(AT)
sparse_attn(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
            const int* __restrict__ list, float* __restrict__ part,
            int S, int D, int bs, int NB, int keep, int perSplit, int nSplit, float scale)
{
    __shared__ float wm[AW], wl[AW];
    __shared__ float wo[AW][128];
    const int sp = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int e0 = sp * perSplit, e1 = min(e0 + perSplit, keep);

    float q[4], o[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = lane + 32 * j;
        q[j] = (d < D) ? __ldg(Q + (size_t)h * D + d) : 0.f;
    }
    float m = -INFINITY, l = 0.f;

    // flattened (entry, offset) iteration; each warp takes every AW-th position
    for (int e = e0; e < e1; ++e) {
        const int blk = list[(size_t)h * NB + e];
        const int p0 = blk * bs, p1 = min(p0 + bs, S);
        for (int p = p0 + warp; p < p1; p += AW) {
            const float* kr = K + ((size_t)h * S + p) * D;
            const float* vr = V + ((size_t)h * S + p) * D;
            float dot = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = lane + 32 * j;
                if (d < D) dot = fmaf(q[j], __ldg(kr + d), dot);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
            const float sc = dot * scale;
            const float mNew = fmaxf(m, sc);
            const float alpha = expf(m - mNew);          // m = -inf -> 0
            const float pw = expf(sc - mNew);
            l = l * alpha + pw;
            m = mNew;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = lane + 32 * j;
                const float vv = (d < D) ? __ldg(vr + d) : 0.f;
                o[j] = fmaf(pw, vv, o[j] * alpha);
            }
        }
    }

    // merge the 4 warps
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = lane + 32 * j;
        if (d < 128) wo[warp][d] = o[j];
    }
    __syncthreads();
    float M = -INFINITY;
    #pragma unroll
    for (int w = 0; w < AW; ++w) M = fmaxf(M, wm[w]);
    float* out = part + ((size_t)h * nSplit + sp) * (D + 2);
    if (threadIdx.x == 0) {
        float L = 0.f;
        for (int w = 0; w < AW; ++w) if (wm[w] != -INFINITY) L += wl[w] * expf(wm[w] - M);
        out[0] = M; out[1] = L;
    }
    for (int d = threadIdx.x; d < D; d += AT) {
        float acc = 0.f;
        for (int w = 0; w < AW; ++w) if (wm[w] != -INFINITY) acc = fmaf(wo[w][d], expf(wm[w] - M), acc);
        out[2 + d] = acc;
    }
}

// ======================================================================== 4. combine
__global__ void combine(const float* __restrict__ part, float* __restrict__ out, int D, int nSplit)
{
    const int h = blockIdx.x;
    const float* p = part + (size_t)h * nSplit * (D + 2);
    float M = -INFINITY;
    for (int s = 0; s < nSplit; ++s) M = fmaxf(M, p[(size_t)s * (D + 2)]);
    float L = 0.f;
    for (int s = 0; s < nSplit; ++s) {
        const float ms = p[(size_t)s * (D + 2)];
        if (ms != -INFINITY) L += p[(size_t)s * (D + 2) + 1] * expf(ms - M);
    }
    const float inv = 1.f / L;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < nSplit; ++s) {
            const float ms = p[(size_t)s * (D + 2)];
            if (ms != -INFINITY) acc = fmaf(p[(size_t)s * (D + 2) + 2 + d], expf(ms - M), acc);
        }
        out[(size_t)h * D + d] = acc * inv;
    }
}

// ======================================================================== host
static void*  g_ws = nullptr;
static size_t g_bytes = 0;

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int num_heads,
                      int seq_len, int head_dim, int block_size, int num_selected)
{
    const int H = num_heads, S = seq_len, D = head_dim, bs = block_size;
    if (H <= 0 || S <= 0 || D <= 0 || bs <= 0) return;
    const int NB = (S + bs - 1) / bs;
    const int keep = num_selected < NB ? num_selected : NB;

    int nSplit = (256 + H - 1) / H;                    // aim for ≥ 256 attention blocks
    if (nSplit > keep) nSplit = keep;
    if (nSplit < 1) nSplit = 1;
    const int perSplit = (keep + nSplit - 1) / nSplit;
    nSplit = (keep + perSplit - 1) / perSplit;

    const size_t need = sizeof(float) * ((size_t)H * NB + (size_t)H * nSplit * (D + 2))
                      + sizeof(int) * (size_t)H * NB + 16;
    if (need > g_bytes) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, need);
        g_bytes = need;
    }
    float* score = reinterpret_cast<float*>(g_ws);
    float* part  = score + (size_t)H * NB;
    int*   list  = reinterpret_cast<int*>(part + (size_t)H * nSplit * (D + 2));

    const float scale = (float)(1.0 / sqrt((double)D));
    block_scores<<<dim3(NB, H), DT>>>(Q, K, score, S, D, bs, NB);
    rank_select<<<dim3((NB + RT - 1) / RT, H), RT>>>(score, list, NB, keep);
    sparse_attn<<<dim3(nSplit, H), AT>>>(Q, K, V, list, part, S, D, bs, NB, keep, perSplit, nSplit, scale);
    combine<<<H, 128>>>(part, output, D, nSplit);
    cudaDeviceSynchronize();
}
