#include <cuda_runtime.h>
#include <math.h>

// Tree-speculative verification attention.
// Node i (query Q[i]) attends to all P context tokens (K_cache/V_cache) plus every tree node on
// its root-to-node path, A(i) (K_tree/V_tree); softmax(q.k / sqrt(D)) weighted sum of values.
//
// Keys live in one combined index space j in [0, P + T): j < P -> context, j >= P -> tree node.
// Stage 1 (split-KV flash attention): block = (key split, tile of BR queries, head).
//   * the ancestor-or-self set of each query in the tile is a 128-bit mask in shared memory,
//     built by walking `parents` (T <= 128);
//   * the block streams its key range in tiles of BC keys: S = (Q*scale) K^T, mask, online
//     softmax (running max m and sum l), O = O*alpha + P V; all fp32;
//   * writes unnormalised (O, m, l) partials for its split.
// Stage 2: per (query, head), merge the splits: M = max m_s, L = sum l_s e^(m_s-M),
//          out = sum O_s e^(m_s-M) / L. Every query sees at least itself, so L > 0.

#define BR 32                 // queries per block
#define BC 32                 // keys per tile
#define THREADS 256
#define TPR 8                 // threads per query row
#define KPT (BC / TPR)        // keys per thread in the score phase
#define SPLIT_KEYS 512        // keys per split (multiple of BC)
#define MAXT 128
#define MASK_WORDS (MAXT / 32)

template <int NJ>   // NJ = ceil(D / 8) rounded to 8/16/32: output columns per thread
__global__ void __launch_bounds__(THREADS)
tree_attn_partial(const float* __restrict__ Q, const float* __restrict__ Kc,
                  const float* __restrict__ Vc, const float* __restrict__ Kt,
                  const float* __restrict__ Vt, const int* __restrict__ parents,
                  float* __restrict__ pO, float* __restrict__ pM, float* __restrict__ pL,
                  int P, int T, int H, int D, float scale) {
    extern __shared__ __align__(16) float smem[];
    const int LD = D + 1;                      // odd stride: conflict-free scalar column reads
    float* Qs = smem;                          // [BR][LD]
    float* Ks = Qs + BR * LD;                  // [BC][LD]
    float* Vs = Ks + BC * LD;                  // [BC][LD]
    float* Ps = Vs + BC * LD;                  // [BR][BC+1]
    __shared__ unsigned anc[BR][MASK_WORDS];

    const int split = blockIdx.x;
    const int row0 = blockIdx.y * BR;
    const int h = blockIdx.z;
    const int total_keys = P + T;
    const int kbeg = split * SPLIT_KEYS;
    const int kend = min(total_keys, kbeg + SPLIT_KEYS);
    const size_t HD = (size_t)H * D;

    const int tid = threadIdx.x;
    const int r = tid / TPR;
    const int lane8 = tid % TPR;
    const int qi = row0 + r;
    const bool row_valid = qi < T;

    // ancestor-or-self masks for the tile's queries
    if (tid < BR) {
        unsigned m[MASK_WORDS] = {0u, 0u, 0u, 0u};
        int j = row0 + tid;
        if (j < T) {
            while (j >= 0) { m[j >> 5] |= 1u << (j & 31); j = parents[j]; }
        }
#pragma unroll
        for (int w = 0; w < MASK_WORDS; w++) anc[tid][w] = m[w];
    }
    // Q tile, pre-scaled
    for (int idx = tid; idx < BR * D; idx += THREADS) {
        int rr = idx / D, d = idx % D;
        int q = row0 + rr;
        Qs[rr * LD + d] = (q < T) ? Q[(size_t)q * HD + (size_t)h * D + d] * scale : 0.0f;
    }

    float m = -INFINITY, l = 0.0f;
    float acc[NJ];
#pragma unroll
    for (int j = 0; j < NJ; j++) acc[j] = 0.0f;

    for (int k0 = kbeg; k0 < kend; k0 += BC) {
        __syncthreads();                       // previous tile consumed (and anc/Qs ready)
        for (int idx = tid; idx < BC * D; idx += THREADS) {
            int c = idx / D, d = idx % D;
            int k = k0 + c;
            float kv = 0.0f, vv = 0.0f;
            if (k < kend) {
                size_t off;
                if (k < P) { off = (size_t)k * HD + (size_t)h * D + d; kv = __ldg(&Kc[off]); vv = __ldg(&Vc[off]); }
                else       { off = (size_t)(k - P) * HD + (size_t)h * D + d; kv = __ldg(&Kt[off]); vv = __ldg(&Vt[off]); }
            }
            Ks[c * LD + d] = kv;
            Vs[c * LD + d] = vv;
        }
        __syncthreads();

        // scores for row r, keys lane8 + 8*j
        float sc[KPT];
#pragma unroll
        for (int j = 0; j < KPT; j++) sc[j] = 0.0f;
        const float* qrow = Qs + r * LD;
        for (int d = 0; d < D; d++) {
            float qv = qrow[d];
#pragma unroll
            for (int j = 0; j < KPT; j++) sc[j] = fmaf(qv, Ks[(lane8 + TPR * j) * LD + d], sc[j]);
        }

        // mask: context keys always visible; tree key n visible iff n in A(qi)
        float tmax = -INFINITY;
#pragma unroll
        for (int j = 0; j < KPT; j++) {
            int k = k0 + lane8 + TPR * j;
            bool ok = row_valid && k < kend;
            if (ok && k >= P) {
                int n = k - P;
                ok = (anc[r][n >> 5] >> (n & 31)) & 1u;
            }
            if (!ok) sc[j] = -INFINITY;
            tmax = fmaxf(tmax, sc[j]);
        }
#pragma unroll
        for (int o = TPR / 2; o > 0; o >>= 1)
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));

        float m_new = fmaxf(m, tmax);
        float m_use = (m_new == -INFINITY) ? 0.0f : m_new;     // fully-masked guard
        float alpha = __expf(m - m_use);
        float psum = 0.0f;
#pragma unroll
        for (int j = 0; j < KPT; j++) {
            float p = __expf(sc[j] - m_use);
            psum += p;
            Ps[r * (BC + 1) + lane8 + TPR * j] = p;
        }
#pragma unroll
        for (int o = TPR / 2; o > 0; o >>= 1)
            psum += __shfl_xor_sync(0xffffffffu, psum, o);
        l = l * alpha + psum;
        m = m_new;
        __syncwarp();   // row r's P values are produced and consumed within one warp

#pragma unroll
        for (int j = 0; j < NJ; j++) acc[j] *= alpha;
        const float* prow = Ps + r * (BC + 1);
        const int cmax = min(BC, kend - k0);
        for (int c = 0; c < cmax; c++) {
            float p = prow[c];
            const float* vrow = Vs + c * LD;
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                int col = lane8 + TPR * j;
                if (col < D) acc[j] = fmaf(p, vrow[col], acc[j]);
            }
        }
    }

    if (row_valid) {
        size_t pi = ((size_t)split * T + qi) * H + h;     // partial index
        float* o = pO + pi * D;
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            int col = lane8 + TPR * j;
            if (col < D) o[col] = acc[j];
        }
        if (lane8 == 0) { pM[pi] = m; pL[pi] = l; }
    }
}

// one block per (query, head); thread d handles output column d
__global__ void tree_attn_combine(const float* __restrict__ pO, const float* __restrict__ pM,
                                  const float* __restrict__ pL, float* __restrict__ out,
                                  int nsplit, int T, int H, int D) {
    const int th = blockIdx.x;                 // = q * H + h
    float M = -INFINITY;
    for (int s = 0; s < nsplit; s++) M = fmaxf(M, pM[(size_t)s * T * H + th]);
    float L = 0.0f;
    for (int s = 0; s < nsplit; s++) {
        float ms = pM[(size_t)s * T * H + th];
        if (ms != -INFINITY) L += pL[(size_t)s * T * H + th] * __expf(ms - M);
    }
    const float invL = 1.0f / L;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float o = 0.0f;
        for (int s = 0; s < nsplit; s++) {
            size_t pi = (size_t)s * T * H + th;
            float ms = pM[pi];
            if (ms != -INFINITY) o += pO[pi * D + d] * __expf(ms - M);
        }
        out[(size_t)th * D + d] = o * invL;
    }
}

// ---------------------------------------------------------------------------------------------
static float* g_buf = nullptr;
static size_t g_bytes = 0;

template <int NJ>
static void launch_partial(dim3 grid, const float* Q, const float* Kc, const float* Vc,
                           const float* Kt, const float* Vt, const int* par, float* pO, float* pM,
                           float* pL, int P, int T, int H, int D) {
    size_t smem = ((size_t)(BR + 2 * BC) * (D + 1) + BR * (BC + 1)) * sizeof(float);
    static size_t configured = 0;
    if (smem > configured) {
        cudaFuncSetAttribute(tree_attn_partial<NJ>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        configured = smem;
    }
    tree_attn_partial<NJ><<<grid, THREADS, smem>>>(Q, Kc, Vc, Kt, Vt, par, pO, pM, pL, P, T, H, D,
                                                   1.0f / sqrtf((float)D));
}

// Q, K_cache, V_cache, K_tree, V_tree, parents, output are device pointers
extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const float* K_tree, const float* V_tree, const int* parents, float* output,
                      int P, int T, int H, int D) {
    const int nsplit = (P + T + SPLIT_KEYS - 1) / SPLIT_KEYS;
    const size_t parts = (size_t)nsplit * T * H;
    size_t bytes = parts * (D + 2) * sizeof(float);
    if (bytes > g_bytes) {
        if (g_buf) cudaFree(g_buf);
        cudaMalloc(&g_buf, bytes);
        g_bytes = bytes;
    }
    float* pO = g_buf;
    float* pM = pO + parts * D;
    float* pL = pM + parts;

    dim3 grid(nsplit, (T + BR - 1) / BR, H);
    if (D <= 64)       launch_partial<8>(grid, Q, K_cache, V_cache, K_tree, V_tree, parents, pO, pM, pL, P, T, H, D);
    else if (D <= 128) launch_partial<16>(grid, Q, K_cache, V_cache, K_tree, V_tree, parents, pO, pM, pL, P, T, H, D);
    else               launch_partial<32>(grid, Q, K_cache, V_cache, K_tree, V_tree, parents, pO, pM, pL, P, T, H, D);

    int ct = D < 32 ? 32 : (D > 256 ? 256 : ((D + 31) / 32) * 32);
    tree_attn_combine<<<T * H, ct>>>(pO, pM, pL, output, nsplit, T, H, D);
}
