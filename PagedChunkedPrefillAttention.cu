#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <algorithm>

// Paged chunked-prefill attention with GQA (fp32 flash-attention style, online softmax).
//
// Work item = (sequence s, kv head g, tile of BR "rows"). A row is a (query token, q head) pair
// for one of the `group` q heads that share kv head g; rows are ordered token-major, so every
// K/V tile gathered from the paged cache is reused by the whole GQA group and by BR/group tokens.
//
// Per item: loop over key tiles of BC keys up to the causal limit of the tile's last token:
//   S = (Q*scale) K^T  ->  online softmax (running max m, sum l)  ->  O = O*alpha + P V
// Query token i of sequence s sits at absolute position ctx - q_len + i and sees keys [0, pos].
//
// The host reads cu_seqlens_q / context_lens (a few hundred bytes) to build a compact work list,
// longest items first, so no blocks are wasted and the long prefill tiles start early.

#define BR 32                 // rows per block
#define BC 32                 // keys per tile
#define THREADS 256           // 8 threads per row
#define TPR 8                 // threads per row
#define KPT (BC / TPR)        // keys per thread in the score phase

struct WorkItem { int s, g, row0, pad; };

template <int NJ>   // NJ = ceil(head_dim / 8): output columns per thread
__global__ void __launch_bounds__(THREADS)
paged_attn_kernel(const float* __restrict__ Q, const float* __restrict__ Kc,
                  const float* __restrict__ Vc, const int* __restrict__ block_table,
                  const int* __restrict__ context_lens, const int* __restrict__ cu_seqlens_q,
                  const WorkItem* __restrict__ items, float* __restrict__ out,
                  int num_q_heads, int num_kv_heads, int D, int block_size,
                  int max_blocks_per_seq, float scale) {
    extern __shared__ __align__(16) float smem[];
    const int LD = D + 2;                      // even stride keeps float2 alignment
    float* Qs = smem;                          // [BR][LD]
    float* Ks = Qs + BR * LD;                  // [BC][LD]
    float* Vs = Ks + BC * LD;                  // [BC][LD]
    float* Ps = Vs + BC * LD;                  // [BR][BC+1]
    __shared__ int phys_slot[BC];              // physical row index (block*bs+slot) or -1

    const WorkItem it = items[blockIdx.x];
    const int s = it.s, g = it.g, row0 = it.row0;
    const int group = num_q_heads / num_kv_heads;
    const int q0 = cu_seqlens_q[s];
    const int q_len = cu_seqlens_q[s + 1] - q0;
    const int ctx = context_lens[s];
    const int R = q_len * group;
    const int pos0 = ctx - q_len;              // absolute position of query token 0

    const int tid = threadIdx.x;
    const int r = tid / TPR;                   // this thread's row in the tile
    const int lane8 = tid % TPR;

    // ---- load Q tile (pre-scaled) ----
    for (int idx = tid; idx < BR * D; idx += THREADS) {
        int rr = idx / D, d = idx % D;
        int ri = row0 + rr;
        float v = 0.0f;
        if (ri < R) {
            int qi = ri / group, h = g * group + ri % group;
            v = Q[((size_t)(q0 + qi) * num_q_heads + h) * D + d] * scale;
        }
        Qs[rr * LD + d] = v;
    }

    const int my_ri = row0 + r;
    const bool row_valid = my_ri < R;
    const int my_kmax = row_valid ? pos0 + my_ri / group : -1;   // last visible key (inclusive)
    const int last_ri = min(R, row0 + BR) - 1;
    const int kend = pos0 + last_ri / group + 1;                  // keys needed by this tile

    float m = -INFINITY, l = 0.0f;
    float acc[NJ];
#pragma unroll
    for (int j = 0; j < NJ; j++) acc[j] = 0.0f;

    const int* bt = block_table + (size_t)s * max_blocks_per_seq;
    const size_t row_stride = (size_t)num_kv_heads * D;          // one cache slot

    for (int k0 = 0; k0 < kend; k0 += BC) {
        __syncthreads();                       // previous tile fully consumed
        if (tid < BC) {
            int k = k0 + tid;
            phys_slot[tid] = (k < kend) ? bt[k / block_size] * block_size + (k & (block_size - 1)) : -1;
        }
        __syncthreads();
        // ---- gather K/V tile from the paged cache (coalesced along head_dim) ----
        for (int idx = tid; idx < BC * D; idx += THREADS) {
            int c = idx / D, d = idx % D;
            int ps = phys_slot[c];
            float kv = 0.0f, vv = 0.0f;
            if (ps >= 0) {
                size_t off = (size_t)ps * row_stride + (size_t)g * D + d;
                kv = __ldg(&Kc[off]);
                vv = __ldg(&Vc[off]);
            }
            Ks[c * LD + d] = kv;
            Vs[c * LD + d] = vv;
        }
        __syncthreads();

        // ---- scores: thread handles row r, keys lane8 + 8*j ----
        float sc[KPT];
#pragma unroll
        for (int j = 0; j < KPT; j++) sc[j] = 0.0f;
        const float* qrow = Qs + r * LD;
        for (int d = 0; d < D; d += 2) {
            float2 q2 = *reinterpret_cast<const float2*>(qrow + d);
#pragma unroll
            for (int j = 0; j < KPT; j++) {
                float2 k2 = *reinterpret_cast<const float2*>(Ks + (lane8 + TPR * j) * LD + d);
                sc[j] = fmaf(q2.x, k2.x, sc[j]);
                sc[j] = fmaf(q2.y, k2.y, sc[j]);
            }
        }

        // ---- causal mask + online softmax (8 lanes of a row reduce via shuffles) ----
        float tmax = -INFINITY;
#pragma unroll
        for (int j = 0; j < KPT; j++) {
            int k = k0 + lane8 + TPR * j;
            if (k > my_kmax) sc[j] = -INFINITY;
            tmax = fmaxf(tmax, sc[j]);
        }
#pragma unroll
        for (int o = TPR / 2; o > 0; o >>= 1)
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));

        float m_new = fmaxf(m, tmax);
        float m_use = (m_new == -INFINITY) ? 0.0f : m_new;      // fully-masked guard
        float alpha = __expf(m - m_use);                          // exp(-inf) = 0 on first tile
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
        __syncwarp();   // Ps row r is written and read only by the 8 lanes of the same warp

        // ---- O = O*alpha + P V : thread owns row r, columns lane8 + 8*j ----
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

    // ---- write output ----
    if (row_valid) {
        int qi = my_ri / group, h = g * group + my_ri % group;
        float inv = 1.0f / l;
        float* orow = out + ((size_t)(q0 + qi) * num_q_heads + h) * D;
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            int col = lane8 + TPR * j;
            if (col < D) orow[col] = acc[j] * inv;
        }
    }
}

// ---------------------------------------------------------------------------------------------
static WorkItem* g_items = nullptr;
static size_t g_items_cap = 0;

template <int NJ>
static void launch(const float* Q, const float* K, const float* V, const int* bt, const int* cl,
                   const int* cu, const WorkItem* items, int n_items, float* out, int nq, int nkv,
                   int D, int bs, int maxb) {
    size_t smem = ((size_t)(BR + 2 * BC) * (D + 2) + BR * (BC + 1)) * sizeof(float);
    static size_t configured = 0;
    if (smem > configured) {
        cudaFuncSetAttribute(paged_attn_kernel<NJ>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        configured = smem;
    }
    paged_attn_kernel<NJ><<<n_items, THREADS, smem>>>(Q, K, V, bt, cl, cu, items, out, nq, nkv, D,
                                                      bs, maxb, 1.0f / sqrtf((float)D));
}

// Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, output are device pointers
extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, const int* cu_seqlens_q,
                      float* output, int num_seqs, int total_q, int num_q_heads, int num_kv_heads,
                      int head_dim, int block_size, int max_blocks_per_seq) {
    std::vector<int> cu(num_seqs + 1), ctx(num_seqs);
    cudaMemcpy(cu.data(), cu_seqlens_q, (num_seqs + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(ctx.data(), context_lens, num_seqs * sizeof(int), cudaMemcpyDeviceToHost);

    const int group = num_q_heads / num_kv_heads;
    struct Tmp { WorkItem w; long long cost; };
    std::vector<Tmp> work;
    for (int s = 0; s < num_seqs; s++) {
        int q_len = cu[s + 1] - cu[s];
        int R = q_len * group;
        for (int row0 = 0; row0 < R; row0 += BR) {
            int last = std::min(R, row0 + BR) - 1;
            long long keys = ctx[s] - q_len + last / group + 1;
            for (int g = 0; g < num_kv_heads; g++) work.push_back({{s, g, row0, 0}, keys});
        }
    }
    std::stable_sort(work.begin(), work.end(),
                     [](const Tmp& a, const Tmp& b) { return a.cost > b.cost; });
    std::vector<WorkItem> items(work.size());
    for (size_t i = 0; i < work.size(); i++) items[i] = work[i].w;
    int n = (int)items.size();
    if (n == 0) return;

    if ((size_t)n > g_items_cap) {
        if (g_items) cudaFree(g_items);
        cudaMalloc(&g_items, n * sizeof(WorkItem));
        g_items_cap = n;
    }
    cudaMemcpy(g_items, items.data(), n * sizeof(WorkItem), cudaMemcpyHostToDevice);

    int D = head_dim;
    if (D <= 64)
        launch<8>(Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, g_items, n, output,
                  num_q_heads, num_kv_heads, D, block_size, max_blocks_per_seq);
    else if (D <= 128)
        launch<16>(Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, g_items, n, output,
                   num_q_heads, num_kv_heads, D, block_size, max_blocks_per_seq);
    else
        launch<32>(Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, g_items, n, output,
                   num_q_heads, num_kv_heads, D, block_size, max_blocks_per_seq);
}
