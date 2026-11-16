/*
INT8 KV-Cache Attention (decode)
Q: (H, D) float32;  K_int8, V_int8: (H, S, D) int8;  k_scale, v_scale: (H, S) float32
  K[h,s,:] = K_int8[h,s,:] * k_scale[h,s]   (same for V)
  output[h] = softmax_s( Q[h] . K[h,s] / sqrt(D) ) @ V[h]

Constraints
 - 1 ≤ H ≤ 64, 1 ≤ S ≤ 32,768, 8 ≤ D ≤ 256 with D % 8 == 0
 - Performance is measured with H = 32, S = 8192, D = 128
*/

/*
Approach: split-K "flash decoding". It's bandwidth-bound: at the benchmark the
caches are 64 MB of int8 and everything else is tiny.

 1. attn_partial   grid (H, splits); each block covers 256 keys with 128 threads.
                   A key row is handled by a lane group of LPK = next_pow2(D/8)
                   lanes, each owning 8 dims. So a lane loads 8 int8 of K (one 8 B
                   load), dots them with its 8 query values held in registers,
                   and the group sums with __shfl_xor. That gives every lane in
                   the group the score, which is then multiplied by k_scale. Each
                   group keeps its own online softmax (m, l) and 8 output dims,
                   updated with 8 int8 of V times v_scale. At D = 128, a warp is
                   2 groups, a block is 8 groups, and each group handles 32 keys.
                   The block then merges its groups through shared memory and
                   writes (m, l, O[D]) for its split.
 2. attn_combine   One block per head merges the splits:
                   M = max m, L = sum l * 2^(m - M), O = sum O_s * 2^(m_s - M) / L.

 - 1/sqrt(D) * log2(e) is folded into Q, so the softmax uses exp2f.
 - At the benchmark: 32 splits x 32 heads = 1024 blocks. Each warp load is 256
   contiguous bytes (2 rows), and K and V are each read exactly once.
 - The partial buffer (H * splits * (D + 2) floats) is allocated once and reused.
*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define THREADS 128
#define NWARPS  (THREADS / 32)
#define KEYS_PER_BLOCK 256
#define DMAX 256

__device__ __forceinline__ void unpack8(uint2 v, float (&f)[8])
{
    const char4 a = *reinterpret_cast<const char4*>(&v.x);
    const char4 b = *reinterpret_cast<const char4*>(&v.y);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
    f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}

// ---------------------------------------------------------------- pass 1
__global__ void __launch_bounds__(THREADS)
attn_partial(const float* __restrict__ Q, const int8_t* __restrict__ Kq,
             const int8_t* __restrict__ Vq, const float* __restrict__ ks,
             const float* __restrict__ vs, float* __restrict__ part,
             int S, int D, int LPK, float qscale, int nSplit)
{
    __shared__ float sm_m[THREADS], sm_l[THREADS];     // per group (≤ 128 groups)
    __shared__ float sm_o[NWARPS * 32 * 8];            // groups x D ≤ 1024 floats

    const int h = blockIdx.x, sp = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gpw = 32 / LPK;                          // groups per warp
    const int G = NWARPS * gpw;                        // groups per block
    const int gl = lane / LPK;                         // group within warp
    const int li = lane % LPK;                         // lane within group
    const int g = warp * gpw + gl;                     // group id in block
    const int d0 = li * 8;
    const bool act = d0 < D;                           // lane owns real dims

    float q[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) q[j] = act ? __ldg(Q + (size_t)h * D + d0 + j) * qscale : 0.f;

    float m = -INFINITY, l = 0.f, o[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = 0.f;

    const int s0 = sp * KEYS_PER_BLOCK;
    const int s1 = min(s0 + KEYS_PER_BLOCK, S);
    const size_t headRow = (size_t)h * S;

    // Every warp runs the same number of iterations, so the __shfl_xor below always
    // has all 32 lanes present. A group with no key this iteration just skips the update.
    const int iters = (s1 - s0 + G - 1) / G;
    for (int it = 0; it < iters; ++it) {
        const int s = s0 + it * G + g;
        const bool valid = s < s1;

        // ---- score
        float kv[8], dot = 0.f;
        uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
        if (act && valid) {
            kraw = __ldg(reinterpret_cast<const uint2*>(Kq + (headRow + s) * D + d0));
            vraw = __ldg(reinterpret_cast<const uint2*>(Vq + (headRow + s) * D + d0));
        }
        unpack8(kraw, kv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) dot = fmaf(q[j], kv[j], dot);
        for (int off = LPK >> 1; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        if (!valid) continue;                          // uniform within the group
        const float sc = dot * __ldg(ks + headRow + s);

        // ---- online softmax update
        const float mNew = fmaxf(m, sc);
        const float alpha = exp2f(m - mNew);           // m = -inf -> 0
        const float p = exp2f(sc - mNew);
        l = l * alpha + p;
        m = mNew;

        float vv[8];
        unpack8(vraw, vv);
        const float pv = p * __ldg(vs + headRow + s);
        #pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = fmaf(pv, vv[j], o[j] * alpha);
    }

    // ---- merge groups within the block
    if (li == 0) { sm_m[g] = m; sm_l[g] = l; }
    if (act) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) sm_o[g * D + d0 + j] = o[j];
    }
    __syncthreads();

    float M = -INFINITY;
    for (int i = 0; i < G; ++i) M = fmaxf(M, sm_m[i]);
    float* out = part + ((size_t)h * nSplit + sp) * (D + 2);
    if (threadIdx.x == 0) {
        float L = 0.f;
        for (int i = 0; i < G; ++i) if (sm_m[i] != -INFINITY) L += sm_l[i] * exp2f(sm_m[i] - M);
        out[0] = M;
        out[1] = L;
    }
    for (int d = threadIdx.x; d < D; d += THREADS) {
        float acc = 0.f;
        for (int i = 0; i < G; ++i)
            if (sm_m[i] != -INFINITY) acc = fmaf(sm_o[i * D + d], exp2f(sm_m[i] - M), acc);
        out[2 + d] = acc;
    }
}

// ---------------------------------------------------------------- pass 2
__global__ void attn_combine(const float* __restrict__ part, float* __restrict__ output,
                             int D, int nSplit)
{
    const int h = blockIdx.x, d = threadIdx.x;
    if (d >= D) return;
    const float* p = part + (size_t)h * nSplit * (D + 2);

    float M = -INFINITY;
    for (int s = 0; s < nSplit; ++s) M = fmaxf(M, p[(size_t)s * (D + 2)]);
    float L = 0.f, acc = 0.f;
    for (int s = 0; s < nSplit; ++s) {
        const float* r = p + (size_t)s * (D + 2);
        const float w = exp2f(r[0] - M);
        L = fmaf(r[1], w, L);
        acc = fmaf(r[2 + d], w, acc);
    }
    output[(size_t)h * D + d] = acc / L;
}

// ---------------------------------------------------------------- host
static float* g_part = nullptr;
static size_t g_cap = 0;

// Q, K_int8, V_int8, k_scale, v_scale, output are device pointers
extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim)
{
    const int H = num_heads, S = seq_len, D = head_dim;
    if (H <= 0 || S <= 0 || D <= 0) return;

    int LPK = 1;
    while (LPK * 8 < D) LPK <<= 1;                     // lanes per key row (pow2, ≤ 32)

    const int nSplit = (S + KEYS_PER_BLOCK - 1) / KEYS_PER_BLOCK;
    const size_t need = (size_t)H * nSplit * (D + 2);
    if (need > g_cap) {
        if (g_part) cudaFree(g_part);
        cudaMalloc(&g_part, need * sizeof(float));
        g_cap = need;
    }

    const float qscale = (1.0f / sqrtf((float)D)) * 1.4426950408889634f;   // * log2(e)
    attn_partial<<<dim3(H, nSplit), THREADS>>>(Q, K_int8, V_int8, k_scale, v_scale, g_part,
                                               S, D, LPK, qscale, nSplit);
    attn_combine<<<H, DMAX>>>(g_part, output, D, nSplit);
    cudaDeviceSynchronize();
}
