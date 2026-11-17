/*
Flash Decoding Attention
Q: (T, H, D), K, V: (S, H, D), output: (T, H, D), float32.
  output[t, h] = softmax_s( Q[t,h] . K[s,h] / sqrt(D) ) @ V[:, h]

Constraints
 - 1 ≤ T ≤ 16, 1 ≤ S ≤ 131,072, 1 ≤ H ≤ 64, 1 ≤ D ≤ 256
 - Performance is measured with T = 1, S = 65,536, H = 32, D = 128
*/

/*
Approach: split-KV ("flash decoding") in two kernels. With T = 1 this is purely
bandwidth-bound: K and V are 2 x 1 GiB at the benchmark and everything else is
tiny, so the goal is to read each K/V byte exactly once with wide, coalesced
loads and enough bytes in flight.

 1. fd_partial   grid (H, nSplit, ceil(T/TQ)), 256 threads = 8 warps.
                 A block owns `chunk` consecutive keys of one head. One warp
                 handles one key row at a time: each lane holds EPL of the D
                 dims (float4 loads when D % 4 == 0, so a warp reads a 512 B
                 row at D = 128 in one instruction). The warp loads U keys' K
                 and V rows up front, computes the U scores with a
                 __shfl_xor reduction, then does one online-softmax update for
                 the group (one rescale per U keys instead of per key).
                 The 8 warps' (m, l, acc) are merged through shared memory and
                 the block writes one unnormalised partial (m, l, O[D]) for
                 each of its queries.
 2. fd_combine   grid (H, T): one block merges the splits of a (t, h) pair:
                 M = max m_s,  c_s = 2^(m_s - M),  out = sum c_s O_s / sum c_s l_s.

 - 1/sqrt(D) * log2(e) is folded into Q, so the softmax uses exp2f; it is a
   max-subtracted online softmax, so it is stable for large scores.
 - Queries are processed TQ (1, 2 or 4) per block so the per-query state stays
   in registers; for T > 4 the extra z-blocks re-read K/V (T > 1 isn't timed).
 - At the benchmark: chunk = 512 keys, 128 splits x 32 heads = 4096 blocks; the
   partial buffer is 2 MB, allocated once and reused across calls.
*/

#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256
#define NWARPS (THREADS / 32)
#define MAX_SPLITS 256
#define KEYS_PER_SPLIT 512

// ------------------------------------------------------------------ helpers
template <bool VEC>
__device__ __forceinline__ int dim_of(int e, int lane)
{
    // VEC: lane owns dims [g*128 + 4*lane, +4) for g = e/4;  scalar: e*32 + lane
    return VEC ? ((e >> 2) * 128 + lane * 4 + (e & 3)) : (e * 32 + lane);
}

template <bool VEC, int EPL>
__device__ __forceinline__ void load_row(const float* __restrict__ row, int lane, int D,
                                         float (&x)[EPL])
{
    if (VEC) {
#pragma unroll
        for (int g = 0; g < EPL / 4; ++g) {
            const int d = g * 128 + lane * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (d < D) v = __ldg(reinterpret_cast<const float4*>(row + d));
            x[g * 4 + 0] = v.x; x[g * 4 + 1] = v.y;
            x[g * 4 + 2] = v.z; x[g * 4 + 3] = v.w;
        }
    } else {
#pragma unroll
        for (int e = 0; e < EPL; ++e) {
            const int d = e * 32 + lane;
            x[e] = (d < D) ? __ldg(row + d) : 0.f;
        }
    }
}

__device__ __forceinline__ float warp_sum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------ pass 1
template <int TQ, bool VEC, int EPL, int U>
__global__ void __launch_bounds__(THREADS)
fd_partial(const float* __restrict__ Q, const float* __restrict__ K,
           const float* __restrict__ V, float* __restrict__ part_o,
           float* __restrict__ part_ml, int T, int S, int H, int D, int chunk,
           int nSplit, float qscale)
{
    extern __shared__ float smem[];
    float* s_acc = smem;                            // [NWARPS][TQ][D]
    float* s_m   = smem + NWARPS * TQ * D;          // [NWARPS][TQ]
    float* s_l   = s_m + NWARPS * TQ;               // [NWARPS][TQ]

    const int h = blockIdx.x, split = blockIdx.y, t0 = blockIdx.z * TQ;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int s_begin = split * chunk;
    const int s_end = min(S, s_begin + chunk);
    const size_t rs = (size_t)H * D;                // stride between key rows

    // scaled queries in registers
    float q[TQ][EPL];
#pragma unroll
    for (int t = 0; t < TQ; ++t) {
        if (t0 + t < T) {
            load_row<VEC, EPL>(Q + ((size_t)(t0 + t) * H + h) * D, lane, D, q[t]);
#pragma unroll
            for (int e = 0; e < EPL; ++e) q[t][e] *= qscale;
        } else {
#pragma unroll
            for (int e = 0; e < EPL; ++e) q[t][e] = 0.f;
        }
    }

    float m[TQ], l[TQ], acc[TQ][EPL];
#pragma unroll
    for (int t = 0; t < TQ; ++t) {
        m[t] = -INFINITY; l[t] = 0.f;
#pragma unroll
        for (int e = 0; e < EPL; ++e) acc[t][e] = 0.f;
    }

    const float* Kh = K + (size_t)h * D;
    const float* Vh = V + (size_t)h * D;

    for (int base = s_begin + warp * U; base < s_end; base += NWARPS * U) {
        float kx[U][EPL], vx[U][EPL];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int s = base + u;
            if (s < s_end) {
                load_row<VEC, EPL>(Kh + (size_t)s * rs, lane, D, kx[u]);
                load_row<VEC, EPL>(Vh + (size_t)s * rs, lane, D, vx[u]);
            } else {
#pragma unroll
                for (int e = 0; e < EPL; ++e) { kx[u][e] = 0.f; vx[u][e] = 0.f; }
            }
        }

#pragma unroll
        for (int t = 0; t < TQ; ++t) {
            float sc[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                float d = 0.f;
#pragma unroll
                for (int e = 0; e < EPL; ++e) d = fmaf(q[t][e], kx[u][e], d);
                d = warp_sum(d);
                sc[u] = (base + u < s_end) ? d : -INFINITY;
            }
            // base < s_end, so sc[0] is finite and mn is finite
            float mn = m[t];
#pragma unroll
            for (int u = 0; u < U; ++u) mn = fmaxf(mn, sc[u]);
            const float corr = exp2f(m[t] - mn);    // m = -inf -> 0
            l[t] *= corr;
#pragma unroll
            for (int e = 0; e < EPL; ++e) acc[t][e] *= corr;
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const float p = exp2f(sc[u] - mn);  // invalid key -> 0
                l[t] += p;
#pragma unroll
                for (int e = 0; e < EPL; ++e) acc[t][e] = fmaf(p, vx[u][e], acc[t][e]);
            }
            m[t] = mn;
        }
    }

    // ---- merge the 8 warps of this block
#pragma unroll
    for (int t = 0; t < TQ; ++t) {
        if (lane == 0) { s_m[warp * TQ + t] = m[t]; s_l[warp * TQ + t] = l[t]; }
#pragma unroll
        for (int e = 0; e < EPL; ++e) {
            const int d = dim_of<VEC>(e, lane);
            if (d < D) s_acc[(warp * TQ + t) * D + d] = acc[t][e];
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < TQ * D; idx += THREADS) {
        const int t = idx / D, d = idx - t * D;
        if (t0 + t >= T) continue;
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) M = fmaxf(M, s_m[w * TQ + t]);
        float L = 0.f, o = 0.f;
        if (M != -INFINITY) {
#pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                const float mw = s_m[w * TQ + t];
                const float c = (mw == -INFINITY) ? 0.f : exp2f(mw - M);
                L = fmaf(s_l[w * TQ + t], c, L);
                o = fmaf(s_acc[(w * TQ + t) * D + d], c, o);
            }
        }
        const size_t pi = ((size_t)(t0 + t) * H + h) * nSplit + split;
        part_o[pi * D + d] = o;
        if (d == 0) { part_ml[pi * 2] = M; part_ml[pi * 2 + 1] = L; }
    }
}

// ------------------------------------------------------------------ pass 2
__global__ void __launch_bounds__(THREADS)
fd_combine(const float* __restrict__ part_o, const float* __restrict__ part_ml,
           float* __restrict__ out, int H, int D, int nSplit)
{
    __shared__ float w[MAX_SPLITS];
    __shared__ float red[NWARPS];
    __shared__ float bval;

    const int h = blockIdx.x, t = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t base = ((size_t)t * H + h) * nSplit;

    const float mv = (tid < nSplit) ? part_ml[(base + tid) * 2] : -INFINITY;
    float v = mv;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (tid == 0) {
        float M = red[0];
        for (int i = 1; i < NWARPS; ++i) M = fmaxf(M, red[i]);
        bval = M;
    }
    __syncthreads();
    const float M = bval;

    // O_s is unnormalised (sum p v), so its weight is 2^(m_s - M) / L and
    // L = sum_s l_s 2^(m_s - M)
    const float c = (tid < nSplit && mv != -INFINITY) ? exp2f(mv - M) : 0.f;
    const float lc = (tid < nSplit) ? part_ml[(base + tid) * 2 + 1] * c : 0.f;
    v = warp_sum(lc);
    __syncthreads();                                 // everyone has read bval
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (tid == 0) {
        float L = 0.f;
        for (int i = 0; i < NWARPS; ++i) L += red[i];
        bval = L;
    }
    __syncthreads();
    const float inv = (bval > 0.f) ? 1.f / bval : 0.f;
    if (tid < nSplit) w[tid] = c * inv;
    __syncthreads();

    for (int d = tid; d < D; d += THREADS) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        const float* p = part_o + base * D + d;
        int s = 0;
        for (; s + 3 < nSplit; s += 4) {
            a0 = fmaf(w[s + 0], p[(size_t)(s + 0) * D], a0);
            a1 = fmaf(w[s + 1], p[(size_t)(s + 1) * D], a1);
            a2 = fmaf(w[s + 2], p[(size_t)(s + 2) * D], a2);
            a3 = fmaf(w[s + 3], p[(size_t)(s + 3) * D], a3);
        }
        for (; s < nSplit; ++s) a0 = fmaf(w[s], p[(size_t)s * D], a0);
        out[((size_t)t * H + h) * D + d] = (a0 + a1) + (a2 + a3);
    }
}

// ------------------------------------------------------------------ host
template <int TQ, bool VEC, int EPL>
static void launch_partial(const float* Q, const float* K, const float* V, float* po,
                           float* pml, int T, int S, int H, int D, int chunk,
                           int nSplit, float qscale)
{
    constexpr int U = (TQ * EPL >= 32) ? 2 : 4;     // keep register use in check
    dim3 grid(H, nSplit, (T + TQ - 1) / TQ);
    const size_t sm = (size_t)NWARPS * TQ * (D + 2) * sizeof(float);
    fd_partial<TQ, VEC, EPL, U><<<grid, THREADS, sm>>>(Q, K, V, po, pml, T, S, H, D,
                                                       chunk, nSplit, qscale);
}

template <int TQ>
static void dispatch_dims(const float* Q, const float* K, const float* V, float* po,
                          float* pml, int T, int S, int H, int D, int chunk,
                          int nSplit, float qscale)
{
    const bool vec = (D % 4) == 0;
    if (vec) {
        if (D <= 128) launch_partial<TQ, true, 4>(Q, K, V, po, pml, T, S, H, D, chunk, nSplit, qscale);
        else          launch_partial<TQ, true, 8>(Q, K, V, po, pml, T, S, H, D, chunk, nSplit, qscale);
    } else {
        if (D <= 128) launch_partial<TQ, false, 4>(Q, K, V, po, pml, T, S, H, D, chunk, nSplit, qscale);
        else          launch_partial<TQ, false, 8>(Q, K, V, po, pml, T, S, H, D, chunk, nSplit, qscale);
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int T, int S,
                      int H, int D)
{
    if (T <= 0 || S <= 0 || H <= 0 || D <= 0) return;

    static int num_sms = 0;
    if (num_sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
        if (num_sms <= 0) num_sms = 80;
    }

    const int TQ = (T == 1) ? 1 : (T == 2) ? 2 : 4;
    const int zg = (T + TQ - 1) / TQ;
    const int keys_step = NWARPS * 4;               // keep chunks a multiple of 32 keys

    // pick the number of splits: ~512 keys each, but enough blocks to fill the GPU
    int nSplit = (S + KEYS_PER_SPLIT - 1) / KEYS_PER_SPLIT;
    const int want = 4 * num_sms;
    if ((long long)H * zg * nSplit < want) {
        const int need = (want + H * zg - 1) / (H * zg);
        const int cap = (S + keys_step - 1) / keys_step;
        nSplit = need < cap ? need : cap;
    }
    if (nSplit > MAX_SPLITS) nSplit = MAX_SPLITS;
    if (nSplit < 1) nSplit = 1;
    int chunk = (S + nSplit - 1) / nSplit;
    chunk = (chunk + keys_step - 1) / keys_step * keys_step;
    nSplit = (S + chunk - 1) / chunk;               // every split is non-empty

    // partial buffer, grown on demand and reused
    static float* part = nullptr;
    static size_t part_cap = 0;
    const size_t n_pairs = (size_t)T * H * nSplit;
    const size_t need_floats = n_pairs * (size_t)D + n_pairs * 2;
    if (need_floats > part_cap) {
        if (part) cudaFree(part);
        cudaMalloc(&part, need_floats * sizeof(float));
        part_cap = need_floats;
    }
    float* po = part;
    float* pml = part + n_pairs * (size_t)D;

    const float qscale = 1.4426950408889634f / sqrtf((float)D);   // log2(e)/sqrt(D)

    switch (TQ) {
        case 1:  dispatch_dims<1>(Q, K, V, po, pml, T, S, H, D, chunk, nSplit, qscale); break;
        case 2:  dispatch_dims<2>(Q, K, V, po, pml, T, S, H, D, chunk, nSplit, qscale); break;
        default: dispatch_dims<4>(Q, K, V, po, pml, T, S, H, D, chunk, nSplit, qscale); break;
    }

    fd_combine<<<dim3(H, T), THREADS>>>(po, pml, output, H, D, nSplit);
    cudaDeviceSynchronize();
}
