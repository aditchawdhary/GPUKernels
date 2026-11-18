#include <cuda_runtime.h>
#include <math.h>

// Fused RoPE + causal grouped-query attention (FlashAttention-style, FP32).
//
//   rope(x, p)[d]     = x[d]   * cos[p][d] - x[d+m] * sin[p][d]     (d <  m = head_dim/2)
//   rope(x, p)[d + m] = x[d+m] * cos[p][d] + x[d]   * sin[p][d]
//   out[h, i] = softmax_{j <= i}( rope(Q[h,i], i) . rope(K[g,j], j) / sqrt(head_dim) ) @ V[g]
//   g = h / (num_q_heads / num_kv_heads)
//
// RoPE is applied while tiles are loaded into shared memory, so rotated Q/K are
// never written to global memory. Each block owns BQ = 32 query rows of one
// query head and streams key tiles 0..(last query) of its KV head with an online
// softmax. 8 warps x 4 queries; in a BK = 32 key tile, lane l scores key l
// (float4 smem reads) and owns output dims 4l..4l+3 for the P.V update.
// Tiles only load ceil(head_dim/4)*4 columns, so small head_dim stays cheap.
// Blocks with the most work (last query tiles) launch first for load balance.

#define BQ 32
#define BK 32
#define DMAX 128
#define DPAD (DMAX + 4)
#define NWARPS 8
#define QPW (BQ / NWARPS)
#define NTHREADS (NWARPS * 32)

struct Smem {
    float Q[BQ][DPAD];
    float K[BK][DPAD];
    float V[BK][DMAX];
    float P[NWARPS][QPW][BK];
};

// Load rows [row0, row0 + nrows) of one head (seq_len x d), optionally rotated.
template <bool ROPE>
__device__ __forceinline__ void load_tile(const float* __restrict__ src, const float* __restrict__ cs,
                                          const float* __restrict__ sn, int row0, int nrows, int S,
                                          int d, int dv, float* dst, int ldst, int tid)
{
    const int m = d >> 1;
    for (int idx = tid; idx < nrows * dv; idx += NTHREADS) {
        const int r = idx / dv, c = idx - r * dv;
        const int gr = row0 + r;
        float v = 0.f;
        if (gr < S && c < d) {
            const float* x = src + (size_t)gr * d;
            if (ROPE) {
                const int f = (c < m) ? c : c - m;               // frequency index
                const float co = __ldg(cs + (size_t)gr * m + f);
                const float si = __ldg(sn + (size_t)gr * m + f);
                v = (c < m) ? __ldg(x + c) * co - __ldg(x + c + m) * si
                            : __ldg(x + c) * co + __ldg(x + c - m) * si;
            } else {
                v = __ldg(x + c);
            }
        }
        dst[r * ldst + c] = v;
    }
}

__global__ void __launch_bounds__(NTHREADS)
rope_attn_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                 const float* __restrict__ V, const float* __restrict__ cs,
                 const float* __restrict__ sn, float* __restrict__ out,
                 int Hq, int Hkv, int S, int d, float scale)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);

    const int tid  = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int h    = blockIdx.y;
    const int g    = h / (Hq / Hkv);
    const int qt   = gridDim.x - 1 - blockIdx.x;                 // heavy tiles first
    const int q0   = qt * BQ;
    const int qLast = min(q0 + BQ, S) - 1;
    const int dv   = (d + 3) & ~3;

    const float* Qh = Q + (size_t)h * S * d;
    const float* Kg = K + (size_t)g * S * d;
    const float* Vg = V + (size_t)g * S * d;

    load_tile<true>(Qh, cs, sn, q0, BQ, S, d, dv, &sm.Q[0][0], DPAD, tid);

    float m[QPW], l[QPW], o[QPW][4];
    int qi[QPW];
#pragma unroll
    for (int a = 0; a < QPW; ++a) {
        m[a] = -INFINITY; l[a] = 0.f;
        o[a][0] = o[a][1] = o[a][2] = o[a][3] = 0.f;
        qi[a] = q0 + warp * QPW + a;
    }

    const int nTiles = qLast / BK + 1;
    for (int t = 0; t < nTiles; ++t) {
        const int k0 = t * BK;
        __syncthreads();
        load_tile<true >(Kg, cs, sn, k0, BK, S, d, dv, &sm.K[0][0], DPAD, tid);
        load_tile<false>(Vg, cs, sn, k0, BK, S, d, dv, &sm.V[0][0], DMAX, tid);
        __syncthreads();

        // ---- scores: 4 queries x key (k0 + lane) ----
        float s[QPW] = {0.f, 0.f, 0.f, 0.f};
        for (int c = 0; c < dv; c += 4) {
            const float4 kv = *reinterpret_cast<const float4*>(&sm.K[lane][c]);
#pragma unroll
            for (int a = 0; a < QPW; ++a) {
                const float4 qv = *reinterpret_cast<const float4*>(&sm.Q[warp * QPW + a][c]);
                s[a] = fmaf(qv.x, kv.x, s[a]);
                s[a] = fmaf(qv.y, kv.y, s[a]);
                s[a] = fmaf(qv.z, kv.z, s[a]);
                s[a] = fmaf(qv.w, kv.w, s[a]);
            }
        }

        // ---- causal mask + online softmax ----
        const int j = k0 + lane;
#pragma unroll
        for (int a = 0; a < QPW; ++a) {
            const int i = qi[a];
            const bool ok = (i < S) && (j <= i);
            const float x = ok ? s[a] * scale : -INFINITY;

            float tmax = x;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
            const float mNew = fmaxf(m[a], tmax);

            float p = 0.f, alpha = 1.f;
            if (mNew != -INFINITY) {                             // warp-uniform
                p = ok ? __expf(x - mNew) : 0.f;
                alpha = (m[a] == -INFINITY) ? 0.f : __expf(m[a] - mNew);
            }
            float psum = p;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) psum += __shfl_xor_sync(0xffffffff, psum, off);

            l[a] = l[a] * alpha + psum;
            m[a] = mNew;
            o[a][0] *= alpha; o[a][1] *= alpha; o[a][2] *= alpha; o[a][3] *= alpha;
            sm.P[warp][a][lane] = p;
        }
        __syncwarp();

        // ---- O += P V ----
        if (lane * 4 < d) {
#pragma unroll 2
            for (int jj = 0; jj < BK; jj += 4) {
                float4 vr[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) vr[u] = *reinterpret_cast<const float4*>(&sm.V[jj + u][lane * 4]);
#pragma unroll
                for (int a = 0; a < QPW; ++a) {
                    const float4 pv = *reinterpret_cast<const float4*>(&sm.P[warp][a][jj]);
                    const float pp[4] = {pv.x, pv.y, pv.z, pv.w};
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        o[a][0] = fmaf(pp[u], vr[u].x, o[a][0]);
                        o[a][1] = fmaf(pp[u], vr[u].y, o[a][1]);
                        o[a][2] = fmaf(pp[u], vr[u].z, o[a][2]);
                        o[a][3] = fmaf(pp[u], vr[u].w, o[a][3]);
                    }
                }
            }
        }
        __syncwarp();
    }

    // ---- normalize and store ----
    float* Oh = out + (size_t)h * S * d;
#pragma unroll
    for (int a = 0; a < QPW; ++a) {
        const int i = qi[a];
        if (i >= S) continue;
        const float inv = 1.f / l[a];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int c = lane * 4 + u;
            if (c < d) Oh[(size_t)i * d + c] = o[a][u] * inv;
        }
    }
}

// Q, K, V, cos, sin, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* cos,
                      const float* sin, float* output, int num_q_heads, int num_kv_heads,
                      int seq_len, int head_dim) {
    const size_t smem = sizeof(Smem);                            // ~54 KB -> opt-in above 48 KB
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(rope_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    dim3 grid((seq_len + BQ - 1) / BQ, num_q_heads);
    const float scale = 1.0f / sqrtf((float)head_dim);
    rope_attn_kernel<<<grid, NTHREADS, smem>>>(Q, K, V, cos, sin, output, num_q_heads, num_kv_heads,
                                               seq_len, head_dim, scale);
    cudaDeviceSynchronize();
}
