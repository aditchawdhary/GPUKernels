#include <cuda_runtime.h>
#include <math.h>

// Causal self-attention, flash-attention style (no M x M score matrix).
//   out = softmax( Q K^T / sqrt(d) + causal_mask ) V
//
// Block = 8 warps, 32 query rows (4 per warp). Loops over 32-key tiles up to the diagonal.
//   scores : lane j owns key j, dot product over d read as float4 from shared memory.
//            Lanes walk the d-chunks in rotated order ((c + lane) & 31) -> no bank conflicts.
//   softmax: online (running max / running sum), per row, via warp shuffles.
//   P @ V  : lane owns 4 output dims (float4), p_j broadcast from shared memory.

#define DMAX 128
#define D4 (DMAX / 4)      // 32 float4 chunks per row
#define BQ 32              // query rows per block
#define BKV 32             // keys per tile (= warp size)
#define NW 8               // warps per block
#define RPW (BQ / NW)      // 4 query rows per warp
#define THREADS (NW * 32)

struct Smem {
    float4 Qs[BQ][D4];
    float4 Ks[BKV][D4];
    float4 Vs[BKV][D4];
    float  Ps[NW][RPW][BKV];
};

__device__ __forceinline__ float4 load_row4(const float* __restrict__ P, int row, int c4,
                                            int M, int d) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (row >= M) return v;
    int c = c4 * 4;
    const float* p = P + (size_t)row * d + c;
    if ((d & 3) == 0) {
        if (c < d) v = __ldg(reinterpret_cast<const float4*>(p));
    } else {
        if (c + 0 < d) v.x = __ldg(p + 0);
        if (c + 1 < d) v.y = __ldg(p + 1);
        if (c + 2 < d) v.z = __ldg(p + 2);
        if (c + 3 < d) v.w = __ldg(p + 3);
    }
    return v;
}

__global__ void __launch_bounds__(THREADS)
causal_flash_attn(const float* __restrict__ Q, const float* __restrict__ K,
                  const float* __restrict__ V, float* __restrict__ O,
                  int M, int d, float scale_log2) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int w = tid >> 5;

    // Heaviest query tiles (most keys) are scheduled first.
    const int qt = gridDim.x - 1 - blockIdx.x;
    const int q0 = qt * BQ;

    for (int idx = tid; idx < BQ * D4; idx += THREADS) {
        int r = idx / D4, c4 = idx % D4;
        sm.Qs[r][c4] = load_row4(Q, q0 + r, c4, M, d);
    }

    float4 o[RPW];
    float m[RPW], l[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        o[r] = make_float4(0.f, 0.f, 0.f, 0.f);
        m[r] = -INFINITY;
        l[r] = 0.f;
    }

    const int kEnd = min(M, q0 + BQ);
    for (int k0 = 0; k0 < kEnd; k0 += BKV) {
        __syncthreads();   // previous tile fully consumed (and Q loaded on first pass)
        for (int idx = tid; idx < BKV * D4; idx += THREADS) {
            int j = idx / D4, c4 = idx % D4;
            sm.Ks[j][c4] = load_row4(K, k0 + j, c4, M, d);
            sm.Vs[j][c4] = load_row4(V, k0 + j, c4, M, d);
        }
        __syncthreads();

        // ---- scores: lane = key j
        float s[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; ++r) s[r] = 0.f;
        #pragma unroll 8
        for (int c = 0; c < D4; ++c) {
            const int x = (c + lane) & (D4 - 1);
            const float4 kv = sm.Ks[lane][x];
            #pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const float4 qv = sm.Qs[w * RPW + r][x];
                s[r] = fmaf(qv.x, kv.x, s[r]);
                s[r] = fmaf(qv.y, kv.y, s[r]);
                s[r] = fmaf(qv.z, kv.z, s[r]);
                s[r] = fmaf(qv.w, kv.w, s[r]);
            }
        }

        // ---- online softmax per row
        const int kj = k0 + lane;
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int qi = q0 + w * RPW + r;
            const bool ok = (kj <= qi) && (kj < M);
            const float val = ok ? s[r] * scale_log2 : -INFINITY;   // log2 domain

            float mt = val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
            const float mn = fmaxf(m[r], mt);          // finite: key 0 is always visible
            const float p = ok ? exp2f(val - mn) : 0.f;
            const float corr = exp2f(m[r] - mn);       // 0 on the first tile

            float ps = p;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);

            l[r] = l[r] * corr + ps;
            m[r] = mn;
            o[r].x *= corr; o[r].y *= corr; o[r].z *= corr; o[r].w *= corr;
            sm.Ps[w][r][lane] = p;
        }
        __syncwarp();

        // ---- o += P @ V  (lane owns dims 4*lane .. 4*lane+3)
        #pragma unroll 4
        for (int j = 0; j < BKV; ++j) {
            const float4 vv = sm.Vs[j][lane];
            #pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const float pj = sm.Ps[w][r][j];
                o[r].x = fmaf(pj, vv.x, o[r].x);
                o[r].y = fmaf(pj, vv.y, o[r].y);
                o[r].z = fmaf(pj, vv.z, o[r].z);
                o[r].w = fmaf(pj, vv.w, o[r].w);
            }
        }
        __syncwarp();
    }

    // ---- normalize and write
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int qi = q0 + w * RPW + r;
        if (qi >= M) continue;
        const float inv = 1.f / l[r];
        const int c = lane * 4;
        float* out = O + (size_t)qi * d + c;
        if (c + 0 < d) out[0] = o[r].x * inv;
        if (c + 1 < d) out[1] = o[r].y * inv;
        if (c + 2 < d) out[2] = o[r].z * inv;
        if (c + 3 < d) out[3] = o[r].w * inv;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    const size_t smem = sizeof(Smem);   // 53,248 bytes -> needs the opt-in above 48 KB
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(causal_flash_attn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        configured = true;
    }
    const float scale_log2 = (1.0f / sqrtf((float)d)) * 1.4426950408889634f;   // log2(e)
    const int blocks = (M + BQ - 1) / BQ;
    causal_flash_attn<<<blocks, THREADS, smem>>>(Q, K, V, output, M, d, scale_log2);
    cudaDeviceSynchronize();
}
