#include <cuda_runtime.h>
#include <math.h>

// Attention with sinks (StreamingLLM):
//   allowed(i, j) = j <= i  &&  ( j < num_sinks  ||  j >= i - window_size + 1 )
//   out_i = softmax_j( Q_i . K_j / sqrt(d) | allowed ) @ V
//
// FlashAttention-style, FP32. Each block owns BQ = 32 consecutive queries and
// streams only the key tiles that can be allowed for them: the tile holding the
// sinks (keys 0..num_sinks-1) plus the tiles covering
// [q0 - window_size + 1, q_last]. Scores never leave the SM; an online softmax
// keeps a running (max, sum, output) per query. Work per query is
// O(window + sinks), not O(M).
//
// 8 warps x 4 queries per warp. Within a key tile of BK = 32 keys:
//   S  : lane l computes the 4 queries' scores against key l (float4 smem reads).
//   PV : lane l owns output dims 4l..4l+3 for the warp's 4 queries.

#define BQ 32
#define BK 32
#define DMAX 128
#define DPAD (DMAX + 4)          // padded row -> conflict-free float4 column reads
#define NWARPS 8
#define QPW (BQ / NWARPS)        // queries per warp = 4
#define NTHREADS (NWARPS * 32)

struct Smem {
    float Q[BQ][DPAD];
    float K[BK][DPAD];
    float V[BK][DMAX];
    float P[NWARPS][QPW][BK];
};

__device__ __forceinline__ void load_rows(const float* __restrict__ src, int row0, int nrows, int M,
                                          int d, float* dst, int ldst, int tid)
{
    for (int idx = tid; idx < nrows * DMAX; idx += NTHREADS) {
        const int r = idx / DMAX, c = idx % DMAX;
        const int gr = row0 + r;
        dst[r * ldst + c] = (gr < M && c < d) ? __ldg(src + (size_t)gr * d + c) : 0.f;
    }
}

__global__ void __launch_bounds__(NTHREADS)
sink_attn_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                 const float* __restrict__ V, float* __restrict__ out,
                 int M, int d, int num_sinks, int window_size, float scale)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int q0   = blockIdx.x * BQ;
    const int qLast = min(q0 + BQ, M) - 1;

    load_rows(Q, q0, BQ, M, d, &sm.Q[0][0], DPAD, tid);

    // Per-query state (warp-uniform values, replicated across lanes).
    float m[QPW], l[QPW], o[QPW][4];
#pragma unroll
    for (int a = 0; a < QPW; ++a) {
        m[a] = -INFINITY; l[a] = 0.f;
        o[a][0] = o[a][1] = o[a][2] = o[a][3] = 0.f;
    }
    int qi[QPW];
#pragma unroll
    for (int a = 0; a < QPW; ++a) qi[a] = q0 + warp * QPW + a;

    const int dv = (d + 3) & ~3;                     // dims rounded up to float4 (padding is zero)
    const int winLo  = max(0, q0 - window_size + 1);
    const int tFirst = (winLo / BK) * BK;
    // Tile list: tile 0 (sinks) if not already covered, then tFirst..qLast.
    const int extra  = (tFirst > 0) ? 1 : 0;
    const int nTiles = extra + (qLast - tFirst) / BK + 1;

    for (int t = 0; t < nTiles; ++t) {
        const int k0 = (extra && t == 0) ? 0 : tFirst + (t - extra) * BK;

        __syncthreads();                             // previous tile fully consumed
        load_rows(K, k0, BK, M, d, &sm.K[0][0], DPAD, tid);
        load_rows(V, k0, BK, M, d, &sm.V[0][0], DMAX, tid);
        __syncthreads();

        // ---- S = Q K^T for this warp's 4 queries vs key (k0 + lane) ----
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

        // ---- mask + online softmax update ----
        const int j = k0 + lane;
#pragma unroll
        for (int a = 0; a < QPW; ++a) {
            const int i = qi[a];
            const bool ok = (i < M) && (j <= i) && (j < num_sinks || j >= i - window_size + 1);
            float x = ok ? s[a] * scale : -INFINITY;

            float tmax = x;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
            const float mNew = fmaxf(m[a], tmax);

            float p = 0.f, alpha = 1.f;
            if (mNew != -INFINITY) {                 // warp-uniform
                p = (x == -INFINITY) ? 0.f : __expf(x - mNew);
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

        // ---- O += P V (lane owns dims 4*lane .. 4*lane+3) ----
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
        __syncwarp();                                // P reused next tile
    }

    // ---- normalize and store ----
#pragma unroll
    for (int a = 0; a < QPW; ++a) {
        const int i = qi[a];
        if (i >= M) continue;
        const float inv = 1.f / l[a];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int c = lane * 4 + u;
            if (c < d) out[(size_t)i * d + c] = o[a][u] * inv;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int num_sinks, int window_size) {
    const size_t smem = sizeof(Smem);                // ~54 KB -> needs the opt-in above 48 KB
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(sink_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const int blocks = (M + BQ - 1) / BQ;
    const float scale = 1.0f / sqrtf((float)d);
    sink_attn_kernel<<<blocks, NTHREADS, smem>>>(Q, K, V, output, M, d, num_sinks, window_size, scale);
    cudaDeviceSynchronize();
}
