#include <cuda_runtime.h>
#include <math.h>

// Causal differential attention (Differential Transformer), FP32.
// Per head h, with D = head_dim and DV = 2 * D:
//   Q = [Q1 | Q2], K = [K1 | K2]   (each half D wide), V is DV wide
//   A1 = softmax_causal(Q1 K1^T / sqrt(D)),  A2 = softmax_causal(Q2 K2^T / sqrt(D))
//   O  = (A1 - lambda_h * A2) V
//   out = O / sqrt(mean(O^2) + 1e-5) * (1 - lambda_init)  (all-zero O -> zero row)
//
// FlashAttention-style: one block = BQ = 32 queries of one head. Key tiles of
// BK = 32 stream through smem; each map keeps its own online-softmax state and
// its own output accumulator (O1 = sum p1 V, O2 = sum p2 V), sharing the V tile.
// The difference, RMS norm and scaling run in the epilogue, so the two attention
// maps never leave the SM.
//  - 8 warps x 4 queries. Scores: lane l handles key l (float4 smem reads).
//    P.V: lane l owns dims {4l..4l+3} (+128 for the second group when DV > 128).
//  - Q/K halves are stored separately, each padded to a multiple of 4 floats,
//    so any head_dim works with float4 reads. Padding is zero-filled.
//  - Shared memory is sized at runtime from head_dim (~58 KB at head_dim = 64).
//  - Heavy (late) query tiles launch first for causal load balance.

#define BQ 32
#define BK 32
#define NWARPS 8
#define QPW (BQ / NWARPS)
#define NTHREADS (NWARPS * 32)
#define RMS_EPS 1e-5f               // RMSNorm epsilon (Differential Transformer default)

// Load rows [row0, row0+nrows) of head h into smem.
// SPLIT: two halves of width D -> [0, D4) and [D4, 2*D4); otherwise contiguous DV.
template <bool SPLIT>
__device__ __forceinline__ void load_tile(const float* __restrict__ src, int row0, int nrows, int S,
                                          int rowStride, int D, int D4, int width, float* dst,
                                          int pitch, int tid)
{
    for (int idx = tid; idx < nrows * width; idx += NTHREADS) {
        const int r = idx / width, c = idx - r * width;
        const int gr = row0 + r;
        int gc;                                   // source column, or -1 for padding
        if (SPLIT) gc = (c < D4) ? (c < D ? c : -1) : ((c - D4) < D ? D + (c - D4) : -1);
        else       gc = (c < 2 * D) ? c : -1;
        dst[r * pitch + c] = (gr < S && gc >= 0) ? __ldg(src + (size_t)gr * rowStride + gc) : 0.f;
    }
}

template <int NG>   // number of 128-wide output dim groups per lane (1: DV <= 128, 2: DV <= 256)
__global__ void __launch_bounds__(NTHREADS)
diff_attn_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                 const float* __restrict__ V, const float* __restrict__ lambdas,
                 float* __restrict__ out, int S, int H, int D, float scale, float outScale)
{
    extern __shared__ __align__(16) float smem[];
    const int D4  = (D + 3) & ~3;
    const int DV  = 2 * D;
    const int DV4 = (DV + 3) & ~3;
    const int pQK = 2 * D4 + 4;                    // pitch for split Q/K rows
    const int pV  = DV4;                           // pitch for V rows (float4-aligned)
    float* sQ  = smem;
    float* sK  = sQ + BQ * pQK;
    float* sV  = sK + BK * pQK;
    float* sP1 = sV + BK * pV;                     // [NWARPS][QPW][BK]
    float* sP2 = sP1 + NWARPS * QPW * BK;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int h  = blockIdx.y;
    const int q0 = (gridDim.x - 1 - blockIdx.x) * BQ;
    const int qLast = min(q0 + BQ, S) - 1;
    const int rowStride = H * DV;
    const float* Qh = Q + (size_t)h * DV;
    const float* Kh = K + (size_t)h * DV;
    const float* Vh = V + (size_t)h * DV;
    const float lam = lambdas[h];

    load_tile<true>(Qh, q0, BQ, S, rowStride, D, D4, 2 * D4, sQ, pQK, tid);

    float m1[QPW], l1[QPW], m2[QPW], l2[QPW];
    float o1[QPW][NG][4], o2[QPW][NG][4];
#pragma unroll
    for (int a = 0; a < QPW; ++a) {
        m1[a] = m2[a] = -INFINITY; l1[a] = l2[a] = 0.f;
#pragma unroll
        for (int gI = 0; gI < NG; ++gI)
#pragma unroll
            for (int u = 0; u < 4; ++u) o1[a][gI][u] = o2[a][gI][u] = 0.f;
    }

    const int nTiles = qLast / BK + 1;
    for (int t = 0; t < nTiles; ++t) {
        const int k0 = t * BK;
        __syncthreads();
        load_tile<true >(Kh, k0, BK, S, rowStride, D, D4, 2 * D4, sK, pQK, tid);
        load_tile<false>(Vh, k0, BK, S, rowStride, D, D4, DV4, sV, pV, tid);
        __syncthreads();

        // ---- scores for both maps: 4 queries x key (k0 + lane) ----
        float s1[QPW] = {0.f, 0.f, 0.f, 0.f}, s2[QPW] = {0.f, 0.f, 0.f, 0.f};
        const float* kr = sK + lane * pQK;
        for (int c = 0; c < D4; c += 4) {
            const float4 ka = *reinterpret_cast<const float4*>(kr + c);
            const float4 kb = *reinterpret_cast<const float4*>(kr + D4 + c);
#pragma unroll
            for (int a = 0; a < QPW; ++a) {
                const float* qr = sQ + (warp * QPW + a) * pQK;
                const float4 qa = *reinterpret_cast<const float4*>(qr + c);
                const float4 qb = *reinterpret_cast<const float4*>(qr + D4 + c);
                s1[a] = fmaf(qa.x, ka.x, fmaf(qa.y, ka.y, fmaf(qa.z, ka.z, fmaf(qa.w, ka.w, s1[a]))));
                s2[a] = fmaf(qb.x, kb.x, fmaf(qb.y, kb.y, fmaf(qb.z, kb.z, fmaf(qb.w, kb.w, s2[a]))));
            }
        }

        // ---- causal mask + two online softmax updates ----
        const int j = k0 + lane;
#pragma unroll
        for (int a = 0; a < QPW; ++a) {
            const int i = q0 + warp * QPW + a;
            const bool ok = (i < S) && (j <= i);
#pragma unroll
            for (int map = 0; map < 2; ++map) {
                float& m = map ? m2[a] : m1[a];
                float& l = map ? l2[a] : l1[a];
                const float x = ok ? (map ? s2[a] : s1[a]) * scale : -INFINITY;
                float tmax = x;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
                const float mNew = fmaxf(m, tmax);
                float p = 0.f, alpha = 1.f;
                if (mNew != -INFINITY) {                         // warp-uniform
                    p = ok ? expf(x - mNew) : 0.f;
                    alpha = (m == -INFINITY) ? 0.f : expf(m - mNew);
                }
                float ps = p;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) ps += __shfl_xor_sync(0xffffffff, ps, off);
                l = l * alpha + ps;
                m = mNew;
#pragma unroll
                for (int gI = 0; gI < NG; ++gI)
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        if (map) o2[a][gI][u] *= alpha; else o1[a][gI][u] *= alpha;
                    }
                (map ? sP2 : sP1)[(warp * QPW + a) * BK + lane] = p;
            }
        }
        __syncwarp();

        // ---- O1 += P1 V, O2 += P2 V ----
#pragma unroll
        for (int gI = 0; gI < NG; ++gI) {
            const int c = gI * 128 + lane * 4;
            if (c >= DV) continue;
            for (int jj = 0; jj < BK; jj += 4) {
                float4 vr[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) vr[u] = *reinterpret_cast<const float4*>(sV + (jj + u) * pV + c);
#pragma unroll
                for (int a = 0; a < QPW; ++a) {
                    const float4 pa = *reinterpret_cast<const float4*>(sP1 + (warp * QPW + a) * BK + jj);
                    const float4 pb = *reinterpret_cast<const float4*>(sP2 + (warp * QPW + a) * BK + jj);
                    const float p1[4] = {pa.x, pa.y, pa.z, pa.w};
                    const float p2[4] = {pb.x, pb.y, pb.z, pb.w};
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        o1[a][gI][0] = fmaf(p1[u], vr[u].x, o1[a][gI][0]);
                        o1[a][gI][1] = fmaf(p1[u], vr[u].y, o1[a][gI][1]);
                        o1[a][gI][2] = fmaf(p1[u], vr[u].z, o1[a][gI][2]);
                        o1[a][gI][3] = fmaf(p1[u], vr[u].w, o1[a][gI][3]);
                        o2[a][gI][0] = fmaf(p2[u], vr[u].x, o2[a][gI][0]);
                        o2[a][gI][1] = fmaf(p2[u], vr[u].y, o2[a][gI][1]);
                        o2[a][gI][2] = fmaf(p2[u], vr[u].z, o2[a][gI][2]);
                        o2[a][gI][3] = fmaf(p2[u], vr[u].w, o2[a][gI][3]);
                    }
                }
            }
        }
        __syncwarp();
    }

    // ---- epilogue: O = O1/l1 - lambda * O2/l2, RMS-normalize over DV, scale ----
#pragma unroll
    for (int a = 0; a < QPW; ++a) {
        const int i = q0 + warp * QPW + a;
        const float i1 = 1.f / l1[a], i2 = lam / l2[a];   // l1, l2 > 0 for i < S (key i is allowed)
        float o[NG][4];
        float ss = 0.f;
#pragma unroll
        for (int gI = 0; gI < NG; ++gI)
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int c = gI * 128 + lane * 4 + u;
                const float v = (c < DV) ? o1[a][gI][u] * i1 - o2[a][gI][u] * i2 : 0.f;
                o[gI][u] = v;
                ss = fmaf(v, v, ss);
            }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
        if (i >= S) continue;
        const float f = outScale * rsqrtf(ss / (float)DV + RMS_EPS);   // all-zero O -> 0 * f = 0
        float* dst = out + (size_t)i * rowStride + (size_t)h * DV;
#pragma unroll
        for (int gI = 0; gI < NG; ++gI)
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int c = gI * 128 + lane * 4 + u;
                if (c < DV) dst[c] = o[gI][u] * f;
            }
    }
}

template <int NG>
static void launch(const float* Q, const float* K, const float* V, const float* lambdas, float* out,
                   int S, int H, int D, float lambda_init)
{
    const int D4 = (D + 3) & ~3, DV4 = (2 * D + 3) & ~3;
    const size_t smem = sizeof(float) * ((size_t)BQ * (2 * D4 + 4) + (size_t)BK * (2 * D4 + 4) +
                                         (size_t)BK * DV4 + 2 * NWARPS * QPW * BK);
    static int configured = 0;
    if ((size_t)configured < smem) {
        cudaFuncSetAttribute(diff_attn_kernel<NG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = (int)smem;
    }
    dim3 grid((S + BQ - 1) / BQ, H);
    diff_attn_kernel<NG><<<grid, NTHREADS, smem>>>(Q, K, V, lambdas, out, S, H, D,
                                                   1.0f / sqrtf((float)D), 1.0f - lambda_init);
}

// Q, K, V, lambdas, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* lambdas,
                      float* output, int seq_len, int num_heads, int head_dim, float lambda_init) {
    if (2 * head_dim <= 128) launch<1>(Q, K, V, lambdas, output, seq_len, num_heads, head_dim, lambda_init);
    else                     launch<2>(Q, K, V, lambdas, output, seq_len, num_heads, head_dim, lambda_init);
    cudaDeviceSynchronize();
}
