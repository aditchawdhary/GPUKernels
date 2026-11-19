#include <cuda_runtime.h>
#include <math.h>

// Sliding-window self-attention (bidirectional):
//   out_i = softmax_{j in [i-w, i+w] ∩ [0, M)}( Q_i . K_j / sqrt(d) ) @ V
//
// window_size <= 32, so a query sees at most 65 keys. Each block owns BQ = 16
// consecutive queries and stages their whole key range [q0 - w, q0 + 15 + w]
// (<= 80 rows) of K and V in shared memory once. Every query's scores then fit
// in registers (lane l scores keys l, l+32, l+64 of the range), so the softmax
// is an exact two-pass max/exp/sum with no online rescaling. Scores can reach
// ~1e5 with inputs in [-100, 100]; subtracting the row max keeps exp() safe.
//  - 4 warps x 4 queries. P (normalized) goes to a per-warp smem row; for P.V
//    lane l owns output dims 4l..4l+3 and walks only the query's own window.
//  - Shared memory is sized at runtime from d and window_size; the K row pitch
//    is chosen so float4 row reads by consecutive lanes are bank-conflict-free.

#define BQ 16
#define NWARPS 4
#define QPW (BQ / NWARPS)
#define NTHREADS (NWARPS * 32)
#define MAXKEYS (BQ + 64)          // BQ + 2 * max window
#define SLOTS 3                    // ceil(65 / 32) key slots per lane

__global__ void __launch_bounds__(NTHREADS)
swa_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
           float* __restrict__ out, int M, int d, int w, int pitch, float scale)
{
    extern __shared__ __align__(16) float smem[];
    const int dv = (d + 3) & ~3;
    const int nk = BQ + 2 * w;                       // staged key rows
    float* sQ = smem;                                // [BQ][pitch]
    float* sK = sQ + BQ * pitch;                     // [nk][pitch]
    float* sV = sK + nk * pitch;                     // [nk][dv]
    float* sP = sV + nk * dv;                        // [NWARPS][QPW][MAXKEYS]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int q0 = blockIdx.x * BQ;
    const int kBase = q0 - w;                        // global row of staged key 0

    for (int idx = tid; idx < BQ * dv; idx += NTHREADS) {
        const int r = idx / dv, c = idx - r * dv, g = q0 + r;
        sQ[r * pitch + c] = (g < M && c < d) ? __ldg(Q + (size_t)g * d + c) : 0.f;
    }
    for (int idx = tid; idx < nk * dv; idx += NTHREADS) {
        const int r = idx / dv, c = idx - r * dv, g = kBase + r;
        const bool ok = (g >= 0 && g < M && c < d);
        sK[r * pitch + c] = ok ? __ldg(K + (size_t)g * d + c) : 0.f;
        sV[r * dv + c]    = ok ? __ldg(V + (size_t)g * d + c) : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int a = 0; a < QPW; ++a) {
        const int ql = warp * QPW + a;               // local query row
        const int i  = q0 + ql;
        if (i >= M) break;                           // warp-uniform
        const int lo = max(i - w, 0), hi = min(i + w, M - 1);   // global window
        const int klo = lo - kBase, khi = hi - kBase;           // staged indices
        const float* qr = sQ + ql * pitch;

        // ---- scores: lane handles staged keys klo + lane + 32*s ----
        float s[SLOTS];
        float mx = -INFINITY;
#pragma unroll
        for (int t = 0; t < SLOTS; ++t) {
            const int kk = klo + lane + 32 * t;
            float x = -INFINITY;
            if (kk <= khi) {
                const float* kr = sK + kk * pitch;
                float acc = 0.f;
                for (int c = 0; c < dv; c += 4) {
                    const float4 qv = *reinterpret_cast<const float4*>(qr + c);
                    const float4 kv = *reinterpret_cast<const float4*>(kr + c);
                    acc = fmaf(qv.x, kv.x, fmaf(qv.y, kv.y, fmaf(qv.z, kv.z, fmaf(qv.w, kv.w, acc))));
                }
                x = acc * scale;
            }
            s[t] = x;
            mx = fmaxf(mx, x);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));

        float sum = 0.f;
#pragma unroll
        for (int t = 0; t < SLOTS; ++t) {
            s[t] = (s[t] == -INFINITY) ? 0.f : expf(s[t] - mx);
            sum += s[t];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        const float inv = 1.f / sum;                 // key i itself is always in the window

        float* pr = sP + (warp * QPW + a) * MAXKEYS;
#pragma unroll
        for (int t = 0; t < SLOTS; ++t) {
            const int off = lane + 32 * t;           // offset within the window
            if (off < MAXKEYS) pr[off] = s[t] * inv;
        }
        __syncwarp();

        // ---- out = P V over the window; lane owns dims 4*lane .. +3 ----
        const int c = lane * 4;
        if (c < d) {
            float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
            const int n = khi - klo + 1;
            for (int jj = 0; jj < n; ++jj) {
                const float p = pr[jj];
                const float4 v = *reinterpret_cast<const float4*>(sV + (klo + jj) * dv + c);
                o.x = fmaf(p, v.x, o.x); o.y = fmaf(p, v.y, o.y);
                o.z = fmaf(p, v.z, o.z); o.w = fmaf(p, v.w, o.w);
            }
            float* dst = out + (size_t)i * d + c;
            dst[0] = o.x;
            if (c + 1 < d) dst[1] = o.y;
            if (c + 2 < d) dst[2] = o.z;
            if (c + 3 < d) dst[3] = o.w;
        }
        __syncwarp();                                // pr reused by next query
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int window_size) {
    const int w  = window_size;
    const int dv = (d + 3) & ~3;
    const int pitch = ((dv / 4) % 2 == 0) ? dv + 4 : dv;   // odd # of float4s per row
    const int nk = BQ + 2 * w;
    const size_t smem = sizeof(float) * ((size_t)BQ * pitch + (size_t)nk * pitch + (size_t)nk * dv +
                                         (size_t)NWARPS * QPW * MAXKEYS);
    static size_t configured = 0;
    if (smem > configured) {
        cudaFuncSetAttribute(swa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = smem;
    }
    const int blocks = (M + BQ - 1) / BQ;
    swa_kernel<<<blocks, NTHREADS, smem>>>(Q, K, V, output, M, d, w, pitch, 1.0f / sqrtf((float)d));
    cudaDeviceSynchronize();
}
