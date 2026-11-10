#include <cuda_runtime.h>

// LayerNorm backward.
//   xhat = (x - mean) * rstd,   g = dy * gamma
//   dx     = rstd * ( g - mean_c(g) - xhat * mean_c(g * xhat) )
//   dgamma = sum_n dy * xhat
//   dbeta  = sum_n dy
//
// Kernel 1 (fused): each block owns a contiguous band of rows. Every thread
// owns a fixed set of columns, so dgamma/dbeta partials accumulate (in that
// thread's own shared-memory slots) across all rows of the band; x and dy
// are read from HBM once.
// Per row: pass 1 builds xhat/g (kept in registers) and the two row sums,
// one block reduction, pass 2 writes dx.
// Kernel 2: column-sum the per-block partials into dgamma / dbeta.

#define RED_COLS 32
#define RED_ROWS 32

// ---------------------------------------------------------------- helpers
template <int T>
__device__ __forceinline__ void blockSum2(float& a, float& b, float (*sh)[2][T / 32], int buf)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xffffffff, a, off);
        b += __shfl_xor_sync(0xffffffff, b, off);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sh[buf][0][warp] = a; sh[buf][1][warp] = b; }
    __syncthreads();
    a = 0.f; b = 0.f;
    #pragma unroll
    for (int w = 0; w < T / 32; ++w) { a += sh[buf][0][w]; b += sh[buf][1][w]; }
    // Double-buffered by row parity: the next write to sh[buf] happens two
    // rows later, after another __syncthreads, so no reader can be overtaken.
}

// Element e (0 <= e < 4K) of thread tid -> column index.
//   VEC4:   groups of 4 contiguous floats, group j at 4*(tid + j*T)
//   scalar: c = tid + e*T  (coalesced)
template <int T, bool VEC4>
__device__ __forceinline__ int colOf(int tid, int e)
{
    return VEC4 ? 4 * (tid + (e >> 2) * T) + (e & 3) : tid + e * T;
}

// Streaming (evict-first) load when the row is never re-read; normal cached
// load when pass 2 will read the row again.
template <bool STREAM, class V>
__device__ __forceinline__ V ld(const V* p) { return STREAM ? __ldcs(p) : __ldg(p); }

// ---------------------------------------------------------------- kernel 1
// T threads, E = 4K elements per thread (C <= E*T).
// CACHE: keep xhat and g in registers between the two passes (small C);
//        otherwise re-read the row (still L1/L2 hot) in pass 2.
// Ask for >= 3 resident 256-thread blocks/SM (<= 85 regs): 24 warps of
// latency hiding for this memory-bound loop.
template <int T, int K, bool CACHE, bool VEC4>
__global__ void __launch_bounds__(T, (T <= 256) ? 3 : 1)
ln_bwd_rows(const float* __restrict__ x, const float* __restrict__ gamma,
            const float* __restrict__ mean, const float* __restrict__ rstd,
            const float* __restrict__ dy, float* __restrict__ dx,
            float* __restrict__ pdg, float* __restrict__ pdb,
            int N, int C, int G)
{
    constexpr int E = 4 * K;
    __shared__ float sh[2][2][T / 32];
    // Column partials for this block: [0,C) dgamma, [C,2C) dbeta.
    // Each slot is only ever touched by the thread that owns that column,
    // so no synchronization is needed. Keeps registers (and occupancy) sane.
    extern __shared__ float4 sacc4[];
    float* sacc = reinterpret_cast<float*>(sacc4);

    const int tid = threadIdx.x;
    const int r0 = (int)(((long long)blockIdx.x * N) / G);
    const int r1 = (int)(((long long)(blockIdx.x + 1) * N) / G);
    const float invC = 1.0f / (float)C;

    float xh[CACHE ? E : 1], gg[CACHE ? E : 1];

    #pragma unroll
    for (int e = 0; e < E; ++e) {
        const int c = colOf<T, VEC4>(tid, e);
        if (c < C) { sacc[c] = 0.f; sacc[C + c] = 0.f; }
    }

    for (int row = r0; row < r1; ++row) {
        const float m = __ldg(mean + row), rs = __ldg(rstd + row);
        const float* xr  = x  + (size_t)row * C;
        const float* dyr = dy + (size_t)row * C;
        float*       dxr = dx + (size_t)row * C;

        // ---- pass 1: load, accumulate row sums and column partials
        float s1 = 0.f, s2 = 0.f;
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            float xv[4], dv[4];
            if (VEC4) {
                const int c = 4 * (tid + j * T);
                if (c < C) {
                    float4 a = ld<CACHE>(reinterpret_cast<const float4*>(xr + c));
                    float4 b = ld<CACHE>(reinterpret_cast<const float4*>(dyr + c));
                    xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
                    dv[0] = b.x; dv[1] = b.y; dv[2] = b.z; dv[3] = b.w;
                } else {
                    #pragma unroll
                    for (int v = 0; v < 4; ++v) { xv[v] = m; dv[v] = 0.f; }
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int c = tid + (j * 4 + v) * T;
                    xv[v] = (c < C) ? ld<CACHE>(xr + c)  : m;
                    dv[v] = (c < C) ? ld<CACHE>(dyr + c) : 0.f;
                }
            }
            float gv[4], h[4];
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int c = colOf<T, VEC4>(tid, j * 4 + v);
                gv[v] = (c < C) ? __ldg(gamma + c) : 0.f;   // L1-resident
                h[v]  = (xv[v] - m) * rs;                   // 0 for padded lanes
                const float g = dv[v] * gv[v];
                s1 += g;
                s2 = fmaf(g, h[v], s2);
                if (CACHE) { xh[j * 4 + v] = h[v]; gg[j * 4 + v] = g; }
            }
            // column partials in this thread's private smem slots
            if (VEC4) {
                const int q = tid + j * T;                  // float4 index
                if (4 * q < C) {
                    float4 ag = sacc4[q], ab = sacc4[(C >> 2) + q];
                    ag.x = fmaf(dv[0], h[0], ag.x); ag.y = fmaf(dv[1], h[1], ag.y);
                    ag.z = fmaf(dv[2], h[2], ag.z); ag.w = fmaf(dv[3], h[3], ag.w);
                    ab.x += dv[0]; ab.y += dv[1]; ab.z += dv[2]; ab.w += dv[3];
                    sacc4[q] = ag; sacc4[(C >> 2) + q] = ab;
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int c = tid + (j * 4 + v) * T;
                    if (c < C) {
                        sacc[c]     = fmaf(dv[v], h[v], sacc[c]);
                        sacc[C + c] += dv[v];
                    }
                }
            }
        }

        blockSum2<T>(s1, s2, sh, row & 1);
        const float mg  = s1 * invC;   // mean(g)
        const float mgh = s2 * invC;   // mean(g * xhat)

        // ---- pass 2: dx
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            float h[4], g[4];
            if (CACHE) {
                #pragma unroll
                for (int v = 0; v < 4; ++v) { h[v] = xh[j * 4 + v]; g[v] = gg[j * 4 + v]; }
            } else {
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int c = colOf<T, VEC4>(tid, j * 4 + v);
                    const bool ok = c < C;
                    h[v] = ok ? (__ldg(xr + c) - m) * rs : 0.f;
                    g[v] = ok ? __ldg(dyr + c) * __ldg(gamma + c) : 0.f;
                }
            }
            float o[4];
            #pragma unroll
            for (int v = 0; v < 4; ++v) o[v] = rs * (g[v] - mg - h[v] * mgh);

            if (VEC4) {
                const int c = 4 * (tid + j * T);
                if (c < C)
                    __stcs(reinterpret_cast<float4*>(dxr + c), make_float4(o[0], o[1], o[2], o[3]));
            } else {
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int c = tid + (j * 4 + v) * T;
                    if (c < C) __stcs(dxr + c, o[v]);
                }
            }
        }
    }

    // ---- write this block's column partials (zeros if the band was empty)
    float* dgp = pdg + (size_t)blockIdx.x * C;
    float* dbp = pdb + (size_t)blockIdx.x * C;
    #pragma unroll
    for (int e = 0; e < E; ++e) {
        const int c = colOf<T, VEC4>(tid, e);
        if (c < C) { dgp[c] = sacc[c]; dbp[c] = sacc[C + c]; }
    }
}

// ---------------------------------------------------------------- kernel 2
// dgamma[c] = sum_b pdg[b][c], dbeta[c] = sum_b pdb[b][c]
__global__ void __launch_bounds__(RED_COLS * RED_ROWS)
ln_bwd_colsum(const float* __restrict__ pdg, const float* __restrict__ pdb,
              float* __restrict__ dgamma, float* __restrict__ dbeta, int G, int C)
{
    __shared__ float sg[RED_ROWS][RED_COLS + 1];
    __shared__ float sb[RED_ROWS][RED_COLS + 1];

    const int c = blockIdx.x * RED_COLS + threadIdx.x;
    float ag = 0.f, ab = 0.f;
    if (c < C) {
        for (int b = threadIdx.y; b < G; b += RED_ROWS) {
            ag += pdg[(size_t)b * C + c];
            ab += pdb[(size_t)b * C + c];
        }
    }
    sg[threadIdx.y][threadIdx.x] = ag;
    sb[threadIdx.y][threadIdx.x] = ab;
    __syncthreads();

    // threadIdx.y == 0 row of threads finishes each column
    if (threadIdx.y == 0 && c < C) {
        float tg = 0.f, tb = 0.f;
        #pragma unroll
        for (int r = 0; r < RED_ROWS; ++r) { tg += sg[r][threadIdx.x]; tb += sb[r][threadIdx.x]; }
        dgamma[c] = tg;
        dbeta[c]  = tb;
    }
}

// ---------------------------------------------------------------- host
template <int T, int K, bool CACHE, bool VEC4>
static void launch(const float* x, const float* gamma, const float* mean, const float* rstd,
                   const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C)
{
    auto kern = ln_bwd_rows<T, K, CACHE, VEC4>;
    const size_t smem = (size_t)2 * C * sizeof(float);   // up to 128 KB at C = 16384
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, T, smem);
    if (occ < 1) occ = 1;
    int G = sms * occ;            // one resident wave; each block takes N/G rows
    if (G > N) G = N;

    // Scratch for per-block partials, grown on demand and reused across calls
    static float* scratch = nullptr;
    static size_t cap = 0;
    const size_t need = (size_t)2 * G * C;
    if (need > cap) {
        if (scratch) cudaFree(scratch);
        cudaMalloc(&scratch, need * sizeof(float));
        cap = need;
    }
    float* pdg = scratch;
    float* pdb = scratch + (size_t)G * C;

    kern<<<G, T, smem>>>(x, gamma, mean, rstd, dy, dx, pdg, pdb, N, C, G);

    dim3 rb(RED_COLS, RED_ROWS);
    ln_bwd_colsum<<<(C + RED_COLS - 1) / RED_COLS, rb>>>(pdg, pdb, dgamma, dbeta, G, C);
}

template <bool VEC4>
static void dispatch(const float* x, const float* gamma, const float* mean, const float* rstd,
                     const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C)
{
    if      (C <= 1024) launch<256, 1, true,  VEC4>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    else if (C <= 2048) launch<256, 2, true,  VEC4>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    else if (C <= 4096) launch<256, 4, true,  VEC4>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    else                launch<512, 8, false, VEC4>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C); // C <= 16384
}

// x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
extern "C" void solve(const float* x, const float* gamma, const float* mean, const float* rstd,
                      const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C)
{
    if ((C & 3) == 0) dispatch<true >(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    else              dispatch<false>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    cudaDeviceSynchronize();
}
