/*
Multi-LoRA Batched Linear
x: (T, K), W: (K, N), A: (L, K, R), B: (L, R, N), adapter_ids: (T) int32 in [-1, L).
  y[t] = x[t] W + (id_t >= 0 ? scale * (x[t] A[id_t]) B[id_t] : 0)
(Spec: leetgpu-challenges #133, tolerance 1e-3.)
Constraints: T ≤ 4096, K, N ≤ 8192, R ≤ 64, L ≤ 64.
Performance is measured with T = 1024, K = N = 4096, R = 16, L = 8.
*/

/*
Approach (2 launches, FP32 throughout):
 1. lora_shrink   one block per token: s[t, :] = scale * x[t] . A[id_t] (R values),
                  or 0 when id_t = -1. Threads are split as (k-group g, rank r):
                  r = tid % R, g = tid / R, so the loads of A[id][k][r] are
                  coalesced over r. The groups stride over K, and their partials
                  are summed through shared memory. That's T*K*R FMAs (67M at the
                  benchmark), which is negligible.
 2. base_gemm_lora  y = x W, an FP32 register-tiled SGEMM (128x128 or 64x64 tiles,
                  BK = 8, 256 threads, double-buffered shared memory with register
                  prefetch). The epilogue adds the LoRA expand for each output row
                  with that row's own adapter: y[t, n] += sum_r s[t, r] * B[id_t][r][n].
                  Since s already holds the scale, rows with id = -1 skip it. B is
                  only L*R*N floats (2 MB at the benchmark), so these reads are
                  L1/L2 hits. The expand costs T*N*R FMAs, about 1% of the base GEMM.
 - Tiles: 128x128 (8x8 per thread) unless that gives fewer than about 2 blocks per
   SM, then 64x64 (4x4 per thread).
 - VEC path (K % 4 == 0 and N % 4 == 0, which covers the benchmark): float4 global
   loads and stores. Otherwise guarded scalar accesses, so any K, N, T works.
*/

#include <cuda_runtime.h>

#define GT 256
#define GBK 8

// ======================================================================== 1. shrink
__global__ void __launch_bounds__(GT)
lora_shrink(const float* __restrict__ x, const float* __restrict__ A, const int* __restrict__ ids,
            float* __restrict__ S, int K, int R, float scale)
{
    __shared__ float red[GT];
    const int t = blockIdx.x, tid = threadIdx.x;
    const int id = ids[t];
    if (id < 0) {                                     // no adapter: s = 0 (uniform branch)
        for (int r = tid; r < R; r += GT) S[(size_t)t * R + r] = 0.f;
        return;
    }
    const int G = GT / R;                             // k-groups (R ≤ 64 -> G ≥ 4)
    const int r = tid % R, g = tid / R;
    float acc = 0.f;
    if (g < G) {
        const float* xr = x + (size_t)t * K;
        const float* Ar = A + (size_t)id * K * R + r;
        for (int k = g; k < K; k += G) acc = fmaf(__ldg(xr + k), __ldg(Ar + (size_t)k * R), acc);
    }
    red[tid] = acc;
    __syncthreads();
    if (tid < R) {
        float s = 0.f;
        for (int gg = 0; gg < G; ++gg) s += red[gg * R + tid];
        S[(size_t)t * R + tid] = s * scale;
    }
}

// ======================================================================== 2. base GEMM + LoRA expand
template <bool VEC>
__device__ __forceinline__ float4 ld4(const float* p, int col, int limit, bool ok)
{
    if (!ok) return make_float4(0.f, 0.f, 0.f, 0.f);
    if (VEC) return (col < limit) ? __ldg(reinterpret_cast<const float4*>(p + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 v;
    v.x = (col     < limit) ? __ldg(p + col)     : 0.f;
    v.y = (col + 1 < limit) ? __ldg(p + col + 1) : 0.f;
    v.z = (col + 2 < limit) ? __ldg(p + col + 2) : 0.f;
    v.w = (col + 3 < limit) ? __ldg(p + col + 3) : 0.f;
    return v;
}

template <int BM, int BN, bool VEC>
__global__ void __launch_bounds__(GT)
base_gemm_lora(const float* __restrict__ X, const float* __restrict__ W,
               const float* __restrict__ Bl, const int* __restrict__ ids,
               const float* __restrict__ S, float* __restrict__ Y,
               int T, int K, int N, int R)
{
    constexpr int TM = BM / 16, TN = BN / 16, HM = TM / 4, HN = TN / 4;
    constexpr int NA = (BM * 2 + GT - 1) / GT;        // float4 loads of A per thread
    constexpr int NB = (BN * 2 + GT - 1) / GT;        // float4 loads of B per thread
    __shared__ __align__(16) float As[2][GBK][BM];
    __shared__ __align__(16) float Bs[2][GBK][BN];

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    float4 ra[NA], rb[NB];

    auto load = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < NA; ++i) {                // A: BM rows x 8 k  (float4 along k)
            const int f = tid + i * GT;
            if (f < BM * 2) {
                const int r = f >> 1, kk = k0 + (f & 1) * 4;
                ra[i] = ld4<VEC>(X + (size_t)(m0 + r) * K, kk, K, m0 + r < T);
            }
        }
        #pragma unroll
        for (int i = 0; i < NB; ++i) {                // B: 8 k x BN cols (float4 along n)
            const int f = tid + i * GT;
            if (f < BN * 2) {
                const int kr = f / (BN / 4), c = (f % (BN / 4)) * 4;
                rb[i] = ld4<VEC>(W + (size_t)(k0 + kr) * N, n0 + c, N, k0 + kr < K);
            }
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int i = 0; i < NA; ++i) {
            const int f = tid + i * GT;
            if (f < BM * 2) {
                const int r = f >> 1, kk = (f & 1) * 4;
                As[buf][kk][r] = ra[i].x; As[buf][kk + 1][r] = ra[i].y;
                As[buf][kk + 2][r] = ra[i].z; As[buf][kk + 3][r] = ra[i].w;
            }
        }
        #pragma unroll
        for (int i = 0; i < NB; ++i) {
            const int f = tid + i * GT;
            if (f < BN * 2) {
                const int kr = f / (BN / 4), c = (f % (BN / 4)) * 4;
                *reinterpret_cast<float4*>(&Bs[buf][kr][c]) = rb[i];
            }
        }
    };

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    load(0); store(0);
    __syncthreads();
    const int KT = (K + GBK - 1) / GBK;
    for (int kt = 0; kt < KT; ++kt) {
        const int cur = kt & 1;
        const bool more = kt + 1 < KT;
        if (more) load((kt + 1) * GBK);
        #pragma unroll
        for (int kk = 0; kk < GBK; ++kk) {
            float a[TM], b[TN];
            #pragma unroll
            for (int h = 0; h < HM; ++h) {
                const float4 v = *reinterpret_cast<const float4*>(&As[cur][kk][h * (BM / 2) + ty * 4]);
                a[h * 4] = v.x; a[h * 4 + 1] = v.y; a[h * 4 + 2] = v.z; a[h * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int h = 0; h < HN; ++h) {
                const float4 v = *reinterpret_cast<const float4*>(&Bs[cur][kk][h * (BN / 2) + tx * 4]);
                b[h * 4] = v.x; b[h * 4 + 1] = v.y; b[h * 4 + 2] = v.z; b[h * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (more) store(cur ^ 1);
        __syncthreads();
    }

    // ---- epilogue: + s[t] B[id_t], then store
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int m = m0 + (i / 4) * (BM / 2) + ty * 4 + (i & 3);
        if (m >= T) continue;
        const int id = __ldg(ids + m);
        if (id >= 0) {
            const float* srow = S + (size_t)m * R;
            const float* bbase = Bl + (size_t)id * R * N;
            for (int r = 0; r < R; ++r) {
                const float sv = __ldg(srow + r);
                const float* brow = bbase + (size_t)r * N;
                #pragma unroll
                for (int h = 0; h < HN; ++h) {
                    const float4 bv = ld4<VEC>(brow, n0 + h * (BN / 2) + tx * 4, N, true);
                    acc[i][h * 4]     = fmaf(sv, bv.x, acc[i][h * 4]);
                    acc[i][h * 4 + 1] = fmaf(sv, bv.y, acc[i][h * 4 + 1]);
                    acc[i][h * 4 + 2] = fmaf(sv, bv.z, acc[i][h * 4 + 2]);
                    acc[i][h * 4 + 3] = fmaf(sv, bv.w, acc[i][h * 4 + 3]);
                }
            }
        }
        float* yrow = Y + (size_t)m * N;
        #pragma unroll
        for (int h = 0; h < HN; ++h) {
            const int n = n0 + h * (BN / 2) + tx * 4;
            if (VEC) {
                if (n < N) *reinterpret_cast<float4*>(yrow + n) =
                    make_float4(acc[i][h * 4], acc[i][h * 4 + 1], acc[i][h * 4 + 2], acc[i][h * 4 + 3]);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) if (n + j < N) yrow[n + j] = acc[i][h * 4 + j];
            }
        }
    }
}

// ======================================================================== host
static float* g_S = nullptr;
static size_t g_cap = 0;

template <int BM, int BN>
static void launch(const float* x, const float* W, const float* B, const int* ids, const float* S,
                   float* y, int T, int K, int N, int R, bool vec)
{
    dim3 grid((N + BN - 1) / BN, (T + BM - 1) / BM);
    if (vec) base_gemm_lora<BM, BN, true ><<<grid, GT>>>(x, W, B, ids, S, y, T, K, N, R);
    else     base_gemm_lora<BM, BN, false><<<grid, GT>>>(x, W, B, ids, S, y, T, K, N, R);
}

// x, W, A, B, adapter_ids, y are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      const int* adapter_ids, float* y, int T, int K, int N, int R, int L,
                      float scale)
{
    if (T <= 0 || K <= 0 || N <= 0 || R <= 0) return;
    const size_t need = (size_t)T * R;
    if (need > g_cap) {
        if (g_S) cudaFree(g_S);
        cudaMalloc(&g_S, need * sizeof(float));
        g_cap = need;
    }
    lora_shrink<<<T, GT>>>(x, A, adapter_ids, g_S, K, R, scale);

    int dev = 0, sms = 80;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const bool vec = (K % 4 == 0) && (N % 4 == 0);
    const long long big = (long long)((T + 127) / 128) * ((N + 127) / 128);
    if (big >= 2LL * sms) launch<128, 128>(x, W, B, adapter_ids, g_S, y, T, K, N, R, vec);
    else                  launch<64, 64>(x, W, B, adapter_ids, g_S, y, T, K, N, R, vec);
    cudaDeviceSynchronize();
}
