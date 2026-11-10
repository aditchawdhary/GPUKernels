#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NT 256

// ---------------- Row statistics: mean and 1/std (stable two-pass) ----------------
__device__ __forceinline__ float block_sum(float v, float* sh) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();                       // protect sh from previous use
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float t = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (lane == 0) sh[0] = t;
    }
    __syncthreads();
    return sh[0];
}

__global__ void __launch_bounds__(256)
stats_kernel(const float* __restrict__ x, int K, float eps,
             float* __restrict__ mean, float* __restrict__ rstd) {
    __shared__ float sh[32];
    const float* r = x + (size_t)blockIdx.x * K;
    const float shift = r[0];

    float s = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) s += r[k] - shift;
    s = block_sum(s, sh);
    const float mu = shift + s / (float)K;

    float v = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) { float d = r[k] - mu; v = fmaf(d, d, v); }
    v = block_sum(v, sh) / (float)K;

    if (threadIdx.x == 0) {
        mean[blockIdx.x] = mu;
        rstd[blockIdx.x] = 1.0f / sqrtf(v + eps);
    }
}

// ---------------- Fused LayerNorm -> GEMM -> bias -> GELU ----------------
template <bool VEC>
__global__ void __launch_bounds__(NT)
fused_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
             const float* __restrict__ beta, const float* __restrict__ W,
             const float* __restrict__ bias, const float* __restrict__ mean,
             const float* __restrict__ rstd, float* __restrict__ y,
             int M, int K, int N) {
    __shared__ __align__(16) float As[2][BK][BM];   // normalized x tile, transposed
    __shared__ __align__(16) float Bs[2][BK][BN];

    const int tid = threadIdx.x;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    // A-tile load mapping: 128 rows x 8 cols, 4 consecutive k per thread.
    const int aRow = tid >> 1, aCol = (tid & 1) * 4;
    const int gaRow = rowBase + aRow;
    const bool aRowOK = gaRow < M;
    const float mu = aRowOK ? mean[gaRow] : 0.0f;
    const float rs = aRowOK ? rstd[gaRow] : 0.0f;
    const float* xrow = x + (size_t)(aRowOK ? gaRow : 0) * K;

    // B-tile load mapping: 8 rows x 128 cols, 4 consecutive n per thread.
    const int bRow = tid >> 5, bCol = (tid & 31) * 4;
    const int gbCol = colBase + bCol;

    float ra[4], rb[4];

    auto gload = [&](int k0) {
        // A: load raw x, normalize + affine on the fly; out-of-range -> 0.
        int gk = k0 + aCol;
        if (VEC) {
            if (aRowOK && gk < K) {
                float4 v = *reinterpret_cast<const float4*>(xrow + gk);
                float4 g = *reinterpret_cast<const float4*>(gamma + gk);
                float4 b = *reinterpret_cast<const float4*>(beta + gk);
                ra[0] = fmaf((v.x - mu) * rs, g.x, b.x);
                ra[1] = fmaf((v.y - mu) * rs, g.y, b.y);
                ra[2] = fmaf((v.z - mu) * rs, g.z, b.z);
                ra[3] = fmaf((v.w - mu) * rs, g.w, b.w);
            } else { ra[0] = ra[1] = ra[2] = ra[3] = 0.0f; }
        } else {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int k = gk + e;
                ra[e] = (aRowOK && k < K) ? fmaf((xrow[k] - mu) * rs, gamma[k], beta[k]) : 0.0f;
            }
        }
        // B: W[k][n].
        int gkb = k0 + bRow;
        if (VEC) {
            if (gkb < K && gbCol < N) {
                float4 v = *reinterpret_cast<const float4*>(W + (size_t)gkb * N + gbCol);
                rb[0] = v.x; rb[1] = v.y; rb[2] = v.z; rb[3] = v.w;
            } else { rb[0] = rb[1] = rb[2] = rb[3] = 0.0f; }
        } else {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int n = gbCol + e;
                rb[e] = (gkb < K && n < N) ? W[(size_t)gkb * N + n] : 0.0f;
            }
        }
    };

    auto sstore = [&](int buf) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) As[buf][aCol + e][aRow] = ra[e];
        *reinterpret_cast<float4*>(&Bs[buf][bRow][bCol]) = make_float4(rb[0], rb[1], rb[2], rb[3]);
    };

    const int tr = tid / (BN / TN);   // 0..15
    const int tc = tid % (BN / TN);   // 0..15

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    gload(0);
    sstore(0);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool hasNext = (k0 + BK) < K;
        if (hasNext) gload(k0 + BK);

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
            float4 a0 = *reinterpret_cast<const float4*>(&As[buf][kk][tr * TM]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[buf][kk][tr * TM + 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[buf][kk][tc * TN]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[buf][kk][tc * TN + 4]);
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
            a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
            b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }

        if (hasNext) sstore(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }

    // Epilogue: + bias, exact GELU, store.
    const float inv_sqrt2 = 0.70710678118654752f;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gr = rowBase + tr * TM + i;
        if (gr >= M) continue;
        float* yrow = y + (size_t)gr * N;
        #pragma unroll
        for (int j4 = 0; j4 < TN; j4 += 4) {
            int gc = colBase + tc * TN + j4;
            float o[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int c = gc + e;
                float v = acc[i][j4 + e] + ((c < N) ? bias[c] : 0.0f);
                o[e] = 0.5f * v * (1.0f + erff(v * inv_sqrt2));
            }
            if (VEC) {
                if (gc < N) *reinterpret_cast<float4*>(yrow + gc) = make_float4(o[0], o[1], o[2], o[3]);
            } else {
                #pragma unroll
                for (int e = 0; e < 4; ++e) if (gc + e < N) yrow[gc + e] = o[e];
            }
        }
    }
}

static inline bool al16(const void* p) { return ((uintptr_t)p & 15) == 0; }

extern "C" void solve(const float* x, const float* gamma, const float* beta, const float* W,
                      const float* bias, float* y, int M, int K, int N, float eps) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    static float* stats = nullptr;
    static int cap = 0;
    if (cap < M) {
        if (stats) cudaFree(stats);
        cap = M;
        cudaMalloc(&stats, 2 * (size_t)cap * sizeof(float));
    }
    float* mean = stats;
    float* rstd = stats + cap;

    stats_kernel<<<M, 256>>>(x, K, eps, mean, rstd);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    bool vec = (K % 4 == 0) && (N % 4 == 0) &&
               al16(x) && al16(gamma) && al16(beta) && al16(W) && al16(y);
    if (vec) fused_kernel<true ><<<grid, NT>>>(x, gamma, beta, W, bias, mean, rstd, y, M, K, N);
    else     fused_kernel<false><<<grid, NT>>>(x, gamma, beta, W, bias, mean, rstd, y, M, K, N);
    cudaDeviceSynchronize();
}
