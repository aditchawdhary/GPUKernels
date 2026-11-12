#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// Fused RMSNorm + W8A8 linear
//   1) per-row: RMSNorm * gamma, symmetric per-token int8 quantization -> Aq (int8), s (float)
//   2) int8 x int8 GEMM with exact int32 accumulation via __dp4a, dequant epilogue
//      output[i,j] = s[i] * w_scale[j] * sum_k q[i,k] * W[j,k]

// ---------------------------------------------------------------- quantize
#define QT 256

__global__ void __launch_bounds__(QT)
rmsnorm_quant(const float* __restrict__ x, const float* __restrict__ gamma,
              int8_t* __restrict__ Aq, float* __restrict__ sOut, int K, int Kp) {
    __shared__ float shS[QT / 32], shM[QT / 32];
    __shared__ float sQscale;
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * K;

    // Pass 1: sum of squares and max |x * gamma| (max|y| = inv_rms * max|x*gamma|).
    float ss = 0.f, mg = 0.f;
    for (int k = threadIdx.x; k < K; k += QT) {
        float v = __ldg(xr + k);
        ss = fmaf(v, v, ss);
        mg = fmaxf(mg, fabsf(v * __ldg(gamma + k)));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
        mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, o));
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { shS[wid] = ss; shM[wid] = mg; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tS = 0.f, tM = 0.f;
        for (int w = 0; w < QT / 32; ++w) { tS += shS[w]; tM = fmaxf(tM, shM[w]); }
        double inv = 1.0 / sqrt((double)tS / K + 1e-6);
        double s = inv * (double)tM / 127.0;           // per-token scale
        sOut[row] = (float)s;
        // y / s = x * gamma * 127 / max|x*gamma|   (inv_rms cancels)
        sQscale = (tM > 0.f) ? (float)(127.0 / (double)tM) : 0.f;
    }
    __syncthreads();
    const float qs = sQscale;

    // Pass 2: quantize, round-to-nearest-even, pack 4 int8 per int, zero-pad to Kp.
    int* outRow = reinterpret_cast<int*>(Aq + (size_t)row * Kp);
    const int Kp4 = Kp >> 2;
    for (int k4 = threadIdx.x; k4 < Kp4; k4 += QT) {
        uint32_t packed = 0;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int k = k4 * 4 + u;
            int q = 0;
            if (k < K) {
                float t = rintf(__ldg(xr + k) * __ldg(gamma + k) * qs);
                t = fminf(fmaxf(t, -127.f), 127.f);
                q = (int)t;
            }
            packed |= ((uint32_t)(q & 0xFF)) << (8 * u);
        }
        outRow[k4] = (int)packed;
    }
}

// Copy W into a K-padded buffer when K is not a multiple of 4.
__global__ void pad_weights(const int8_t* __restrict__ W, int8_t* __restrict__ Wp,
                            int N, int K, int Kp) {
    size_t total = (size_t)N * Kp;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        int n = (int)(i / Kp), k = (int)(i % Kp);
        Wp[i] = (k < K) ? W[(size_t)n * K + k] : (int8_t)0;
    }
}

// ---------------------------------------------------------------- int8 GEMM
#define BM 128
#define BN 128
#define BK4 8            // 8 ints = 32 int8 values of K per tile
#define GT 256           // 16 x 16 threads, 8 x 8 outputs each

__global__ void __launch_bounds__(GT)
gemm_dp4a(const int* __restrict__ A, const int* __restrict__ B,
          const float* __restrict__ sA, const float* __restrict__ sB,
          float* __restrict__ C, int M, int N, int Kp4) {
    __shared__ int As[BK4][BM + 4];
    __shared__ int Bs[BK4][BN + 4];

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    int acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0;

    // Register prefetch of the next tile (4 ints of A and 4 of B per thread).
    int ra[4], rb[4];
    auto load = [&](int k0) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int e = tid + l * GT;
            int r = e / BK4, kk = e % BK4;
            int gk = k0 + kk;
            int gr = m0 + r, gc = n0 + r;
            ra[l] = (gr < M && gk < Kp4) ? __ldg(A + (size_t)gr * Kp4 + gk) : 0;
            rb[l] = (gc < N && gk < Kp4) ? __ldg(B + (size_t)gc * Kp4 + gk) : 0;
        }
    };

    load(0);
    for (int k0 = 0; k0 < Kp4; k0 += BK4) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int e = tid + l * GT;
            int r = e / BK4, kk = e % BK4;
            As[kk][r] = ra[l];
            Bs[kk][r] = rb[l];
        }
        __syncthreads();
        if (k0 + BK4 < Kp4) load(k0 + BK4);   // overlap global loads with compute

        #pragma unroll
        for (int kk = 0; kk < BK4; ++kk) {
            int a[8], b[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = As[kk][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = Bs[kk][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    float wsv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int gc = n0 + tx + 16 * j;
        wsv[j] = (gc < N) ? __ldg(sB + gc) : 0.f;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int gr = m0 + ty + 16 * i;
        if (gr >= M) continue;
        float si = __ldg(sA + gr);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int gc = n0 + tx + 16 * j;
            if (gc < N) C[(size_t)gr * N + gc] = (float)acc[i][j] * si * wsv[j];
        }
    }
}

// x, gamma, W, w_scale, output are device pointers
extern "C" void solve(const float* x, const float* gamma, const int8_t* W, const float* w_scale,
                      float* output, int M, int N, int K) {
    const int Kp = (K + 3) & ~3;          // K padded to a multiple of 4 bytes
    const int Kp4 = Kp >> 2;

    static int8_t* Aq = nullptr;  static size_t capA = 0;
    static float*  sA = nullptr;  static size_t capS = 0;
    static int8_t* Wp = nullptr;  static size_t capW = 0;

    size_t needA = (size_t)M * Kp;
    if (needA > capA) { if (Aq) cudaFree(Aq); cudaMalloc(&Aq, needA); capA = needA; }
    if ((size_t)M > capS) { if (sA) cudaFree(sA); cudaMalloc(&sA, M * sizeof(float)); capS = M; }

    const int8_t* Wuse = W;
    if (K != Kp) {
        size_t needW = (size_t)N * Kp;
        if (needW > capW) { if (Wp) cudaFree(Wp); cudaMalloc(&Wp, needW); capW = needW; }
        pad_weights<<<1024, 256>>>(W, Wp, N, K, Kp);
        Wuse = Wp;
    }

    rmsnorm_quant<<<M, QT>>>(x, gamma, Aq, sA, K, Kp);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_dp4a<<<grid, GT>>>(reinterpret_cast<const int*>(Aq),
                            reinterpret_cast<const int*>(Wuse),
                            sA, w_scale, output, M, N, Kp4);
    cudaDeviceSynchronize();
}
