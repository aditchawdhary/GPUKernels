#include <cuda_runtime.h>

// Stage 1: qkv = x @ W_qkv   (tiled SGEMM, optional split-K into partial buffers)
// Stage 2: sum split-K partials, apply RoPE to Q and K, write Q_out and scatter
//          K/V into slot positions[b] of the caches.

#define BM 64
#define BN 64
#define BK 16
#define GEMM_THREADS 256   // 16 x 16 threads, each computes a 4 x 4 micro-tile

__global__ void __launch_bounds__(GEMM_THREADS)
sgemm_splitk(const float* __restrict__ X, const float* __restrict__ W,
             float* __restrict__ P, int Bm, int Kd, int N, int kPerSplit) {
    __shared__ float As[BK][BM];      // transposed A tile
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int split = blockIdx.z;
    const int kBeg = split * kPerSplit;
    const int kEnd = min(Kd, kBeg + kPerSplit);

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    for (int k0 = kBeg; k0 < kEnd; k0 += BK) {
        // A tile: BM x BK (rows of x), stored transposed.
        #pragma unroll
        for (int l = 0; l < (BM * BK) / GEMM_THREADS; ++l) {
            int e = tid + l * GEMM_THREADS;
            int r = e / BK, c = e % BK;
            int gr = m0 + r, gk = k0 + c;
            As[c][r] = (gr < Bm && gk < kEnd) ? __ldg(X + (size_t)gr * Kd + gk) : 0.f;
        }
        // B tile: BK x BN (rows of W), coalesced along N.
        #pragma unroll
        for (int l = 0; l < (BK * BN) / GEMM_THREADS; ++l) {
            int e = tid + l * GEMM_THREADS;
            int r = e / BN, c = e % BN;
            int gk = k0 + r, gc = n0 + c;
            Bs[r][c] = (gk < kEnd && gc < N) ? __ldg(W + (size_t)gk * N + gc) : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[k][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    float* Ps = P + (size_t)split * Bm * N;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int gr = m0 + ty + 16 * i;
        if (gr >= Bm) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int gc = n0 + tx + 16 * j;
            if (gc < N) Ps[(size_t)gr * N + gc] = acc[i][j];
        }
    }
}

// One thread per (b, head, j) with j < D/2; handles elements j and j + D/2.
__global__ void rope_scatter(const float* __restrict__ P, int splits,
                             const float* __restrict__ cs, const int* __restrict__ positions,
                             float* __restrict__ K_cache, float* __restrict__ V_cache,
                             float* __restrict__ Q_out,
                             int Bm, int H_q, int H_kv, int D, int S_max) {
    const int half = D >> 1;
    const int Htot = H_q + 2 * H_kv;
    const int N = Htot * D;
    const long long total = (long long)Bm * Htot * half;
    const size_t splitStride = (size_t)Bm * N;

    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        int j = (int)(idx % half);
        long long r = idx / half;
        int h = (int)(r % Htot);
        int b = (int)(r / Htot);

        size_t col1 = (size_t)b * N + (size_t)h * D + j;
        size_t col2 = col1 + half;
        float t1 = 0.f, t2 = 0.f;
        for (int s = 0; s < splits; ++s) {
            t1 += P[s * splitStride + col1];
            t2 += P[s * splitStride + col2];
        }

        int p = positions[b];
        if (h < H_q + H_kv) {
            float c  = __ldg(cs + (size_t)p * D + j);
            float sn = __ldg(cs + (size_t)p * D + half + j);
            float o1 = t1 * c - t2 * sn;
            float o2 = t2 * c + t1 * sn;
            if (h < H_q) {
                size_t dst = ((size_t)b * H_q + h) * D;
                Q_out[dst + j] = o1;
                Q_out[dst + j + half] = o2;
            } else {
                int hk = h - H_q;
                size_t dst = (((size_t)b * H_kv + hk) * S_max + p) * D;
                K_cache[dst + j] = o1;
                K_cache[dst + j + half] = o2;
            }
        } else {
            int hv = h - H_q - H_kv;
            size_t dst = (((size_t)b * H_kv + hv) * S_max + p) * D;
            V_cache[dst + j] = t1;            // values are not rotated
            V_cache[dst + j + half] = t2;
        }
    }
}

// x, W_qkv, cos_sin_cache, positions, K_cache, V_cache, Q_out are device pointers
extern "C" void solve(const float* x, const float* W_qkv, const float* cos_sin_cache,
                      const int* positions, float* K_cache, float* V_cache, float* Q_out, int B,
                      int d_model, int H_q, int H_kv, int D, int S_max) {
    const int N = (H_q + 2 * H_kv) * D;

    static int numSMs = 0;
    if (!numSMs) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
        if (numSMs <= 0) numSMs = 80;
    }

    // Split K when there are too few output tiles to fill the GPU (small B).
    const int tilesM = (B + BM - 1) / BM;
    const int tilesN = (N + BN - 1) / BN;
    const int baseBlocks = tilesM * tilesN;
    int splits = (4 * numSMs + baseBlocks - 1) / baseBlocks;
    int maxSplits = (d_model + 255) / 256;          // keep >= 256 k per split
    if (splits > maxSplits) splits = maxSplits;
    if (splits > 16) splits = 16;
    if (splits < 1) splits = 1;
    int kPerSplit = (d_model + splits - 1) / splits;
    kPerSplit = (kPerSplit + BK - 1) / BK * BK;
    splits = (d_model + kPerSplit - 1) / kPerSplit;

    static float* partials = nullptr;
    static size_t cap = 0;
    size_t need = (size_t)splits * B * N;
    if (need > cap) {
        if (partials) cudaFree(partials);
        cudaMalloc(&partials, need * sizeof(float));
        cap = need;
    }

    dim3 g1(tilesN, tilesM, splits);
    sgemm_splitk<<<g1, GEMM_THREADS>>>(x, W_qkv, partials, B, d_model, N, kPerSplit);

    long long pairs = (long long)B * (H_q + 2 * H_kv) * (D / 2);
    int t2 = 256;
    long long b2 = (pairs + t2 - 1) / t2;
    if (b2 > numSMs * 32) b2 = numSMs * 32;
    if (b2 < 1) b2 = 1;
    rope_scatter<<<(int)b2, t2>>>(partials, splits, cos_sin_cache, positions,
                                  K_cache, V_cache, Q_out, B, H_q, H_kv, D, S_max);
    cudaDeviceSynchronize();
}
