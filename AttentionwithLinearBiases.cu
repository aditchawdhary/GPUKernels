#include <cuda_runtime.h>
#include <math.h>

// ALiBi attention:  out = softmax( Q K^T / sqrt(d) + alpha * (i - j) ) V
//   1) S = Q K^T * (1/sqrt(d)) + alpha*(i-j)      tiled SGEMM (B transposed) + epilogue
//   2) P = row-softmax(S)                          in place
//   3) out = P V                                   tiled SGEMM

#define BM 64
#define BN 64
#define BK 16
#define GT 256   // 16x16 threads, each a 4x4 micro-tile

// C[Mr, Nc] = A[Mr, Kd] * op(B)
//   TRANS_B = true : B is [Nc, Kd] row-major (use B^T)
//   TRANS_B = false: B is [Kd, Nc] row-major
// If ALIBI: C = C*scale + alpha*(row - col)
template <bool TRANS_B, bool ALIBI>
__global__ void __launch_bounds__(GT)
sgemm(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
      int Mr, int Nc, int Kd, float scale, float alpha) {
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < Kd; k0 += BK) {
        // A tile (BM x BK), coalesced along K, stored transposed.
        #pragma unroll
        for (int l = 0; l < (BM * BK) / GT; ++l) {
            int e = tid + l * GT;
            int r = e / BK, c = e % BK;
            int gr = m0 + r, gk = k0 + c;
            As[c][r] = (gr < Mr && gk < Kd) ? __ldg(A + (size_t)gr * Kd + gk) : 0.f;
        }
        // B tile (BK x BN)
        #pragma unroll
        for (int l = 0; l < (BK * BN) / GT; ++l) {
            int e = tid + l * GT;
            if (TRANS_B) {
                int c = e / BK, k = e % BK;            // coalesced along K
                int gc = n0 + c, gk = k0 + k;
                Bs[k][c] = (gc < Nc && gk < Kd) ? __ldg(B + (size_t)gc * Kd + gk) : 0.f;
            } else {
                int k = e / BN, c = e % BN;            // coalesced along N
                int gc = n0 + c, gk = k0 + k;
                Bs[k][c] = (gc < Nc && gk < Kd) ? __ldg(B + (size_t)gk * Nc + gc) : 0.f;
            }
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

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int gr = m0 + ty + 16 * i;
        if (gr >= Mr) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int gc = n0 + tx + 16 * j;
            if (gc >= Nc) continue;
            float v = acc[i][j];
            if (ALIBI) v = v * scale + alpha * (float)(gr - gc);
            C[(size_t)gr * Nc + gc] = v;
        }
    }
}

#define ST 256
__device__ __forceinline__ float block_reduce(float v, bool isMax, float* sh) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float w = __shfl_xor_sync(0xffffffffu, v, o);
        v = isMax ? fmaxf(v, w) : v + w;
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (lane < ST / 32) ? sh[lane] : (isMax ? -INFINITY : 0.f);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float w = __shfl_xor_sync(0xffffffffu, v, o);
        v = isMax ? fmaxf(v, w) : v + w;
    }
    __syncthreads();   // sh reusable afterwards
    return v;          // valid in all threads
}

// One block per row; row length N <= 2048 -> at most 8 values per thread.
__global__ void __launch_bounds__(ST)
softmax_rows(float* __restrict__ S, int N) {
    __shared__ float sh[ST / 32];
    float* row = S + (size_t)blockIdx.x * N;

    float mx = -INFINITY;
    for (int j = threadIdx.x; j < N; j += ST) mx = fmaxf(mx, row[j]);
    mx = block_reduce(mx, true, sh);

    float sum = 0.f;
    for (int j = threadIdx.x; j < N; j += ST) {
        float e = expf(row[j] - mx);
        row[j] = e;
        sum += e;
    }
    sum = block_reduce(sum, false, sh);

    float inv = 1.f / sum;
    for (int j = threadIdx.x; j < N; j += ST) row[j] *= inv;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) {
    static float* S = nullptr;
    static size_t cap = 0;
    size_t need = (size_t)M * N;
    if (need > cap) {
        if (S) cudaFree(S);
        cudaMalloc(&S, need * sizeof(float));
        cap = need;
    }

    const float scale = 1.0f / sqrtf((float)d);

    dim3 g1((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm<true, true><<<g1, GT>>>(Q, K, S, M, N, d, scale, alpha);

    softmax_rows<<<M, ST>>>(S, N);

    dim3 g3((d + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm<false, false><<<g3, GT>>>(S, V, output, M, d, N, 1.f, 0.f);

    cudaDeviceSynchronize();
}
