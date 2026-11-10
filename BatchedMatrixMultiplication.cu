#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define NTHREADS 256   // (BM/TM) * (BN/TN)

__global__ void __launch_bounds__(NTHREADS)
bmm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
           int M, int N, int K) {
    __shared__ float As[BK][BM + 4];   // A tile stored transposed: As[k][m]
    __shared__ float Bs[BK][BN];

    const int batch = blockIdx.z;
    const float* Ab = A + (size_t)batch * M * K;
    const float* Bb = B + (size_t)batch * K * N;
    float*       Cb = C + (size_t)batch * M * N;

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN);    // 0..15
    const int ty = tid / (BN / TN);    // 0..15

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile (BM x BK), coalesced along k, stored transposed.
        #pragma unroll
        for (int i = 0; i < (BM * BK) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int r = idx / BK, c = idx % BK;
            int gr = rowBase + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? Ab[(size_t)gr * K + gc] : 0.0f;
        }
        // Load B tile (BK x BN), coalesced along n.
        #pragma unroll
        for (int i = 0; i < (BK * BN) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int r = idx / BN, c = idx % BN;
            int gr = k0 + r, gc = colBase + c;
            Bs[r][c] = (gr < K && gc < N) ? Bb[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty + i * (BM / TM)];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx + j * (BN / TN)];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gr = rowBase + ty + i * (BM / TM);
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gc = colBase + tx + j * (BN / TN);
            if (gc < N) Cb[(size_t)gr * N + gc] = acc[i][j];
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 block(NTHREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    bmm_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
