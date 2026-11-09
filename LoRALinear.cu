#include <cuda_runtime.h>

// output = x·Wᵀ + lora_scale · (x·Aᵀ)·Bᵀ
//   x: batch×d_in, W: d_out×d_in, A: rank×d_in, B: d_out×rank  (row-major)
//   1) T = lora_scale · x·Aᵀ   (batch×rank, split-K across the GPU)
//   2) output = x·Wᵀ + T·Bᵀ    (one fused kernel)

#define BM 64
#define BN 64
#define BK 16
#define NT 256

__device__ __forceinline__ void load_tile(const float* __restrict__ P, int ld, int rows,
                                          int r0, int k0, int kend,
                                          float (*S)[BM + 1], int tid) {
#pragma unroll
    for (int i = 0; i < (BM * BK) / NT; ++i) {
        const int idx = tid + i * NT;
        const int r = idx / BK, k = idx % BK;
        const int gr = r0 + r, gk = k0 + k;
        S[k][r] = (gr < rows && gk < kend) ? P[(size_t)gr * ld + gk] : 0.0f;
    }
}

__device__ __forceinline__ void accumulate(const float* __restrict__ X, const float* __restrict__ Y,
                                           int ld, int kbeg, int kend, int M, int N,
                                           int m0, int n0, float acc[4][4],
                                           float (*As)[BM + 1], float (*Bs)[BN + 1],
                                           int tid, int tx, int ty) {
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        load_tile(X, ld, M, m0, k0, kend, As, tid);
        load_tile(Y, ld, N, n0, k0, kend, Bs, tid);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(NT)
gemm_nt(const float* __restrict__ X1, const float* __restrict__ Y1, int K1, int k_chunk,
        const float* __restrict__ X2, const float* __restrict__ Y2, int K2,
        float* __restrict__ C, int M, int N, float alpha, int atomic) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];

    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    const int kb = blockIdx.z * k_chunk;
    const int ke = min(K1, kb + k_chunk);
    accumulate(X1, Y1, K1, kb, ke, M, N, m0, n0, acc, As, Bs, tid, tx, ty);

    if (alpha != 1.0f) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] *= alpha;
    }

    if (X2 != nullptr)
        accumulate(X2, Y2, K2, 0, K2, M, N, m0, n0, acc, As, Bs, tid, tx, ty);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = m0 + ty + 16 * i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = n0 + tx + 16 * j;
            if (c >= N) continue;
            if (atomic) atomicAdd(&C[(size_t)r * N + c], acc[i][j]);
            else        C[(size_t)r * N + c] = acc[i][j];
        }
    }
}

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      float* output, int batch, int d_in, int d_out, int rank,
                      float lora_scale) {
    float* T = nullptr;
    cudaMalloc(&T, sizeof(float) * batch * rank);
    cudaMemset(T, 0, sizeof(float) * batch * rank);

    int splits = d_in / 256;
    if (splits < 1) splits = 1;
    if (splits > 32) splits = 32;
    int k_chunk = (d_in + splits - 1) / splits;
    k_chunk = ((k_chunk + BK - 1) / BK) * BK;
    splits = (d_in + k_chunk - 1) / k_chunk;

    dim3 g1((rank + BN - 1) / BN, (batch + BM - 1) / BM, splits);
    gemm_nt<<<g1, NT>>>(x, A, d_in, k_chunk, nullptr, nullptr, 0,
                        T, batch, rank, lora_scale, 1);

    dim3 g2((d_out + BN - 1) / BN, (batch + BM - 1) / BM, 1);
    gemm_nt<<<g2, NT>>>(x, W, d_in, d_in, T, B, rank,
                        output, batch, d_out, 1.0f, 0);

    cudaDeviceSynchronize();
    cudaFree(T);
}
