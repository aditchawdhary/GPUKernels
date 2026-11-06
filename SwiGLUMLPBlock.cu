#include <cuda_runtime.h>
#include <math.h>

#define BM  64     // rows of C per block
#define BN  64     // cols of C per block
#define BK  16     // K-slice loaded per step
#define TPB 256    // threads per block; each computes a 4x4 patch

__device__ __forceinline__ float silu(float z) { return z / (1.0f + expf(-z)); }

// GLU = true :  C = silu(A @ B1) * (A @ B2)
// GLU = false:  C = A @ B1
// A is M x K, B1/B2 are K x N, C is M x N, all row-major.
template <bool GLU>
__global__ void __launch_bounds__(TPB)
gemm_kernel(const float* __restrict__ A, const float* __restrict__ B1,
            const float* __restrict__ B2, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BK][BM + 1];          // A tile, stored transposed (+1 avoids bank conflicts)
    __shared__ float B1s[BK][BN];
    __shared__ float B2s[GLU ? BK : 1][BN];

    int tid = threadIdx.x;
    int tx  = tid % 16, ty = tid / 16;        // this thread owns rows ty+16i, cols tx+16j
    int bm  = blockIdx.y * BM, bn = blockIdx.x * BN;

    float acc1[4][4] = {}, acc2[4][4] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load of the A tile (64 x 16): 4 elements per thread.
        for (int r = 0; r < 4; r++) {
            int e = tid + r * TPB, row = e / BK, col = e % BK;
            int gr = bm + row, gc = k0 + col;
            As[col][row] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
        }
        // Cooperative load of the B tile(s) (16 x 64): 4 elements per thread.
        for (int r = 0; r < 4; r++) {
            int e = tid + r * TPB, row = e / BN, col = e % BN;
            int gr = k0 + row, gc = bn + col;
            bool ok = gr < K && gc < N;
            B1s[row][col] = ok ? B1[(size_t)gr * N + gc] : 0.0f;
            if (GLU) B2s[row][col] = ok ? B2[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();

        // Multiply the tiles: each thread updates its 4x4 patch.
#pragma unroll
        for (int k = 0; k < BK; k++) {
            float a[4], b1[4], b2[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[k][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                b1[j] = B1s[k][tx + 16 * j];
                if (GLU) b2[j] = B2s[k][tx + 16 * j];
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc1[i][j] += a[i] * b1[j];
                    if (GLU) acc2[i][j] += a[i] * b2[j];
                }
        }
        __syncthreads();
    }

    // Write results (activation fused in for the GLU pass).
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = bm + ty + 16 * i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int c = bn + tx + 16 * j;
            if (c >= N) continue;
            C[(size_t)r * N + c] = GLU ? silu(acc1[i][j]) * acc2[i][j] : acc1[i][j];
        }
    }
}

// x, W_gate, W_up, W_down, output are device pointers
extern "C" void solve(const float* x, const float* W_gate, const float* W_up,
                      const float* W_down, float* output, int M, int d_model, int d_ffn) {
    // Scratch buffer for H (M x d_ffn), grown only when needed.
    static float* H = nullptr;
    static size_t capacity = 0;
    size_t need = (size_t)M * d_ffn;
    if (need > capacity) {
        if (H) cudaFree(H);
        cudaMalloc(&H, need * sizeof(float));
        capacity = need;
    }

    dim3 block(TPB);

    // 1. H = silu(x @ W_gate) * (x @ W_up)
    dim3 grid1((d_ffn + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<true><<<grid1, block>>>(x, W_gate, W_up, H, M, d_ffn, d_model);

    // 2. output = H @ W_down
    dim3 grid2((d_model + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<false><<<grid2, block>>>(H, W_down, nullptr, output, M, d_model, d_ffn);
}
