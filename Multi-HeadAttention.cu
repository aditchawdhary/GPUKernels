#include <cuda_runtime.h>
#include <math.h>

// Multi-head self-attention (no projections, matching the reference):
//   for each head z: out[:, z*dk:(z+1)*dk] = softmax( Q_z K_z^T / sqrt(dk) ) V_z
// Heads are processed in groups (blockIdx.z = head in group):
//   1) S_z = Q_z K_z^T * scale      batched tiled SGEMM (B transposed), strided head views
//   2) row softmax of S_z           in place
//   3) out_z = S_z V_z              batched tiled SGEMM, written straight into the head's columns

#define BM 64
#define BN 64
#define BK 16
#define GT 256   // 16 x 16 threads, 4 x 4 outputs each

// C[Mr, Nc] = scale * A[Mr, Kd] * op(B), all row-major with leading dimensions.
// Batch z offsets: A += z*strideA, B += z*strideB, C += z*strideC.
template <bool TRANS_B>
__global__ void __launch_bounds__(GT)
sgemm_batched(const float* __restrict__ A, int lda, size_t strideA,
              const float* __restrict__ B, int ldb, size_t strideB,
              float* __restrict__ C, int ldc, size_t strideC,
              int Mr, int Nc, int Kd, float scale) {
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];

    const int z = blockIdx.z;
    A += z * strideA;
    B += z * strideB;
    C += z * strideC;

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
        #pragma unroll
        for (int l = 0; l < (BM * BK) / GT; ++l) {
            int e = tid + l * GT;
            int r = e / BK, c = e % BK;                // coalesced along K
            int gr = m0 + r, gk = k0 + c;
            As[c][r] = (gr < Mr && gk < Kd) ? __ldg(A + (size_t)gr * lda + gk) : 0.f;
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / GT; ++l) {
            int e = tid + l * GT;
            if (TRANS_B) {                             // B is [Nc, Kd]
                int c = e / BK, k = e % BK;
                int gc = n0 + c, gk = k0 + k;
                Bs[k][c] = (gc < Nc && gk < Kd) ? __ldg(B + (size_t)gc * ldb + gk) : 0.f;
            } else {                                   // B is [Kd, Nc]
                int k = e / BN, c = e % BN;
                int gc = n0 + c, gk = k0 + k;
                Bs[k][c] = (gc < Nc && gk < Kd) ? __ldg(B + (size_t)gk * ldb + gc) : 0.f;
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
            if (gc < Nc) C[(size_t)gr * ldc + gc] = acc[i][j] * scale;
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
    __syncthreads();
    return v;
}

// One block per row of length N (rows of all heads in the group are contiguous).
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
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {
    const int dk = d_model / h;
    const size_t NN = (size_t)N * N;

    // Score buffer: as many heads per pass as fit in ~256 MB (always at least one head).
    const size_t budget = (size_t)64 << 20;            // floats (= 256 MB)
    int hb = (int)(budget / NN);
    if (hb < 1) hb = 1;
    if (hb > h) hb = h;

    static float* S = nullptr;
    static size_t cap = 0;
    size_t need = (size_t)hb * NN;
    if (need > cap) {
        if (S) cudaFree(S);
        cudaMalloc(&S, need * sizeof(float));
        cap = need;
    }

    const float scale = 1.0f / sqrtf((float)dk);

    for (int h0 = 0; h0 < h; h0 += hb) {
        const int nh = (h - h0 < hb) ? (h - h0) : hb;
        const float* Qh = Q + (size_t)h0 * dk;
        const float* Kh = K + (size_t)h0 * dk;
        const float* Vh = V + (size_t)h0 * dk;
        float* Oh = output + (size_t)h0 * dk;

        dim3 g1((N + BN - 1) / BN, (N + BM - 1) / BM, nh);
        sgemm_batched<true><<<g1, GT>>>(Qh, d_model, dk, Kh, d_model, dk,
                                        S, N, NN, N, N, dk, scale);

        softmax_rows<<<nh * N, ST>>>(S, N);

        dim3 g3((dk + BN - 1) / BN, (N + BM - 1) / BM, nh);
        sgemm_batched<false><<<g3, GT>>>(S, N, NN, Vh, d_model, dk,
                                         Oh, d_model, dk, N, dk, N, 1.0f);
    }
    cudaDeviceSynchronize();
}
