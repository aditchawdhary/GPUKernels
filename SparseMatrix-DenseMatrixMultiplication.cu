#include <cuda_runtime.h>

// C (M x K) = A (M x N) * B (N x K), row-major fp32. A is stored dense and ~35% of its entries
// are non-zero, scattered uniformly at random.
//
// Why a dense GEMM: with random ~35% density, essentially every tile of A contains non-zeros,
// so tile skipping finds nothing to skip, and a CSR SpMM would re-read a full row of B
// (K floats) from cache for every non-zero (~6 GB of L2 traffic at the benchmark size). A
// register-blocked dense SGEMM does the 8.6 GFLOP from shared memory and wins comfortably.
//
// Kernel: 128 x 128 output tile per block, 256 threads, 8 x 8 outputs per thread (split into
// four 4x4 quadrants to avoid shared-memory bank conflicts), BK = 8, double-buffered shared
// memory with register prefetch (one __syncthreads per k-step). float4 global loads/stores
// when N and K are multiples of 4 and pointers are 16-byte aligned; guarded scalar otherwise.

#define BM 128
#define BN 128
#define BK 8
#define THREADS 256

template <bool VEC>
__global__ void __launch_bounds__(THREADS)
sgemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
             int M, int N, int K) {
    __shared__ __align__(16) float As[2][BK][BM];   // A tile, transposed (k-major)
    __shared__ __align__(16) float Bs[2][BK][BN];

    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;                  // column offset in C / B

    // load mapping: A tile 128 rows x 8 k  -> thread loads row a_r, k a_k..a_k+3
    //               B tile 8 k x 128 cols  -> thread loads k b_r, cols b_c..b_c+3
    const int a_r = tid >> 1, a_k = (tid & 1) * 4;
    const int b_r = tid >> 5, b_c = (tid & 31) * 4;

    float ra[4], rb[4];
    auto load = [&](int k0) {
        const int gr = m0 + a_r, gk = k0 + a_k;
        if (VEC) {
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < M && gk < N) v = __ldg(reinterpret_cast<const float4*>(&A[(size_t)gr * N + gk]));
            ra[0] = v.x; ra[1] = v.y; ra[2] = v.z; ra[3] = v.w;
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++)
                ra[i] = (gr < M && gk + i < N) ? __ldg(&A[(size_t)gr * N + gk + i]) : 0.0f;
        }
        const int bk = k0 + b_r, bc = n0 + b_c;
        if (VEC) {
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (bk < N && bc < K) v = __ldg(reinterpret_cast<const float4*>(&B[(size_t)bk * K + bc]));
            rb[0] = v.x; rb[1] = v.y; rb[2] = v.z; rb[3] = v.w;
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++)
                rb[i] = (bk < N && bc + i < K) ? __ldg(&B[(size_t)bk * K + bc + i]) : 0.0f;
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 4; i++) As[buf][a_k + i][a_r] = ra[i];
        *reinterpret_cast<float4*>(&Bs[buf][b_r][b_c]) = make_float4(rb[0], rb[1], rb[2], rb[3]);
    };

    // compute mapping: rows {ty*4 + i, 64 + ty*4 + i}, cols {tx*4 + j, 64 + tx*4 + j}
    const int tx = tid & 15, ty = tid >> 4;
    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    load(0);
    store(0);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < N; k0 += BK) {
        const bool more = k0 + BK < N;
        if (more) load(k0 + BK);

#pragma unroll
        for (int k = 0; k < BK; k++) {
            float4 a0 = *reinterpret_cast<const float4*>(&As[buf][k][ty * 4]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[buf][k][64 + ty * 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[buf][k][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[buf][k][64 + tx * 4]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }

        if (more) store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }

    // epilogue
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int row = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
        if (row >= M) continue;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int col = n0 + h * 64 + tx * 4;
            float* dst = &C[(size_t)row * K + col];
            if (VEC) {
                if (col < K)
                    *reinterpret_cast<float4*>(dst) = make_float4(acc[i][h * 4 + 0], acc[i][h * 4 + 1],
                                                                  acc[i][h * 4 + 2], acc[i][h * 4 + 3]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; j++)
                    if (col + j < K) dst[j] = acc[i][h * 4 + j];
            }
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz) {
    dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
    size_t addr = reinterpret_cast<size_t>(A) | reinterpret_cast<size_t>(B) | reinterpret_cast<size_t>(C);
    bool vec = (N % 4 == 0) && (K % 4 == 0) && ((addr & 15) == 0);
    if (vec) sgemm_kernel<true><<<grid, THREADS>>>(A, B, C, M, N, K);
    else     sgemm_kernel<false><<<grid, THREADS>>>(A, B, C, M, N, K);
}
