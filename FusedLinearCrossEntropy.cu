#include <cuda_runtime.h>
#include <float.h>

// Fused linear + cross-entropy.
//
//   loss = mean_i [ logsumexp_v (X_i . W_v) - X_i . W_{t_i} ]
//
// Kernel 1 (fused GEMM + partial LSE): each block computes a BM x BN tile of
//   logits in registers (never written to global memory), then reduces each
//   row of the tile to a (max, sum exp(z - max)) pair and stores that pair.
//   Only N * ceil(V/BN) pairs hit memory (~8 MB at the benchmark size,
//   compared with 512 MB for the full logits matrix).
// Kernel 2: one warp per row merges the per-tile pairs (online LSE merge),
//   computes the target logit with a dot product, and writes the row loss.
// Kernel 3: a single block sums the row losses (fixed order, deterministic).

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define GEMM_THREADS 256   // (BM/TM) * (BN/TN)

// ---------------------------------------------------------------------------
// Tile loader: loads a 128 x 8 slice (rows [row0, row0+128), cols [k0, k0+8))
// of a row-major matrix with leading dimension D into smem transposed as
// dst[k][row]. Each of the 256 threads loads 4 consecutive k values.
// ---------------------------------------------------------------------------
template <bool VEC>
__device__ __forceinline__ void load_tile(const float* __restrict__ src, int rows, int D,
                                          int row0, int k0, float (*dst)[BM + 4], int tid)
{
    const int r  = tid >> 1;          // 0..127
    const int kk = (tid & 1) * 4;     // 0 or 4
    const int gr = row0 + r;
    const int gk = k0 + kk;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
    if (gr < rows) {
        const float* p = src + (size_t)gr * D + gk;
        if (VEC && gk + 3 < D) {
            float4 t = __ldg(reinterpret_cast<const float4*>(p));
            v0 = t.x; v1 = t.y; v2 = t.z; v3 = t.w;
        } else {
            if (gk     < D) v0 = __ldg(p);
            if (gk + 1 < D) v1 = __ldg(p + 1);
            if (gk + 2 < D) v2 = __ldg(p + 2);
            if (gk + 3 < D) v3 = __ldg(p + 3);
        }
    }
    dst[kk    ][r] = v0;
    dst[kk + 1][r] = v1;
    dst[kk + 2][r] = v2;
    dst[kk + 3][r] = v3;
}

// ---------------------------------------------------------------------------
// Kernel 1: fused logits tile + per-row partial (max, sumexp)
// Thread (ty, tx) in a 16x16 grid owns rows {ty*4+i, 64+ty*4+i} and
// cols {tx*4+j, 64+tx*4+j}, i,j in 0..3 (bank-conflict-free float4 smem reads).
// ---------------------------------------------------------------------------
template <bool VEC>
__global__ void __launch_bounds__(GEMM_THREADS, 2)
flce_tile_kernel(const float* __restrict__ X, const float* __restrict__ W,
                 float2* __restrict__ partials, int N, int D, int V, int numVT)
{
    __shared__ __align__(16) float As[2][BK][BM + 4];
    __shared__ __align__(16) float Bs[2][BK][BN + 4];

    const int tid = threadIdx.x;
    const int tx  = tid & 15;
    const int ty  = tid >> 4;
    const int row0 = blockIdx.x * BM;   // rows of X (consecutive blocks share a W tile -> L2 reuse)
    const int col0 = blockIdx.y * BN;   // vocab tile

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int numK = (D + BK - 1) / BK;

    load_tile<VEC>(X, N, D, row0, 0, As[0], tid);
    load_tile<VEC>(W, V, D, col0, 0, Bs[0], tid);
    __syncthreads();

    for (int kt = 0; kt < numK; ++kt) {
        const int cur = kt & 1;
        // Prefetch next K slice into the other buffer while computing this one.
        if (kt + 1 < numK) {
            load_tile<VEC>(X, N, D, row0, (kt + 1) * BK, As[cur ^ 1], tid);
            load_tile<VEC>(W, V, D, col0, (kt + 1) * BK, Bs[cur ^ 1], tid);
        }
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            float4 a0 = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[cur][k][64 + ty * 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[cur][k][64 + tx * 4]);
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
            a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
            b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    // ---- Epilogue: per-row (max, sum exp) over this tile's BN columns ----
    bool colValid[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        int c = col0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + (j - 4));
        colValid[j] = c < V;
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        float m = -FLT_MAX;
#pragma unroll
        for (int j = 0; j < TN; ++j)
            if (colValid[j]) m = fmaxf(m, acc[i][j]);
        // The 16 threads sharing ty are 16 consecutive lanes (a half-warp).
#pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));

        float s = 0.f;
#pragma unroll
        for (int j = 0; j < TN; ++j)
            if (colValid[j]) s += __expf(acc[i][j] - m);
#pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            s += __shfl_xor_sync(0xffffffff, s, off);

        const int r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
        if (tx == 0 && r < N)
            partials[(size_t)r * numVT + blockIdx.y] = make_float2(m, s);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: merge partials per row, subtract target logit -> row loss
// ---------------------------------------------------------------------------
__global__ void flce_row_kernel(const float* __restrict__ X, const float* __restrict__ W,
                                const int* __restrict__ targets,
                                const float2* __restrict__ partials,
                                float* __restrict__ rowLoss, int N, int D, int numVT)
{
    const int lane = threadIdx.x & 31;
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= N) return;

    // Online merge of (max, sumexp) pairs.
    const float2* p = partials + (size_t)row * numVT;
    float m = -FLT_MAX, s = 0.f;
    for (int t = lane; t < numVT; t += 32) {
        float2 q = p[t];
        float mn = fmaxf(m, q.x);
        s = s * __expf(m - mn) + q.y * __expf(q.x - mn);
        m = mn;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float m2 = __shfl_xor_sync(0xffffffff, m, off);
        float s2 = __shfl_xor_sync(0xffffffff, s, off);
        float mn = fmaxf(m, m2);
        s = s * __expf(m - mn) + s2 * __expf(m2 - mn);
        m = mn;
    }

    // Target logit z_{i, t_i} = X_i . W_{t_i}
    const int t = targets[row];
    const float* x = X + (size_t)row * D;
    const float* w = W + (size_t)t * D;
    float z = 0.f;
    for (int k = lane; k < D; k += 32) z = fmaf(__ldg(x + k), __ldg(w + k), z);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) z += __shfl_xor_sync(0xffffffff, z, off);

    if (lane == 0) rowLoss[row] = m + logf(s) - z;
}

// ---------------------------------------------------------------------------
// Kernel 3: deterministic mean of row losses
// ---------------------------------------------------------------------------
__global__ void flce_mean_kernel(const float* __restrict__ rowLoss, float* __restrict__ loss, int N)
{
    __shared__ double sh[32];
    double v = 0.0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) v += (double)rowLoss[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < (int)(blockDim.x >> 5)) ? sh[lane] : 0.0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) loss[0] = (float)(v / (double)N);
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
static float2* g_partials = nullptr;
static size_t  g_partialsCap = 0;
static float*  g_rowLoss = nullptr;
static size_t  g_rowLossCap = 0;

// X, W, targets, loss are device pointers
extern "C" void solve(const float* X, const float* W, const int* targets, float* loss, int N, int D,
                      int V) {
    const int numVT = (V + BN - 1) / BN;
    const int numMT = (N + BM - 1) / BM;

    // Scratch buffers are cached across calls so repeated (benchmark) calls skip cudaMalloc.
    size_t needP = (size_t)N * numVT;
    if (needP > g_partialsCap) {
        if (g_partials) cudaFree(g_partials);
        cudaMalloc(&g_partials, needP * sizeof(float2));
        g_partialsCap = needP;
    }
    if ((size_t)N > g_rowLossCap) {
        if (g_rowLoss) cudaFree(g_rowLoss);
        cudaMalloc(&g_rowLoss, (size_t)N * sizeof(float));
        g_rowLossCap = N;
    }

    dim3 grid(numMT, numVT);
    const bool vec = (D % 4 == 0) &&
                     ((reinterpret_cast<size_t>(X) | reinterpret_cast<size_t>(W)) % 16 == 0);
    if (vec)
        flce_tile_kernel<true><<<grid, GEMM_THREADS>>>(X, W, g_partials, N, D, V, numVT);
    else
        flce_tile_kernel<false><<<grid, GEMM_THREADS>>>(X, W, g_partials, N, D, V, numVT);

    const int rowThreads = 256;
    const int rowBlocks  = (int)(((size_t)N * 32 + rowThreads - 1) / rowThreads);
    flce_row_kernel<<<rowBlocks, rowThreads>>>(X, W, targets, g_partials, g_rowLoss, N, D, numVT);

    flce_mean_kernel<<<1, 1024>>>(g_rowLoss, loss, N);
    cudaDeviceSynchronize();
}
