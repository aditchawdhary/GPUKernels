#include <cuda_runtime.h>
#include <math.h>

// Linear self-attention (Katharopoulos et al., 2020), non-causal:
//   phi(x)  = elu(x) + 1 = (x > 0) ? x + 1 : exp(x)
//   S       = sum_j phi(K_j)^T V_j          (d x d)
//   z       = sum_j phi(K_j)                (d)
//   out_i   = phi(Q_i) S / (phi(Q_i) . z)
//
// O(M d^2) instead of O(M^2 d):
//   K1: blocks split the M rows; each accumulates a partial (S, z) in registers.
//   K2: reduce the per-block partials (in double, fixed order -> deterministic).
//   K3: per query row, out = phi(Q_i) S / (phi(Q_i) . z).
//
// Inputs reach |x| = 100, so phi(Q) entries can be as small as e^-100 (below
// FP32's normal range). The output is invariant to scaling phi(Q_i), so K3
// rescales each row by its largest entry, computed in log space:
//   log phi(x) = (x > 0) ? log1p(x) : x,   phi'(x) = exp(log phi(x) - rowmax).

#define DMAX 128
#define K1_THREADS 256
#define K1_ROWS 32                                   // rows per smem tile
#define K1_EPT ((DMAX * DMAX) / K1_THREADS)          // S entries per thread = 64
#define K3_ROWS 16

__device__ __forceinline__ float phi(float x) { return x > 0.f ? x + 1.f : expf(x); }

// ---------------- K1: partial S and z over a chunk of rows ----------------
__global__ void __launch_bounds__(K1_THREADS)
kv_partial_kernel(const float* __restrict__ K, const float* __restrict__ V,
                  float* __restrict__ partial, int M, int d, int rowsPerBlock)
{
    __shared__ float sK[K1_ROWS][DMAX];
    __shared__ float sV[K1_ROWS][DMAX];

    const int tid = threadIdx.x;
    const int dd  = d * d;
    const int r0  = blockIdx.x * rowsPerBlock;
    const int r1  = min(M, r0 + rowsPerBlock);

    float acc[K1_EPT];
#pragma unroll
    for (int k = 0; k < K1_EPT; ++k) acc[k] = 0.f;
    float zacc = 0.f;

    for (int t0 = r0; t0 < r1; t0 += K1_ROWS) {
        const int nr = min(K1_ROWS, r1 - t0);
        __syncthreads();
        for (int idx = tid; idx < K1_ROWS * DMAX; idx += K1_THREADS) {
            const int r = idx / DMAX, c = idx % DMAX;
            const bool ok = (r < nr) && (c < d);
            const size_t g = (size_t)(t0 + r) * d + c;
            sK[r][c] = ok ? phi(__ldg(K + g)) : 0.f;
            sV[r][c] = ok ? __ldg(V + g) : 0.f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < K1_EPT; ++k) {
            const int e = tid + k * K1_THREADS;
            if (e < dd) {                            // uniform across the loop body
                const int a = e / d, c = e - a * d;
                float s = acc[k];
                for (int r = 0; r < nr; ++r) s = fmaf(sK[r][a], sV[r][c], s);
                acc[k] = s;
            }
        }
        if (tid < d)
            for (int r = 0; r < nr; ++r) zacc += sK[r][tid];
    }

    float* p = partial + (size_t)blockIdx.x * (dd + d);
#pragma unroll
    for (int k = 0; k < K1_EPT; ++k) {
        const int e = tid + k * K1_THREADS;
        if (e < dd) p[e] = acc[k];
    }
    if (tid < d) p[dd + tid] = zacc;
}

// ---------------- K2: reduce partials -> S (dd) and z (d) ----------------
__global__ void reduce_kernel(const float* __restrict__ partial, float* __restrict__ Sz,
                              int n, int numPartials)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    double s = 0.0;
    for (int b = 0; b < numPartials; ++b) s += (double)partial[(size_t)b * n + e];
    Sz[e] = (float)s;
}

// ---------------- K3: out_i = phi(Q_i) S / (phi(Q_i) . z) ----------------
// Block = DMAX threads (thread c owns output column c), K3_ROWS query rows.
__global__ void __launch_bounds__(DMAX)
output_kernel(const float* __restrict__ Q, const float* __restrict__ Sz,
              float* __restrict__ out, int M, int d)
{
    __shared__ float sQ[K3_ROWS][DMAX];
    __shared__ float sDen[K3_ROWS];

    const int tid  = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;      // 4 warps
    const int row0 = blockIdx.x * K3_ROWS;
    const float* S = Sz;
    const float* z = Sz + d * d;

    // Load log phi(Q) for the block's rows.
    for (int idx = tid; idx < K3_ROWS * DMAX; idx += DMAX) {
        const int r = idx / DMAX, c = idx % DMAX;
        const int gr = row0 + r;
        float v = -INFINITY;
        if (gr < M && c < d) {
            const float x = __ldg(Q + (size_t)gr * d + c);
            v = x > 0.f ? log1pf(x) : x;
        }
        sQ[r][c] = v;
    }
    __syncthreads();

    // Each warp normalizes K3_ROWS/4 rows: phi' = exp(logphi - rowmax), den = phi' . z
    for (int r = warp; r < K3_ROWS; r += 4) {
        float mx = -INFINITY;
        for (int c = lane; c < d; c += 32) mx = fmaxf(mx, sQ[r][c]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
        float den = 0.f;
        for (int c = lane; c < DMAX; c += 32) {
            const float w = (c < d && mx != -INFINITY) ? expf(sQ[r][c] - mx) : 0.f;
            sQ[r][c] = w;
            if (c < d) den = fmaf(w, __ldg(z + c), den);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) den += __shfl_xor_sync(0xffffffff, den, off);
        if (lane == 0) sDen[r] = den;
    }
    __syncthreads();

    if (tid >= d) return;
    float acc[K3_ROWS];
#pragma unroll
    for (int r = 0; r < K3_ROWS; ++r) acc[r] = 0.f;
    for (int a = 0; a < d; ++a) {
        const float s = __ldg(S + a * d + tid);      // coalesced across threads, L2-resident
#pragma unroll
        for (int r = 0; r < K3_ROWS; ++r) acc[r] = fmaf(sQ[r][a], s, acc[r]);
    }
#pragma unroll
    for (int r = 0; r < K3_ROWS; ++r) {
        const int gr = row0 + r;
        if (gr < M) out[(size_t)gr * d + tid] = acc[r] / sDen[r];
    }
}

// ---------------- host ----------------
static float* g_partial = nullptr;
static size_t g_partialCap = 0;
static float* g_Sz = nullptr;

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    const int n = d * d + d;
    int numPartials = sms > 0 ? sms : 108;
    int rowsPerBlock = (M + numPartials - 1) / numPartials;
    numPartials = (M + rowsPerBlock - 1) / rowsPerBlock;   // no empty blocks

    const size_t need = (size_t)numPartials * n;
    if (need > g_partialCap) {
        if (g_partial) cudaFree(g_partial);
        cudaMalloc(&g_partial, need * sizeof(float));
        g_partialCap = need;
    }
    if (!g_Sz) cudaMalloc(&g_Sz, (DMAX * DMAX + DMAX) * sizeof(float));

    kv_partial_kernel<<<numPartials, K1_THREADS>>>(K, V, g_partial, M, d, rowsPerBlock);
    reduce_kernel<<<(n + 255) / 256, 256>>>(g_partial, g_Sz, n, numPartials);
    output_kernel<<<(M + K3_ROWS - 1) / K3_ROWS, DMAX>>>(Q, g_Sz, output, M, d);
    cudaDeviceSynchronize();
}
