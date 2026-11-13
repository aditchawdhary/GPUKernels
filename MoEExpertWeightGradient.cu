#include <cuda_runtime.h>

// dW[g] = X[s:e]^T * dY[s:e]   (K x N per expert), reduction over the ragged token dim.
//
// Work decomposition:
//   * Output tiles of BK x BN (rows of dW = K dim, cols = N dim).
//   * The token range of every expert is cut into "chunks" of <= CHUNK rows, so a heavily
//     loaded expert is spread over many blocks instead of creating a long tail.
//     An empty expert still gets one chunk, which writes zeros.
//   * Experts with a single chunk store results directly. Experts with several chunks
//     are zeroed by a tiny pre-pass and accumulated with atomicAdd.
//   * Everything is decided on the device from group_offsets: no host sync, no scratch memory.

#define BK 128          // output tile rows   (K dim)
#define BN 128          // output tile cols   (N dim)
#define BM 8            // reduction rows (tokens) per shared-memory stage
#define TM 8            // per-thread output rows
#define TN 8            // per-thread output cols
#define THREADS 256     // (BK/TM) * (BN/TN)
#define CHUNK 1024      // max tokens handled by one block

__device__ __forceinline__ int num_chunks(int rows) {
    return rows == 0 ? 1 : (rows + CHUNK - 1) / CHUNK;
}

// Map a linear chunk id to (group, chunk-in-group, chunks-in-group). Returns false if past the end.
__device__ __forceinline__ bool locate_chunk(const int* offs, int G, int cid,
                                             int& g, int& c, int& nch) {
    int base = 0;
    for (int i = 0; i < G; i++) {
        int n = num_chunks(offs[i + 1] - offs[i]);
        if (cid < base + n) { g = i; c = cid - base; nch = n; return true; }
        base += n;
    }
    return false;
}

// Zero dW[g] only for experts that will be accumulated with atomics.
__global__ void zero_split_groups(const int* offs, float* dW, int G, size_t KN) {
    int g = blockIdx.y;
    if (num_chunks(offs[g + 1] - offs[g]) <= 1) return;
    float* out = dW + (size_t)g * KN;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < KN;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = 0.0f;
}

__global__ void __launch_bounds__(THREADS)
moe_wgrad_kernel(const float* __restrict__ X, const float* __restrict__ dY,
                 const int* __restrict__ offs, float* __restrict__ dW,
                 int G, int K, int N) {
    int g, c, nch;
    if (!locate_chunk(offs, G, blockIdx.z, g, c, nch)) return;   // uniform per block

    const int s0   = offs[g];
    const int rows = offs[g + 1] - s0;
    // balanced split of [s0, s0+rows) into nch pieces
    const int m_beg = s0 + (int)((long long)rows * c / nch);
    const int m_end = s0 + (int)((long long)rows * (c + 1) / nch);

    const int k0 = blockIdx.y * BK;
    const int n0 = blockIdx.x * BN;

    __shared__ __align__(16) float As[BM][BK];   // X  slice: tokens x K
    __shared__ __align__(16) float Bs[BM][BN];   // dY slice: tokens x N

    const int tid = threadIdx.x;
    const int tx  = tid % (BN / TN);   // 0..15 -> output columns
    const int ty  = tid / (BN / TN);   // 0..15 -> output rows

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    // Load mapping: each stage loads BM*BK = 1024 floats of X and 1024 of dY;
    // 256 threads x 4 elements, consecutive threads -> consecutive columns (coalesced).
    const int lcol = tid % BK;          // 0..127
    const int lrow = tid / BK;          // 0..1
    float ra[4], rb[4];

    auto load_regs = [&](int m) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int mm = m + lrow + r * 2;
            bool okm = mm < m_end;
            int kk = k0 + lcol, nn = n0 + lcol;
            ra[r] = (okm && kk < K) ? __ldg(&X [(size_t)mm * K + kk]) : 0.0f;
            rb[r] = (okm && nn < N) ? __ldg(&dY[(size_t)mm * N + nn]) : 0.0f;
        }
    };

    if (m_beg < m_end) load_regs(m_beg);

    for (int m = m_beg; m < m_end; m += BM) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
            As[lrow + r * 2][lcol] = ra[r];
            Bs[lrow + r * 2][lcol] = rb[r];
        }
        __syncthreads();

        if (m + BM < m_end) load_regs(m + BM);   // prefetch next stage while computing

#pragma unroll
        for (int t = 0; t < BM; t++) {
            float a[TM], b[TN];
            float4 a0 = *reinterpret_cast<const float4*>(&As[t][ty * TM]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[t][ty * TM + 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[t][tx * TN]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[t][tx * TN + 4]);
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
            a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
            b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    float* out = dW + (size_t)g * K * N;
    const bool atomic = nch > 1;
#pragma unroll
    for (int i = 0; i < TM; i++) {
        int k = k0 + ty * TM + i;
        if (k >= K) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            int n = n0 + tx * TN + j;
            if (n >= N) continue;
            if (atomic) atomicAdd(&out[(size_t)k * N + n], acc[i][j]);
            else        out[(size_t)k * N + n] = acc[i][j];   // also writes zeros for empty experts
        }
    }
}

// X, dY, group_offsets, dW are device pointers
extern "C" void solve(const float* X, const float* dY, const int* group_offsets, float* dW, int G,
                      int M_total, int K, int N) {
    size_t KN = (size_t)K * N;

    dim3 zgrid(64, G);
    zero_split_groups<<<zgrid, 256>>>(group_offsets, dW, G, KN);

    // Upper bound on total chunks: sum ceil(rows/CHUNK) (+1 for empties) <= M_total/CHUNK + G
    int max_chunks = (M_total + CHUNK - 1) / CHUNK + G;
    dim3 grid((N + BN - 1) / BN, (K + BK - 1) / BK, max_chunks);
    moe_wgrad_kernel<<<grid, THREADS>>>(X, dY, group_offsets, dW, G, K, N);
}
