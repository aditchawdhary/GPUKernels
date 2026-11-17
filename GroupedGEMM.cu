/*
Grouped GEMM
A: (M_total, K); B: (G, K, N); C: (M_total, N); group_offsets: (G + 1), all row-major.
For each group g with s = off[g], e = off[g+1]:  C[s:e] = A[s:e] x B[g].
Empty groups write nothing.

Constraints
 - G ≤ 64, M_total ≤ 16,384, K, N ≤ 4096; values in [-1, 1]; float32
 - Performance is measured with G = 8, M_total = 8192, K = 1024, N = 2048
*/

/*
Approach: an FP32 register-tiled SGEMM with a device-side tile schedule, so a
single launch covers every group and no tile ever mixes two groups' weights.

 - Schedule: group g gets ceil(rows_g / 128) row tiles. Each block reads the
   G + 1 offsets, builds the tile prefix in shared memory, and maps its
   blockIdx.y to (group, first row). The grid is launched with the upper bound
   ceil(M_total / 128) + G row tiles, and surplus blocks exit at once. The
   offsets stay on the device, so the host never synchronizes to read them.
   Rows past a group's end are zero-filled on load and never stored.
 - Tile: 128 x 128 x 8, 256 threads. Each thread computes an 8 x 8 block, split
   as rows {4ty..4ty+3, 64+4ty..} x cols {4tx..4tx+3, 64+4tx..}, so the float4
   shared-memory reads are conflict-free. A is stored transposed (As[k][m]) and
   B row-major (Bs[k][n]). Shared memory is double-buffered, and the next tile is
   prefetched into registers while the current one is computed (16 KB smem).
   Per k-step a thread does 4 float4 shared-memory reads for 64 FMAs.
 - VEC path (K % 4 == 0 and N % 4 == 0, which covers the benchmark): float4
   global loads and stores. Otherwise the same kernel uses guarded scalar
   accesses, so any shape is handled.
At the benchmark: 16 column tiles x 64 row tiles = 1024 working blocks,
34 GFLOP of FP32.
*/

#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 8
#define THREADS 256
#define MAX_G 64

template <bool VEC>
__device__ __forceinline__ float4 load4(const float* p, int col, int limit, bool rowOk)
{
    // loads p[col .. col+3] (columns < limit), zeros elsewhere
    if (!rowOk) return make_float4(0.f, 0.f, 0.f, 0.f);
    if (VEC) {
        if (col < limit) return __ldg(reinterpret_cast<const float4*>(p + col));   // limit % 4 == 0
        return make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 v;
    v.x = (col     < limit) ? __ldg(p + col)     : 0.f;
    v.y = (col + 1 < limit) ? __ldg(p + col + 1) : 0.f;
    v.z = (col + 2 < limit) ? __ldg(p + col + 2) : 0.f;
    v.w = (col + 3 < limit) ? __ldg(p + col + 3) : 0.f;
    return v;
}

template <bool VEC>
__global__ void __launch_bounds__(THREADS)
grouped_sgemm(const float* __restrict__ A, const float* __restrict__ B,
              const int* __restrict__ offs, float* __restrict__ C,
              int G, int K, int N)
{
    __shared__ __align__(16) float As[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN];
    __shared__ int tileStart[MAX_G + 1];
    __shared__ int sInfo[3];                       // group, row start, row end

    const int tid = threadIdx.x;

    // ---- map blockIdx.y -> (group, row range)
    if (tid == 0) {
        int acc = 0;
        for (int g = 0; g < G; ++g) {
            tileStart[g] = acc;
            const int rows = offs[g + 1] - offs[g];
            acc += (rows + BM - 1) / BM;
        }
        tileStart[G] = acc;
        const int t = blockIdx.y;
        int grp = -1;
        if (t < acc) {
            grp = 0;
            while (tileStart[grp + 1] <= t) ++grp;
        }
        sInfo[0] = grp;
        if (grp >= 0) {
            sInfo[1] = offs[grp] + (t - tileStart[grp]) * BM;
            sInfo[2] = offs[grp + 1];
        }
    }
    __syncthreads();
    const int grp = sInfo[0];
    if (grp < 0) return;                           // surplus block (uniform)
    const int r0 = sInfo[1], rEnd = sInfo[2];
    const int c0 = blockIdx.x * BN;
    const float* Bg = B + (size_t)grp * K * N;

    // ---- load assignments
    const int aRow = tid >> 1, aK = (tid & 1) * 4;          // A tile: 128 rows x 8 k
    const int bK = tid >> 5,  bCol = (tid & 31) * 4;        // B tile: 8 k x 128 cols
    const bool aRowOk = (r0 + aRow) < rEnd;
    const float* aPtr = A + (size_t)(r0 + aRow) * K;

    const int tx = tid & 15, ty = tid >> 4;

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

    const int KT = (K + BK - 1) / BK;

    // first tile
    float4 ra = load4<VEC>(aPtr, aK, K, aRowOk);
    float4 rb = load4<VEC>(Bg + (size_t)bK * N, c0 + bCol, N, bK < K);
    As[0][aK + 0][aRow] = ra.x; As[0][aK + 1][aRow] = ra.y;
    As[0][aK + 2][aRow] = ra.z; As[0][aK + 3][aRow] = ra.w;
    *reinterpret_cast<float4*>(&Bs[0][bK][bCol]) = rb;
    __syncthreads();

    for (int kt = 0; kt < KT; ++kt) {
        const int cur = kt & 1;
        const bool more = kt + 1 < KT;
        if (more) {                                // prefetch next tile into registers
            const int k0 = (kt + 1) * BK;
            ra = load4<VEC>(aPtr, k0 + aK, K, aRowOk);
            rb = load4<VEC>(Bg + (size_t)(k0 + bK) * N, c0 + bCol, N, (k0 + bK) < K);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[cur][kk][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[cur][kk][64 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[cur][kk][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[cur][kk][64 + tx * 4]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }

        if (more) {                                // other buffer: no one is reading it
            const int nxt = cur ^ 1;
            As[nxt][aK + 0][aRow] = ra.x; As[nxt][aK + 1][aRow] = ra.y;
            As[nxt][aK + 2][aRow] = ra.z; As[nxt][aK + 3][aRow] = ra.w;
            *reinterpret_cast<float4*>(&Bs[nxt][bK][bCol]) = rb;
        }
        __syncthreads();
    }

    // ---- store
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = r0 + ((i < 4) ? (ty * 4 + i) : (64 + ty * 4 + i - 4));
        if (r >= rEnd) continue;
        float* crow = C + (size_t)r * N;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = c0 + h * 64 + tx * 4;
            const float* v = &acc[i][h * 4];
            if (VEC) {
                if (c < N) *reinterpret_cast<float4*>(crow + c) = make_float4(v[0], v[1], v[2], v[3]);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) if (c + j < N) crow[c + j] = v[j];
            }
        }
    }
}

// A, B, group_offsets, C are device pointers
extern "C" void solve(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N)
{
    if (G <= 0 || M_total <= 0 || K <= 0 || N <= 0) return;
    const int maxRowTiles = (M_total + BM - 1) / BM + G;    // ≥ sum_g ceil(rows_g / BM)
    dim3 grid((N + BN - 1) / BN, maxRowTiles);
    if ((K % 4 == 0) && (N % 4 == 0))
        grouped_sgemm<true><<<grid, THREADS>>>(A, B, group_offsets, C, G, K, N);
    else
        grouped_sgemm<false><<<grid, THREADS>>>(A, B, group_offsets, C, G, K, N);
    cudaDeviceSynchronize();
}
