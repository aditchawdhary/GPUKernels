#include <cuda_runtime.h>
#include <math.h>

// All-pairs shortest paths: blocked Floyd-Warshall (Venkataraman et al.).
//
// The matrix is split into B x B tiles (B = 64). For each tile-round kb the
// k-loop for k in [kb*B, kb*B + B) runs in three phases:
//   Phase 1: the diagonal tile (kb, kb) runs plain FW over its 64 k's in smem.
//   Phase 2: tiles in tile-row kb and tile-column kb, each in smem, using the
//            finished diagonal tile (still sequential in k).
//   Phase 3: every other tile (i, j): D[i][j] = min_k D[i][k] + D[k][j], where
//            D[i][k] is in column-tile kb and D[k][j] in row-tile kb. Those are
//            final for this round, so this is a min-plus tile product with no
//            sequential dependence. Phase 3 is ~all of the work: 256 threads,
//            each owning a 4 x 4 sub-tile in registers.
// This gives the same result as the textbook k-ordered loop. At step k, row k
// and column k don't change (D[k][k] = 0 with no negative cycles), so the
// phases can read them freely.
//
// Out-of-range elements (N not a multiple of 64) are loaded as +inf and never
// written; inf + finite = inf never wins a min, so they are inert.

#define B 64
#define THREADS 256

__device__ __forceinline__ float ld(const float* d, int N, int r, int c)
{
    return (r < N && c < N) ? d[(size_t)r * N + c] : INFINITY;
}

// ---------------- Phase 1: diagonal tile ----------------
__global__ void __launch_bounds__(THREADS)
fw_phase1(float* __restrict__ d, int N, int kb)
{
    __shared__ float T[B][B + 1];
    const int base = kb * B;
    for (int e = threadIdx.x; e < B * B; e += THREADS) {
        const int i = e / B, j = e % B;
        T[i][j] = ld(d, N, base + i, base + j);
    }
    __syncthreads();
    for (int k = 0; k < B; ++k) {
        for (int e = threadIdx.x; e < B * B; e += THREADS) {
            const int i = e / B, j = e % B;
            T[i][j] = fminf(T[i][j], T[i][k] + T[k][j]);
        }
        __syncthreads();
    }
    for (int e = threadIdx.x; e < B * B; e += THREADS) {
        const int i = e / B, j = e % B;
        if (base + i < N && base + j < N) d[(size_t)(base + i) * N + base + j] = T[i][j];
    }
}

// ---------------- Phase 2: row tiles (blockIdx.y = 0) and column tiles (= 1) ----------------
__global__ void __launch_bounds__(THREADS)
fw_phase2(float* __restrict__ d, int N, int kb)
{
    const int t = blockIdx.x;
    if (t == kb) return;
    __shared__ float Dg[B][B + 1];
    __shared__ float T[B][B + 1];
    const bool isRow = (blockIdx.y == 0);
    const int base = kb * B;
    const int r0 = isRow ? base : t * B;     // tile origin
    const int c0 = isRow ? t * B : base;

    for (int e = threadIdx.x; e < B * B; e += THREADS) {
        const int i = e / B, j = e % B;
        Dg[i][j] = ld(d, N, base + i, base + j);
        T[i][j]  = ld(d, N, r0 + i, c0 + j);
    }
    __syncthreads();

    for (int k = 0; k < B; ++k) {
        for (int e = threadIdx.x; e < B * B; e += THREADS) {
            const int i = e / B, j = e % B;
            const float via = isRow ? Dg[i][k] + T[k][j]    // i in pivot block, j in tile
                                    : T[i][k] + Dg[k][j];   // i in tile, j in pivot block
            T[i][j] = fminf(T[i][j], via);
        }
        __syncthreads();
    }
    for (int e = threadIdx.x; e < B * B; e += THREADS) {
        const int i = e / B, j = e % B;
        if (r0 + i < N && c0 + j < N) d[(size_t)(r0 + i) * N + c0 + j] = T[i][j];
    }
}

// ---------------- Phase 3: independent min-plus tile updates ----------------
__global__ void __launch_bounds__(THREADS)
fw_phase3(float* __restrict__ d, int N, int kb)
{
    const int ti = blockIdx.y, tj = blockIdx.x;
    if (ti == kb || tj == kb) return;

    __shared__ __align__(16) float At[B][B + 4];   // column-tile (ti, kb), transposed: At[k][i]
    __shared__ __align__(16) float Bk[B][B + 4];   // row-tile (kb, tj): Bk[k][j]

    const int base = kb * B, r0 = ti * B, c0 = tj * B;
    for (int e = threadIdx.x; e < B * B; e += THREADS) {
        const int i = e / B, j = e % B;            // j fastest -> coalesced global reads
        At[j][i] = ld(d, N, r0 + i, base + j);
        Bk[i][j] = ld(d, N, base + i, c0 + j);
    }

    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    float c[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) c[a][b] = ld(d, N, r0 + ty * 4 + a, c0 + tx * 4 + b);
    __syncthreads();

#pragma unroll 8
    for (int k = 0; k < B; ++k) {
        const float4 av = *reinterpret_cast<const float4*>(&At[k][ty * 4]);
        const float4 bv = *reinterpret_cast<const float4*>(&Bk[k][tx * 4]);
        const float a[4] = {av.x, av.y, av.z, av.w};
        const float b[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
        for (int p = 0; p < 4; ++p)
#pragma unroll
            for (int q = 0; q < 4; ++q) c[p][q] = fminf(c[p][q], a[p] + b[q]);
    }

#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int r = r0 + ty * 4 + a;
        if (r >= N) continue;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int cc = c0 + tx * 4 + b;
            if (cc < N) d[(size_t)r * N + cc] = c[a][b];
        }
    }
}

// dist, output are device pointers
extern "C" void solve(const float* dist, float* output, int N) {
    cudaMemcpy(output, dist, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    const int nt = (N + B - 1) / B;
    for (int kb = 0; kb < nt; ++kb) {
        fw_phase1<<<1, THREADS>>>(output, N, kb);
        if (nt > 1) {
            fw_phase2<<<dim3(nt, 2), THREADS>>>(output, N, kb);
            fw_phase3<<<dim3(nt, nt), THREADS>>>(output, N, kb);
        }
    }
    cudaDeviceSynchronize();
}
