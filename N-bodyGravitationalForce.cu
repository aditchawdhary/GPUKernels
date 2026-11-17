/*
N-body Gravitational Force
positions: (N, 3), masses: (N), forces: (N, 3), float32, row-major.
Spec from the challenge's reference implementation (leetgpu-challenges #101):
  F_i = sum_j m_j * (p_j - p_i) / (|p_j - p_i|^2 + eps^2)^(3/2),   eps = 1e-3
(no G and no m_i factor; the j = i term is zero because p_j - p_i = 0). Tolerance 1e-2.

Constraints: N ≤ 65,536, positions in [-10, 10], masses in (0, 100].
Performance is measured with N = 8192.
*/

/*
Approach: the classic shared-memory tiled all-pairs kernel, plus a split over the
source bodies so a small N still fills the GPU.

 - Each thread owns one target body i (128 threads per block) and keeps its
   position and force accumulator in registers.
 - Source bodies are staged through shared memory 128 at a time as float4
   (x, y, z, m). Every thread then runs the inner loop over the tile
   (shared-memory broadcast reads, about 20 flops per pair, one rsqrtf):
     d = p_j - p_i,  r2 = |d|^2 + eps^2,  inv = rsqrt(r2),  F += m_j * inv^3 * d
 - Split-J: N = 8192 gives only 64 target blocks, so the source range is also cut
   into S slices: grid (target blocks, S), about 512 blocks or more. Each block
   writes a partial force to scratch [S][N][3], and a tiny kernel sums the S
   partials in a fixed order, so the result is deterministic. With S = 1 the main
   kernel writes forces directly.
At the benchmark: 67M pair interactions, about 1.3 GFLOP, compute-bound and well
under a millisecond.
*/

#include <cuda_runtime.h>
#include <math.h>

#define T 128
#define EPS2 (1e-3f * 1e-3f)

__global__ void __launch_bounds__(T)
nbody_partial(const float* __restrict__ pos, const float* __restrict__ mass,
              float* __restrict__ out, int N, int jPerSplit)
{
    __shared__ float4 tile[T];

    const int i = blockIdx.x * T + threadIdx.x;
    const bool active = i < N;
    float xi = 0.f, yi = 0.f, zi = 0.f;
    if (active) { xi = pos[3 * i]; yi = pos[3 * i + 1]; zi = pos[3 * i + 2]; }

    const int j0 = blockIdx.y * jPerSplit;
    const int j1 = min(j0 + jPerSplit, N);
    float fx = 0.f, fy = 0.f, fz = 0.f;

    for (int base = j0; base < j1; base += T) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < j1)
            ? make_float4(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2], mass[j])
            : make_float4(0.f, 0.f, 0.f, 0.f);       // m = 0 -> no contribution
        __syncthreads();

        const int cnt = min(T, j1 - base);
        #pragma unroll 8
        for (int k = 0; k < cnt; ++k) {
            const float4 b = tile[k];
            const float dx = b.x - xi, dy = b.y - yi, dz = b.z - zi;
            const float r2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, EPS2)));
            const float inv = rsqrtf(r2);
            const float s = b.w * inv * inv * inv;
            fx = fmaf(s, dx, fx);
            fy = fmaf(s, dy, fy);
            fz = fmaf(s, dz, fz);
        }
        __syncthreads();
    }

    if (active) {
        float* o = out + ((size_t)blockIdx.y * N + i) * 3;
        o[0] = fx; o[1] = fy; o[2] = fz;
    }
}

__global__ void reduce_splits(const float* __restrict__ part, float* __restrict__ forces,
                              int N, int S)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;     // over N*3
    if (idx >= 3 * N) return;
    float acc = 0.f;
    for (int s = 0; s < S; ++s) acc += part[(size_t)s * 3 * N + idx];
    forces[idx] = acc;
}

static float* g_part = nullptr;
static size_t g_cap = 0;

// positions, masses, forces are device pointers
extern "C" void solve(const float* positions, const float* masses, float* forces, int N)
{
    if (N <= 0) return;
    const int iBlocks = (N + T - 1) / T;
    const int jTiles = (N + T - 1) / T;
    int S = (512 + iBlocks - 1) / iBlocks;                    // aim for ≥ 512 blocks
    if (S > jTiles) S = jTiles;
    if (S < 1) S = 1;
    const int jPerSplit = ((jTiles + S - 1) / S) * T;         // whole tiles per split
    S = (N + jPerSplit - 1) / jPerSplit;

    if (S == 1) {
        nbody_partial<<<dim3(iBlocks, 1), T>>>(positions, masses, forces, N, jPerSplit);
    } else {
        const size_t need = (size_t)S * 3 * N;
        if (need > g_cap) {
            if (g_part) cudaFree(g_part);
            cudaMalloc(&g_part, need * sizeof(float));
            g_cap = need;
        }
        nbody_partial<<<dim3(iBlocks, S), T>>>(positions, masses, g_part, N, jPerSplit);
        reduce_splits<<<(3 * N + 255) / 256, 256>>>(g_part, forces, N, S);
    }
    cudaDeviceSynchronize();
}
