#include <cuda_runtime.h>

// output = A^P via binary exponentiation: at most 8 matmuls for P <= 20
// (P = 20 -> 4 squarings + 1 multiply) instead of P - 1.
// The order of operations mirrors torch.linalg.matrix_power:
//   z = A; result = none
//   loop over bits of P (LSB first): if bit: result = result @ z; z = z @ z
//
// Matmul: shared-memory tiled FP32 SGEMM, TILE x TILE output per block,
// BK = 16, each thread computes a 4x4 micro-tile from registers.
// TILE = 32 (64 threads) when there aren't enough 64-tiles to fill the GPU
// (N = 512 -> 256 blocks), TILE = 64 (256 threads) otherwise.

#define BK 16

template <int TILE>
__global__ void __launch_bounds__((TILE / 4) * (TILE / 4))
sgemm(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N)
{
    constexpr int TD = TILE / 4;                  // threads per dim
    constexpr int NT = TD * TD;                   // threads per block
    __shared__ float As[BK][TILE + 1];            // A tile, k-major (transposed)
    __shared__ float Bs[BK][TILE];

    const int tx = threadIdx.x % TD, ty = threadIdx.x / TD;
    const int bm = blockIdx.y * TILE, bn = blockIdx.x * TILE;

    float acc[4][4] = {};

    for (int k0 = 0; k0 < N; k0 += BK) {
        // A tile: TILE rows x BK cols; consecutive threads walk along k
        for (int id = threadIdx.x; id < TILE * BK; id += NT) {
            const int r = id / BK, k = id % BK;
            const int gr = bm + r, gk = k0 + k;
            As[k][r] = (gr < N && gk < N) ? __ldg(A + (size_t)gr * N + gk) : 0.f;
        }
        // B tile: BK rows x TILE cols; consecutive threads walk along n
        for (int id = threadIdx.x; id < TILE * BK; id += NT) {
            const int k = id / TILE, c = id % TILE;
            const int gk = k0 + k, gc = bn + c;
            Bs[k][c] = (gk < N && gc < N) ? __ldg(B + (size_t)gk * N + gc) : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[k][ty + TD * i];   // broadcast across tx
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx + TD * j];   // consecutive banks
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = bm + ty + TD * i;
        if (r >= N) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = bn + tx + TD * j;
            if (c < N) C[(size_t)r * N + c] = acc[i][j];
        }
    }
}

static int g_tile = 64;

static void matmul(const float* A, const float* B, float* C, int N)
{
    if (g_tile == 64) {
        dim3 grid((N + 63) / 64, (N + 63) / 64);
        sgemm<64><<<grid, 256>>>(A, B, C, N);
    } else {
        dim3 grid((N + 31) / 32, (N + 31) / 32);
        sgemm<32><<<grid, 64>>>(A, B, C, N);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P)
{
    const size_t bytes = (size_t)N * N * sizeof(float);

    if (P == 1) {
        cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice);
        return;
    }

    int dev = 0, sms = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const int t64 = (N + 63) / 64;
    g_tile = (t64 * t64 >= 2 * sms) ? 64 : 32;

    // Scratch: two z buffers + one result buffer (ping-ponged with output)
    static float* scratch = nullptr;
    static size_t cap = 0;
    if (3 * bytes > cap) {
        if (scratch) cudaFree(scratch);
        cudaMalloc(&scratch, 3 * bytes);
        cap = 3 * bytes;
    }
    float* zbuf[2] = { scratch, scratch + (size_t)N * N };
    float* rbuf[2] = { scratch + 2 * (size_t)N * N, output };

    const float* z = input;   // current A^(2^i)
    int zi = 0;
    const float* r = nullptr; // running result
    int ri = 0;

    int p = P;
    while (p > 0) {
        if (p & 1) {
            if (!r) {
                // first factor: result = z (copy, so z's buffer can be reused)
                cudaMemcpyAsync(rbuf[ri], z, bytes, cudaMemcpyDeviceToDevice);
            } else {
                matmul(r, z, rbuf[ri ^ 1], N);   // result = result @ z
                ri ^= 1;
            }
            r = rbuf[ri];
        }
        p >>= 1;
        if (p > 0) {                             // z = z @ z (skip after last bit)
            matmul(z, z, zbuf[zi], N);
            z = zbuf[zi];
            zi ^= 1;
        }
    }

    if (r != output) cudaMemcpyAsync(output, r, bytes, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}
