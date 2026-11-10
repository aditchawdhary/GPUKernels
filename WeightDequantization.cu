#include <cuda_runtime.h>

// Y[i][j] = X[i][j] * S[i / T][j / T],   S is ceil(M/T) x ceil(N/T)
//
// Purely memory-bound (512 MB moved at 8192 x 8192): read X once, write Y once.
// - One block per row, so the scale row index (i >> log2 T) is block-uniform.
// - T is a power of two >= 16, so a float4 group (4 aligned columns) never
//   straddles a tile: one scale per float4, found with a shift.
// - S is tiny (64x64 floats at the benchmark size) and stays in L1/L2.
// - Streaming loads/stores (__ldcs/__stcs) since nothing is reused.

#define THREADS 256

__global__ void __launch_bounds__(THREADS)
dequant_vec4(const float* __restrict__ X, const float* __restrict__ S,
             float* __restrict__ Y, int N, int sCols, int logT)
{
    const int row = blockIdx.x;
    const float* srow = S + (size_t)(row >> logT) * sCols;
    const float4* x4 = reinterpret_cast<const float4*>(X + (size_t)row * N);
    float4*       y4 = reinterpret_cast<float4*>(Y + (size_t)row * N);
    const int N4 = N >> 2;
    const int logT4 = logT - 2;                     // float4 index -> tile column

    #pragma unroll 4
    for (int c = threadIdx.x; c < N4; c += THREADS) {
        const float  s = __ldg(srow + (c >> logT4));
        float4 v = __ldcs(x4 + c);
        v.x *= s; v.y *= s; v.z *= s; v.w *= s;
        __stcs(y4 + c, v);
    }
}

__global__ void __launch_bounds__(THREADS)
dequant_scalar(const float* __restrict__ X, const float* __restrict__ S,
               float* __restrict__ Y, int N, int sCols, int logT)
{
    const int row = blockIdx.x;
    const float* srow = S + (size_t)(row >> logT) * sCols;
    const float* x = X + (size_t)row * N;
    float*       y = Y + (size_t)row * N;
    for (int j = threadIdx.x; j < N; j += THREADS)
        __stcs(y + j, __ldcs(x + j) * __ldg(srow + (j >> logT)));
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE)
{
    int logT = 0;
    while ((1 << logT) < TILE_SIZE) ++logT;        // TILE_SIZE in {16, 32, 64, 128}
    const int sCols = (N + TILE_SIZE - 1) / TILE_SIZE;

    // float4 path needs 16 B aligned rows and tiles at least 4 columns wide
    // (the example uses TILE_SIZE = 2, below the stated constraint)
    if ((N & 3) == 0 && TILE_SIZE >= 4) dequant_vec4<<<M, THREADS>>>(X, S, Y, N, sCols, logT);
    else              dequant_scalar<<<M, THREADS>>>(X, S, Y, N, sCols, logT);
    cudaDeviceSynchronize();
}
