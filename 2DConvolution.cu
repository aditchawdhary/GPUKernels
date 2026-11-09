#include <cuda_runtime.h>

// Tiled 2D "valid" convolution (cross-correlation, as defined in the problem):
//   output[i][j] = sum_{m,n} input[i+m][j+n] * kernel[m][n]

#define MAX_K   31
#define TILE_W  32
#define TILE_H  32
#define BLOCK_X 32
#define BLOCK_Y 8
#define ROWS_PER_THREAD (TILE_H / BLOCK_Y)

__constant__ float c_kernel[MAX_K * MAX_K];

__global__ void conv2d_tiled(const float* __restrict__ input, float* __restrict__ output,
                             int in_rows, int in_cols, int kr, int kc,
                             int out_rows, int out_cols) {
    extern __shared__ float tile[];

    const int sw = TILE_W + kc - 1;          // shared tile width
    const int sh = TILE_H + kr - 1;          // shared tile height
    const int row0 = blockIdx.y * TILE_H;    // top-left of this block's output tile
    const int col0 = blockIdx.x * TILE_W;

    // Cooperative, coalesced load of the input tile (with halo) into shared memory.
    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;
    for (int idx = tid; idx < sh * sw; idx += BLOCK_X * BLOCK_Y) {
        const int r = idx / sw, c = idx - r * sw;
        const int gr = row0 + r, gc = col0 + c;
        tile[idx] = (gr < in_rows && gc < in_cols) ? input[(size_t)gr * in_cols + gc] : 0.0f;
    }
    __syncthreads();

    const int tx = threadIdx.x;
    float acc[ROWS_PER_THREAD];
#pragma unroll
    for (int k = 0; k < ROWS_PER_THREAD; ++k) acc[k] = 0.0f;

    for (int m = 0; m < kr; ++m) {
        for (int n = 0; n < kc; ++n) {
            const float w = c_kernel[m * kc + n];
#pragma unroll
            for (int k = 0; k < ROWS_PER_THREAD; ++k) {
                const int ty = threadIdx.y + k * BLOCK_Y;
                acc[k] += tile[(ty + m) * sw + tx + n] * w;
            }
        }
    }

    const int oc = col0 + tx;
#pragma unroll
    for (int k = 0; k < ROWS_PER_THREAD; ++k) {
        const int orow = row0 + threadIdx.y + k * BLOCK_Y;
        if (orow < out_rows && oc < out_cols)
            output[(size_t)orow * out_cols + oc] = acc[k];
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    const int out_rows = input_rows - kernel_rows + 1;
    const int out_cols = input_cols - kernel_cols + 1;
    if (out_rows <= 0 || out_cols <= 0) return;

    cudaMemcpyToSymbol(c_kernel, kernel, sizeof(float) * kernel_rows * kernel_cols, 0,
                       cudaMemcpyDeviceToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((out_cols + TILE_W - 1) / TILE_W, (out_rows + TILE_H - 1) / TILE_H);
    size_t smem = sizeof(float) * (TILE_H + kernel_rows - 1) * (TILE_W + kernel_cols - 1);

    conv2d_tiled<<<grid, block, smem>>>(input, output, input_rows, input_cols,
                                        kernel_rows, kernel_cols, out_rows, out_cols);
    cudaDeviceSynchronize();
}
