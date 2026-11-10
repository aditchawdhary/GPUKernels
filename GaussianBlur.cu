#include <cuda_runtime.h>

// output[i][j] = sum_{m,n} input[i+m][j+n] * kernel[m+kh/2][n+kw/2]   (zero padding)
//
// - Filter (<= 21x21 = 441 floats) lives in constant memory: every thread in a
//   warp reads the same tap at the same time -> broadcast, no bandwidth cost.
// - Each block loads a zero-padded input tile (+halo) into shared memory once,
//   then each thread computes RY outputs from shared memory.

#define TX 32            // block width  (= output tile width)
#define TY 8             // block height
#define RY 4             // outputs per thread, stacked vertically
#define OUT_H (TY * RY)  // output tile height = 32
#define MAXK 21

__constant__ float cK[MAXK * MAXK];

__global__ void __launch_bounds__(TX * TY)
blur_kernel(const float* __restrict__ input, float* __restrict__ output,
            int rows, int cols, int kh, int kw)
{
    // +1 column of padding keeps row starts off the same bank
    __shared__ float tile[OUT_H + MAXK - 1][TX + MAXK - 1 + 1];

    const int kh2 = kh >> 1, kw2 = kw >> 1;
    const int tileH = OUT_H + kh - 1;
    const int tileW = TX + kw - 1;
    const int ox0 = blockIdx.x * TX;
    const int oy0 = blockIdx.y * OUT_H;

    // Cooperative load of tile + halo, zero outside the image
    for (int y = threadIdx.y; y < tileH; y += TY) {
        const int gy = oy0 + y - kh2;
        const bool rowIn = (gy >= 0 && gy < rows);
        for (int x = threadIdx.x; x < tileW; x += TX) {
            const int gx = ox0 + x - kw2;
            tile[y][x] = (rowIn && gx >= 0 && gx < cols)
                         ? __ldg(&input[(size_t)gy * cols + gx]) : 0.0f;
        }
    }
    __syncthreads();

    const int tx = threadIdx.x;
    const int gx = ox0 + tx;

    float acc[RY];
    #pragma unroll
    for (int r = 0; r < RY; ++r) acc[r] = 0.0f;

    // For each filter tap, update all RY outputs of this thread
    for (int m = 0; m < kh; ++m) {
        for (int n = 0; n < kw; ++n) {
            const float w = cK[m * kw + n];
            #pragma unroll
            for (int r = 0; r < RY; ++r)
                acc[r] = fmaf(tile[threadIdx.y + r * TY + m][tx + n], w, acc[r]);
        }
    }

    if (gx < cols) {
        #pragma unroll
        for (int r = 0; r < RY; ++r) {
            const int gy = oy0 + threadIdx.y + r * TY;
            if (gy < rows) output[(size_t)gy * cols + gx] = acc[r];
        }
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    cudaMemcpyToSymbol(cK, kernel, sizeof(float) * kernel_rows * kernel_cols,
                       0, cudaMemcpyDeviceToDevice);

    dim3 block(TX, TY);
    dim3 grid((input_cols + TX - 1) / TX, (input_rows + OUT_H - 1) / OUT_H);
    blur_kernel<<<grid, block>>>(input, output, input_rows, input_cols,
                                 kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
