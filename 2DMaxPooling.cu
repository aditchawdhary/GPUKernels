#include <cuda_runtime.h>
#include <math.h>

// 2D max pooling, NCHW, float32.
// out[n,c,oh,ow] = max over input[n,c, oh*stride - padding + i, ow*stride - padding + j],
// i,j in [0, kernel_size). Padded positions are ignored (treated as -inf).
__global__ void maxpool2d_kernel(const float* __restrict__ input, float* __restrict__ output,
                                 int NC, int H, int W, int H_out, int W_out,
                                 int k, int stride, int padding) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc = blockIdx.z;
    if (ow >= W_out || oh >= H_out || nc >= NC) return;

    const float* plane = input + (size_t)nc * H * W;

    // Clip the window to the valid input region once, so the inner loop has no bounds checks.
    const int h0 = oh * stride - padding, w0 = ow * stride - padding;
    const int hs = max(h0, 0), he = min(h0 + k, H);
    const int ws = max(w0, 0), we = min(w0 + k, W);

    float m = -INFINITY;
    for (int h = hs; h < he; ++h) {
        const float* row = plane + (size_t)h * W;
        for (int w = ws; w < we; ++w)
            m = fmaxf(m, __ldg(row + w));
    }
    output[((size_t)nc * H_out + oh) * W_out + ow] = m;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    const int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    if (H_out <= 0 || W_out <= 0) return;

    const int NC = N * C;   // ≤ 51,200, fits in gridDim.z (max 65,535)
    dim3 block(32, 8);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, NC);
    maxpool2d_kernel<<<grid, block>>>(input, output, NC, H, W, H_out, W_out,
                                      kernel_size, stride, padding);
    cudaDeviceSynchronize();
}
