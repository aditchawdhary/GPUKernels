#include <cuda_runtime.h>

__global__ void subarray_sum_kernel(const int* __restrict__ input, int* __restrict__ output,
                                    int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int sum = 0;
    // Blocks stride over rows; threads stride over columns (coalesced reads).
    for (int r = S_ROW + blockIdx.x; r <= E_ROW; r += gridDim.x) {
        const int* row = input + (size_t)r * M;
        for (int c = S_COL + threadIdx.x; c <= E_COL; c += blockDim.x)
            sum += __ldg(row + c);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);

    __shared__ int warp_sums[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0;
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0) atomicAdd(output, sum);
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M,
                      int S_ROW, int E_ROW, int S_COL, int E_COL) {
    cudaMemset(output, 0, sizeof(int));
    const int rows = E_ROW - S_ROW + 1;
    const int blocks = rows < 2048 ? rows : 2048;
    subarray_sum_kernel<<<blocks, 256>>>(input, output, M, S_ROW, E_ROW, S_COL, E_COL);
    cudaDeviceSynchronize();
}
