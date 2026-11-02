#include <cuda_runtime.h>

#define TILE 16
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    __shared__ float tile[TILE][TILE + 1];

    int column = TILE * blockIdx.x + threadIdx.x;
    int row = TILE * blockIdx.y + threadIdx.y;
    
    if (row < rows && column < cols)
    {
        tile[threadIdx.y][threadIdx.x] = input[row * cols + column];
    }

    __syncthreads();

    column = TILE * blockIdx.y + threadIdx.x;
    row = TILE * blockIdx.x + threadIdx.y;

    if (row < cols && column < rows)
    {
        output[row * rows + column] = tile[threadIdx.x][threadIdx.y];
    }


    // int column = blockDim.x  * blockIdx.x + threadIdx.x;
    // int row = blockDim.y  * blockIdx.y + threadIdx.y;
    
    // if (column < cols && row < rows)
    // {
    //     output[column * rows + row] = input[row * cols + column];
    // }
    

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
