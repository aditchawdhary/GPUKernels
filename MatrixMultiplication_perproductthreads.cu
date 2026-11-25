#include <cuda_runtime.h>

__global__ void matmul_per_product_kernel(const float* A, const float* B, float* C,
                                          int M, int N, int K)
{
  int k = blockDim.x * blockIdx.x + threadIdx.x;  // column of C
  int i = blockDim.y * blockIdx.y + threadIdx.y;  // row of C
  int j = blockDim.z * blockIdx.z + threadIdx.z;  // shared (reduction) dimension

  if (i < M && j < N && k < K)
  {
    float product = A[i * N + j] * B[j * K + k];
    atomicAdd(&C[i * K + k], product);            // N threads add into the same cell
  }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K)
{
  // C must start at zero, since every thread adds into it
  cudaMemset(C, 0, (size_t)M * K * sizeof(float));

  dim3 threadsPerBlock(8, 8, 8);  //  512 threads (max is 1024 per block)
  dim3 blocksPerGrid((K + 7) / 8, (M + 7) / 8, (N + 7) / 8);

  matmul_per_product_kernel<<<blocksPerGrid, threadsPerblock>>>(A, B, C, M, N, K);
}
