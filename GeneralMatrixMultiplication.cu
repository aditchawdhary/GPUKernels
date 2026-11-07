/*
General Matrix Multiplication (GEMM)

Implement a basic General Matrix Multiplication (GEMM). Given matrix A of dimensions M x K, matrix B of dimensions K x N, input/output matrix C of dimensions M x N, and scalar multipliers alpha and beta, compute the operation:

C = alpha * (A x B) + beta * C_initial

The input matrices, A, B and the initial state of C contain 16-bit floating-point numbers (FP16/half). All matrices are stored in row-major order. The scalars alpha and beta are 32-bit floats.

Implementation Requirements
 - Use only native features (external libraries other than WMMA are not permitted).
 - The solve function signature must remain unchanged.
 - Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16.
 - The final result must be stored back into matrix C as half.

Example:
Input: (Note: Input matrices A, B, C_initial are FP16 type for the problem)

Matrix A (M=2, K=3)
[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

Matrix B (K=3, N=2) 
[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

Matrix C_initial (M=2, N=2):
[[1.0, 1.0], [1.0, 1.0]]
alpha = 1.0 (FP32) 
beta = 0.0 (FP32)
  
Output (FP16):
Matrix  C (M=2, N=2):
[[22.0, 28.0], [49.0, 64.0]]
 
Constraints
 - 16 ≤ M, N, K ≤ 4096
 - Performance is measured with K = 1,024, M = 1,024, N = 1,024
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void gemm_kernel(const half* A, const half* B, half* C,
                            int M, int N, int K, float alpha, float beta)
{
    // Tiles of A and B, stored as float so the math is done in FP32
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;   // which row of C this thread computes
    int col = blockIdx.x * TILE + tx;   // which column of C this thread computes

    float acc = 0.0f;                   // FP32 accumulator

    // Walk along K one tile at a time
    for (int t = 0; t < K; t += TILE) {
        int aCol = t + tx;
        int bRow = t + ty;

        // Each thread loads one element of A and one of B (0 if out of bounds)
        As[ty][tx] = (row < M && aCol < K) ? __half2float(A[row * K + aCol]) : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? __half2float(B[bRow * N + col]) : 0.0f;
        __syncthreads();

        // Dot product of this tile's row of A and column of B
        for (int k = 0; k < TILE; k++)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    // C = alpha * (A x B) + beta * C_initial, converted back to half
    if (row < M && col < N) {
        float cInit = __half2float(C[row * N + col]);
        C[row * N + col] = __float2half(alpha * acc + beta * cInit);
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    gemm_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
