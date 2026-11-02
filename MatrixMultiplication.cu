/**
* Matrix Multiplication
* Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. 
* Given matrix A of dimensions M x N and matrix B of dimensions N x K, compute the product matrix C = A x B, which will have dimensions M x K. All matrices are stored in row-major format.
*
* Implementation Requirements
* * Use only native features (external libraries are not permitted)
* * The solve function signature must remain unchanged
* * The final result must be stored in matrix C
*
* Example 1:
* Input:
* Matrix A (2x2): [[1.0, 2.0], 
*                  [3.0, 4.0]]
* 
* Matrix B (2x2): [[5.0, 6.0], 
*                  [7.0, 8.0]]
* 
* Output:
* Matrix C (2x2): [[19.0, 22.0], 
*                  [43.0, 50.0]]
* 
*
* Example 2:
* Input:
* Matrix A (1x3) = [[1.0, 2.0, 3.0]] 
* 
* Matrix B (3x1) = [[4.0], [5.0], [6.0]]
* 
* Output:
* Matrix C (1x1) = [[32.0]]
* 
*
* Constraints
* * 1 ≤ M, N, K ≤ 8192
* * Performance is measured with M = 8192, N = 6144, K = 4096
*/

#include <cuda_runtime.h>
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (row < M && col < K)
    {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
        {
          sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

/*
what matters isn’t the order a single thread reads in, it’s what the 32 threads of a warp read at the same moment.

Each thread reads the same thing in both versions. No matter how you map x and y, the thread computing C[row][col] walks across row row of A and down column col of B. 
That loop is identical in both versions, so it can’t explain the speed difference.

The difference is between threads. The GPU runs a warp of 32 threads in lockstep, and those 32 threads have consecutive threadIdx.x values (with the same threadIdx.y). 
On each loop iteration, all 32 issue a load together. If their 32 addresses sit next to each other, the hardware fetches them in one or two transactions. If they’re scattered, it needs up to 32.

Take one loop step i and look at what a warp touches:
____________________________|_____________________________________________|_________________________________________________________|__________
	                          |    x → column (fast)	                      |      x → row (slow)                                     |
____________________________|_____________________________________________|_________________________________________________________|__________
Warp’s threads differ in	  |    column (0, 1, 2 … 31)	                  |      row (0, 1, 2 … 31)                                 |
____________________________|_____________________________________________|_________________________________________________________|__________
B[i*K + column]	            |    adjacent addresses ✅	                  |      all the same address (broadcast, fine)             |
____________________________|_____________________________________________|_________________________________________________________|__________
A[row*N + i]	              |    all the same address (broadcast, fine)	  |    addresses N apart ❌                                 |
____________________________|_____________________________________________|_________________________________________________________|__________
C[row*K + column]	          |    adjacent addresses ✅	                  |      addresses K apart ❌                               |
____________________________|_____________________________________________|_________________________________________________________|__________

So the rule is: map threadIdx.x to the index that’s contiguous in memory. In row-major storage, that’s the column index. Moving one column right is +1 address, while moving one row down is +K addresses.

On your two grid options: the first argument of dim3 is always x and the second is always y. So the choice is really “x covers K (columns)” or “x covers M (rows)”. Both give correct results as long as the kernel agrees, but only the first gives coalesced access:

cuda
dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,   // x → columns
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);  // y → rows

In short: data is stored row by row, so neighboring threads should handle neighboring columns.

*/
