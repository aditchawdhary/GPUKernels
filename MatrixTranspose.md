
# Matrix Transpose
<img width="426" height="187" alt="image" src="https://github.com/user-attachments/assets/9de9ffaf-cb07-45eb-a6e1-79164e18de27" />


Write a program that transposes a matrix of 32-bit floating point numbers on a GPU. 
The transpose of a matrix switches its rows and columns. Given a matrix A of dimensions 
`rows x cols`, the transpose A^T will have dimensions `cols x rows`. All matrices are stored in row-major format.

```
A =  [[1, 2], [3, 4], [5, 6]]    3 × 2
rows → cols
Aᵀ = [[1, 3, 5], [2, 4, 6]]      2 × 3
```
Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the matrix output

```
Example 1:
Input: 2×3 matrix
[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
 
Output: 3×2 matrix
[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
```
 
```
Example 2:
Input: 3×1 matrix
[[1.0], [2.0], [3.0]]
 
Output: 1×3 matrix
[[1.0, 2.0, 3.0]]
```
 
Constraints
- 1 ≤ rows, cols ≤ 8192
- Input matrix dimensions: rows × cols
- Output matrix dimensions: cols × rows
- Performance is measured with cols = 6,000, rows = 7,000

```cuda
#include <cuda_runtime>
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column < cols && row < rows)
    {
        output[rows * column + row] = input[cols * row + column];
    }  
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

```
