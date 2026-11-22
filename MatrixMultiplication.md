## Matrix Multiplication

Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix `A` of dimensions `M x N` and matrix `B` of dimensions `N x K`, compute the product matrix `C = A x B`, which will have dimensions 
`M x K`. All matrices are stored in row-major format.

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The `solve` function signature must remain unchanged
 - The final result must be stored in matrix `C`

## Example 1:
<img width="711" height="396" alt="image" src="https://github.com/user-attachments/assets/efd7185b-def4-4ba5-a48d-60a3ac8a0ac5" />


## Example 2:
<img width="543" height="363" alt="image" src="https://github.com/user-attachments/assets/744ed0bc-8866-4ed9-b12d-4ac3343765d4" />
 

## Constraints
 - 1 ≤ `M`, `N`, `K` ≤ 8192
 - Performance is measured with `M = 8192`, `N = 6144`, `K = 4096`
