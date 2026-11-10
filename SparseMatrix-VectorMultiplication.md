### Sparse Matrix-Vector Multiplication
Implement a GPU program that performs sparse matrix-vector multiplication. Given a sparse matrix `A` of dimensions `M x N`  and a dense vector `x` of length `N` , compute the product vector `y = A * x`, which will have length `M`. A is stored in row-major order. nnz is the number of non-zero elements in `A`. Mathematically, the operation is defined as:

Mathematically, the operation is defined as:
<img width="1234" height="86" alt="image" src="https://github.com/user-attachments/assets/6eeb1fb3-a3fd-4c12-a131-41419684a79e" />

The matrix `A` is approximately 60 - 70% sparse.

#### Implementation Requirements
 - Use only GPU native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in vector y

#### Example:
<img width="1260" height="472" alt="image" src="https://github.com/user-attachments/assets/8efc5f6a-70f2-47cc-b7a8-4d270aa9450c" />


#### Constraints
 - 1 ≤ `M`, `N` ≤ 10,000
 - The matrix `A` is approximately 60-70% sparse (i.e., 60-70% of elements are zero)
 - Performance is measured with `M` = 1,000, `N` = 10,000
