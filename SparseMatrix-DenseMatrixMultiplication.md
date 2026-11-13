## Sparse Matrix-Dense Matrix Multiplication

Implement a GPU program that multiplies a sparse matrix `A` of dimensions `M × N` by a dense matrix `B` of dimensions `N × K`, producing a dense output matrix `C` of dimensions `M × K`. All matrices are stored in row-major order using `32-bit` floats. 
The matrix `A` is approximately `60–70%` sparse (i.e., `60–70%` of elements are zero), and `nnz` gives the number of non-zero elements in A.

Mathematically, the operation is defined as:
<img width="813" height="93" alt="image" src="https://github.com/user-attachments/assets/4a4f1dce-3dd1-4682-bb36-92a30e1dfc9c" />


## Implementation Requirements
 - Use only GPU native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in matrix `C`

## Example
<img width="817" height="509" alt="image" src="https://github.com/user-attachments/assets/433f544e-5a43-4bcf-8845-d0e3d2652a28" />


## Constraints
 - 1 ≤ `M`, `N`, `K` ≤ 8,192
 - All values in `A` and `B` are `32-bit` floats in the range `[−10, 10]`
 - The matrix A is approximately `60–70%` sparse
 - Performance is measured with `M = 4,096`, `N = 2,048`, `K = 512`

## `solve.cu`
```
#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz) {}

```
