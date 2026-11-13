## Muon Optimizer
Implement the update computation of one Muon optimizer step. Muon (Keller Jordan, 2024) is the optimizer used to train Moonlight and Kimi K2; instead of scaling each gradient coordinate independently like AdamW, 
it orthogonalizes the whole momentum-updated gradient matrix, so every singular direction of the update contributes almost equally. Given a gradient matrix `G` and the optimizer's momentum buffer `Buf`, compute the orthogonalized, 
rescaled update and write it to `output`. All tensors are `float32`.

<img width="1464" height="393" alt="image" src="https://github.com/user-attachments/assets/3a4dc13c-50ee-4283-b7c7-8e35b731e419" />


Scope. This challenge computes the Muon update direction only: there is no parameter update, no weight decay, and no Nesterov momentum. 
Correctness is judged against the `float32` 5-step Newton–Schulz iteration specified above, not against the exact SVD polar factor of `V`.

## Implementation Requirements
Implement `solve(G, Buf, output, M, N, momentum)`:

 - `G`: shape `(M, N)`, the current gradient of a weight matrix; it is a read-only input and must not be modified
 - `Buf`: shape `(M, N)`, the momentum buffer; it is a read-only input and must not be modified
 - `output`: shape `(M, N)`, the orthogonalized update
 - `M, N`: the dimensions of the weight matrix
 - `momentum`: the momentum coefficient used to combine `Buf` and `G`
 - Use only native features of the framework you choose (no external libraries); keep the function signature unchanged
 - Write the result into `output`; every entry of `output` must be written
 - Run exactly 5 Newton–Schulz steps with the coefficients above; when `M > N`, the iteration may equivalently be run on the transposed matrix so the Gram matrix `X XT` is formed on the smaller dimension
 - All matrix-multiplication inputs and accumulations must be `float32` or higher precision (no `TF32`, `float16` or `bfloat16` operands); reduced-precision matrix multiplication does not meet the tolerance

## Example
Input: `M = 2`, `N = 3`, `momentum = 0.9`:

<img width="727" height="312" alt="image" src="https://github.com/user-attachments/assets/f7137481-0159-444c-87da-3e0ac7290866" />


## Constraints
 - All tensors are `float32`
 - 1 ≤ `M` ≤ 1,024
 - 1 ≤ `N` ≤ 1,024
 - 0 ≤ `momentum` < 1
 - Entries of `G` and `Buf` are drawn from a standard normal distribution in the tests, apart from dedicated edge cases with all-zero and `1e-8-magnitude` inputs
 - Performance is measured with `M = 1,024`, `N = 1,024`, `momentum = 0.9`

## `solve.cu`
```
#include <cuda_runtime.h>

// G, Buf, output are device pointers
extern "C" void solve(const float* G, const float* Buf, float* output, int M, int N,
                      float momentum) {}

```
