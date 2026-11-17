## Fused Residual Add and LayerNorm with Saved Statistics

Implement the fused pre-norm residual block entry used in GPT-2, BERT, and most LayerNorm transformers: add the sublayer output x to the residual stream residual, apply Layer Normalization with learned gamma and beta, and in the same pass write out everything the next layer and the backward pass need. All inputs are float32; x and residual have shape (N, C) (one row per token), gamma and beta have shape (C,). You must produce four outputs: the normalized output y (N, C), the updated residual stream residual_out (N, C), and the per-row mean and reciprocal standard deviation rstd, each of shape (N,).

<img width="1012" height="328" alt="image" src="https://github.com/user-attachments/assets/0ad3b6cf-30cd-42a1-89ef-549540eec15e" />

<img width="1034" height="252" alt="image" src="https://github.com/user-attachments/assets/d5db8530-4392-4b4a-ab77-9cb61d7d166a" />

Store h in `residual_out`, \mu in `mean`, r and in `rstd`. The variance is the biased (population) variance. Rows may have a large mean relative to their spread, so compute the variance in a numerically stable way.
 
 ## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The results must be stored in `y`, `residual_out`, `mean`, and `rstd`
 - The outputs are not `zero-initialized`; every element must be written

## Example
Input (`N` = 2, `C` = 4, `eps` = `1e-5`):
  
<img width="1045" height="382" alt="image" src="https://github.com/user-attachments/assets/60894b23-c59d-4659-95a3-69dec98193d5" />
 
  
## Constraints
 - 1 ≤ `N` ≤ 65,536
 - 1 ≤ `C` ≤ 16,384
 - All tensors are `float32`
 - Values of `x` and `residual` are in `[−1,000, 1,000]`; values of `gamma` are in `[−2, 2]` and `beta` in `[−0.5, 0.5]`
 - `1e-6` ≤ `eps` ≤ `1.0`
 - Performance is measured with `N` = `16,384`, `C` = `4,096`

## `solve.cu`
```
#include <cuda_runtime.h>

// x, residual, gamma, beta, y, residual_out, mean, rstd are device pointers
extern "C" void solve(const float* x, const float* residual, const float* gamma, const float* beta,
                      float* y, float* residual_out, float* mean, float* rstd, int N, int C,
                      float eps) {}

```
