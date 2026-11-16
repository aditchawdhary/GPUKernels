## Fused Residual Add and RMS Norm

Implement a fused kernel that adds an input tensor x to a residual tensor and then RMS-normalizes the result, scaling each feature by a learned per-feature weight vector. 
Inputs are `float32` tensors: `x` and `residual` have shape `(N, C)` where `N` is the number of tokens and `C` is the hidden dimension, and weight has shape `(C,)`. 
This pattern appears before every sublayer in modern LLMs such as LLaMA and Mistral: fusing the addition and normalization into a single kernel saves a full read–write pass over the data compared to two separate operations.

<img width="1020" height="252" alt="image" src="https://github.com/user-attachments/assets/aa230ba8-5962-4385-90cb-7c1fb1aa45b4" />

For each row i (token), the kernel must compute:
<img width="1022" height="105" alt="image" src="https://github.com/user-attachments/assets/57ada3df-f122-4721-af64-1ece0c3b6222" />

 
 

Each row's RMS is computed independently, making this a per-token normalization. The challenge is to fuse the addition and normalization into a single GPU kernel, avoiding writing the intermediate sum z to global memory.

## Implementation Requirements
 - Do not change the function signature.
 - Do not use external libraries beyond what is available in the starter.
 - out[i, j] must equal (x[i,j] + residual[i,j]) / rms_i × weight[j] where rms_i is computed over the full row i.

## Example
```
N = 1, C = 4
x        = [1.0,  0.0, -1.0,  2.0]
residual = [0.5,  1.5,  0.5, -0.5]
weight   = [1.0,  1.0,  1.0,  1.0]
eps      = 1e-5

z   = [1.5,  1.5, -0.5,  1.5]
rms = sqrt((1.5^2 + 1.5^2 + 0.5^2 + 1.5^2) / 4 + 1e-5)
    = sqrt(1.75 + 1e-5) ≈ 1.32288

out ≈ [1.1339,  1.1339, -0.3780,  1.1339]
```

## Constraints
 - 1 ≤ `N` ≤ `65,536`
 - 1 ≤ `C` ≤ `65,536`
 - `x`, `residual`, `weight`, and `out` are `float32`
 - Values in `x` and residual are in the range `[−10, 10]`
 - weight values are in the range `[0.5, 1.5]`
 - `eps = 1e-5`
 - Performance is measured with `N = 4,096`, `C = 4,096`

## `solve.cu`
```
#include <cuda_runtime.h>

// x, residual, weight, out are device pointers
extern "C" void solve(const float* x, const float* residual, const float* weight, float* out, int N,
                      int C, float eps) {}

```
