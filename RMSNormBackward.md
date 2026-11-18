## RMS Norm Backward

Implement the backward pass of RMS Normalization, the normalization layer used in every block of LLaMA, Mistral, and Qwen. 
The forward pass normalized each row (token) of the `float32` input `x` of shape `(N, C)` and scaled it by a learned per-feature weight of shape `(C,)`. 
Given `x`, weight, and the upstream gradient `dy` of shape `(N, C)`, compute the input gradient `dx` of shape `(N, C)` and the weight gradient dweight of shape `(C,)`. 
The per-row statistic is not saved from the forward pass and must be recomputed.

<img width="1020" height="740" alt="image" src="https://github.com/user-attachments/assets/644eec38-3dab-4b23-a3c3-acf48a0ecf47" />

 
 
 

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The results must be stored in dx and dweight
 - `dx` and `dweight` are not zero-initialized; every element must be written

## Example
<img width="1303" height="329" alt="image" src="https://github.com/user-attachments/assets/730dfc6e-7d39-4fe9-85e2-3811b09e320c" />

 

## Constraints
 - 1 ≤ `N` ≤ `65,536`
 - 1 ≤ `C` ≤ `16,384`
 - All tensors are `float32`
 - Values of `x` are in `[−250, 250]`; `values` of `dy` are in `[−10, 10]`; values of `weight` are in `[−2, 2]`
 - `1e-6` ≤ `eps` ≤ 1.0
 - Performance is measured with `N` = `16,384`, `C` = `4,096`

## `solve.cu`
```
#include <cuda_runtime.h>

// x, weight, dy, dx, dweight are device pointers
extern "C" void solve(const float* x, const float* weight, const float* dy, float* dx,
                      float* dweight, int N, int C, float eps) {}

```
