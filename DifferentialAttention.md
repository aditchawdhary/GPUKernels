## Differential Attention

Implement the causal differential attention layer of the Differential Transformer. Every head splits its query and key into two halves, computes two causal softmax attention maps, 
and attends with their difference `A1 - λ·A2`, which cancels attention noise that both maps share. The per-head result is RMS-normalized over its `2 × head_dim` features and scaled by `1 - lambda_init`. All tensors are `float32`.

<img width="1022" height="380" alt="image" src="https://github.com/user-attachments/assets/77e7b62f-86e1-4c37-a333-476ebe46afbd" />

## Implementation Requirements
Implement `solve(Q, K, V, lambdas, output, seq_len, num_heads, head_dim, lambda_init)`:

 - `Q`, `K`, `V`: shape `(seq_len, num_heads, 2 * head_dim)`
 - `lambdas`: shape `(num_heads,)`, the per-head weight `λh` of the second map
 - `output`: shape `(seq_len, num_heads, 2 * head_dim)`
 - `lambda_init`: a scalar; the normalized output is multiplied by `1 - lambda_init`

<img width="1213" height="305" alt="image" src="https://github.com/user-attachments/assets/8edbdef1-359f-48ab-a4a7-66a6d13b7de6" />

 - Use only native features of the framework you choose (no external libraries); keep the function signature unchanged
 - Write the result into `output`; every element must be written
 - Compute scores and accumulate in `float32` or better; reduced-precision (`TF32/FP16`) products exceed the tolerance

## Example
Input: `seq_len = 2`, `num_heads = 1`, `head_dim = 1`, `lambdas = [0.5]`, `lambda_init = 0.2`. Each row below is one token's `2 × head_dim` features for head 0:
  
<img width="1028" height="317" alt="image" src="https://github.com/user-attachments/assets/ac992966-e153-46f8-b802-98bdead005dd" />

 

## Constraints
 - 1 ≤ `seq_len` ≤ 8,192
 - 1 ≤ `num_heads` ≤ 64
 - 1 ≤ `head_dim` ≤ 128
 - -1 ≤ `lambdas[h]` ≤ 1 and 0 ≤ `lambda_init` < 1
 - `Q`, `K`, `V` are drawn from a standard normal distribution; `Q` may be scaled so that scores reach about `±30`
 - When `O[i]` is all zeros the output row is all zeros
 - Performance is measured with `seq_len = 4,096`, `num_heads = 32`, `head_dim = 64`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, lambdas, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* lambdas,
                      float* output, int seq_len, int num_heads, int head_dim, float lambda_init) {}

```
