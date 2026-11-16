## Split-KV Attention Reduction

When decoding a token against a long KV cache, Flash-Decoding splits the KV cache into num_splits chunks and computes a partial attention output on each chunk in parallel. 
Each partial output `partial_out[s]` is the softmax of that chunk's scores applied to that chunk's values, and each `partial_lse[s]` is the log-sum-exp of the chunk's scores. 
This kernel merges the per-chunk results into the final attention output using the log-sum-exp trick, so no chunk's contribution overflows when scores span very different magnitudes.

<img width="800" height="302" alt="image" src="https://github.com/user-attachments/assets/97c26c59-ccb3-41d7-8c64-21317eeed04e" />


<img width="1211" height="249" alt="image" src="https://github.com/user-attachments/assets/60131534-c216-4a14-a76c-5c32efeac65e" />

 

## Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - `partial_out` has shape `(num_splits, num_heads, head_dim)` with `float32` values
 - `partial_lse` has shape `(num_splits, num_heads)` with `float32` values
 - Subtract `m_h` before exponentiating to keep the merge numerically stable
 - The final result must be stored in the output tensor with shape (num_heads, head_dim)

## Example 1:
<img width="1208" height="360" alt="image" src="https://github.com/user-attachments/assets/48d5cf8c-0e89-4012-a65e-aa6cf1db8df8" />

## Constraints
 - 1 ≤ `num_splits` ≤ 256
 - 1 ≤ `num_heads` ≤ 128
 - 1 ≤ `head_dim` ≤ 256
 - All tensors are `float32`
 - Performance is measured with `num_splits` = 128, `num_heads` = 64, `head_dim` = 128

## `solve.cu`
```
#include <cuda_runtime.h>

// partial_out, partial_lse, output are device pointers
extern "C" void solve(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim) {}

```
