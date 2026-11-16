INT8 KV-Cache Attention

Implement decode-phase multi-head attention where the key and value caches are stored as `int8` with per-token scale factors. 
This memory layout halves `KV-cache` bandwidth versus `float32` and is used in production LLM serving systems such as TensorRT-LLM and vLLM. 
Given a query tensor `Q` for a single new token, `int8` key cache `K_int8`, int8 value cache `V_int8`, and per-token scales `k_scale` and `v_scale`, `dequantize` the caches and compute scaled dot-product attention to produce `output`. 
All non-integer tensors use `float32`.

## Implementation Requirements
 - Implement the function `solve(Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim)`.
 - Do not change the function signature or use external libraries beyond the standard GPU frameworks.
 - Write the result into the provided output buffer.
 - Dequantize using per-token scales: `K_float[h, s, d] = K_int8[h, s, d] × k_scale[h, s]` (and analogously for `V`).
 - Use scaled dot-product attention with scale factor `1 / sqrt(head_dim)` and a softmax over the sequence dimension.

## Example
With `num_heads` = 1, `seq_len` = 3, `head_dim` = 4:

<img width="1044" height="822" alt="image" src="https://github.com/user-attachments/assets/c95e5e61-778f-451e-b062-caee23d47007" />


## Constraints
 - 1 ≤ `num_heads` ≤ 64
 - 1 ≤ `seq_len` ≤ 32,768
 - 8 ≤ `head_dim` ≤ 256; `head_dim` is a multiple of 8
 - `K_int8` and `V_int8` values are in `[-128, 127]`
 - All scale `values` are positive `float32`
 - Performance is measured with `num_heads = 32`, `seq_len = 8,192`, `head_dim = 128`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K_int8, V_int8, k_scale, v_scale, output are device pointers
extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {}

```
