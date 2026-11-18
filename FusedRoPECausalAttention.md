## Fused RoPE Causal Attention

Implement causal **grouped-query attention with rotary positional embeddings** (RoPE) fused in, as in the attention layers of Llama, Qwen and Mistral. 
Before the scores are computed, every query and key vector is rotated by an angle that depends on its position, using precomputed `cos` / `sin` tables, so the attention score between two tokens depends on how far apart they are. 
Each query attends to itself and all earlier tokens, and several query heads share each key/value head. All tensors are `float32`.

<img width="687" height="167" alt="image" src="https://github.com/user-attachments/assets/ccb4c624-aa63-4ab6-82e4-3065e8411e73" />

## Implementation Requirements
Implement `solve(Q, K, V, cos, sin, output, num_q_heads, num_kv_heads, seq_len, head_dim)`:

 - `Q`: shape (num_q_heads, seq_len, head_dim), queries before RoPE
 - `K`, `V`: shape (num_kv_heads, seq_len, head_dim), keys before RoPE and values (values are not rotated)
 - `cos`, `sin`: shape (seq_len, head_dim / 2), the rotation table for each position and frequency
 - `output`: shape (num_q_heads, seq_len, head_dim), the attention output

RoPE uses the rotate-half (Llama / GPT-NeoX) layout: with <img width="143" height="28" alt="image" src="https://github.com/user-attachments/assets/f62b9db7-4df3-4ff0-892f-f0252fea09f3" />, 
element d of the first half is paired with element `d + m`, for <img width="93" height="28" alt="image" src="https://github.com/user-attachments/assets/4fd7cee7-c32f-41a6-aeed-b3f909e662fe" /> . For a vector `x` at position `p`:

<img width="1021" height="39" alt="image" src="https://github.com/user-attachments/assets/2e0396ef-25bc-4443-8528-2db1d4886a5a" />

Query head h uses key/value head `g = h / (num_q_heads / num_kv_heads)` (integer division), so consecutive query heads share a key/value head. For query position `i` and key position `j ≤ i`:

<img width="1028" height="73" alt="image" src="https://github.com/user-attachments/assets/0ad2287a-ad0c-4628-90b2-4a79d42fcd70" />

 
Query `i` is rotated with row `i` of the tables and key `j` with row `j`. Later keys `(j > i)` get zero weight.

 - Use only native features of the framework you choose (no external libraries); keep the function signature unchanged
 - Write the result into `output`; every entry of `output` must be written, and the inputs must not be modified
 - Accumulate dot products and the softmax in `float32` or better

## Example
Input: `num_q_heads` = 1, `num_kv_heads` = 1, `seq_len` = 3, `head_dim` = 4. Pair 0 (elements 0 and 2) turns by 90° per position and pair 1 (elements 1 and 3) does not turn:
 
 <img width="1029" height="273" alt="image" src="https://github.com/user-attachments/assets/67559cf5-9a7b-459d-81c6-af57730d56b3" />


## Constraints
 - All tensors are `float32`
 - `1 ≤ num_kv_heads ≤ num_q_heads ≤ 32`, and `num_q_heads` is divisible by `num_kv_heads`
 - 1 ≤ `seq_len` ≤ 4,096; `2 ≤ head_dim ≤ 128`, and `head_dim` is even
 - -1 ≤ `cos`, `sin` ≤ 1
 - Performance is measured with `num_q_heads = 16`, `num_kv_heads = 4`, `seq_len = 4,096`, `head_dim = 6`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, cos, sin, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* cos,
                      const float* sin, float* output, int num_q_heads, int num_kv_heads,
                      int seq_len, int head_dim) {}

```
