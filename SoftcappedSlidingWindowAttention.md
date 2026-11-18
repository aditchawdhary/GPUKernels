## Softcapped Sliding Window Attention

Implement causal sliding window attention with logit soft-capping and grouped query attention, as used in the local attention layers of Gemma 2. 
Each query attends only to itself and the previous window_size - 1 tokens, and every attention score is squashed by softcap · tanh(score / softcap) before the softmax. Several query heads share each key/value head. All tensors are `float32`.

<img width="309" height="304" alt="image" src="https://github.com/user-attachments/assets/38c31fdd-6cbf-4c7f-90ab-be7e48263298" />


## Implementation Requirements
Implement `solve(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim, window_size, softcap)`:

 - `Q`: shape `(num_q_heads, seq_len, head_dim)`, queries
 - `K`, `V`: shape `(num_kv_heads, seq_len, head_dim)`, keys and values
 - `output`: shape `(num_q_heads, seq_len, head_dim)`, the attention output
 - `window_size`: number of keys each query can see, counting itself; `softcap`: the logit cap
Query head `h` uses key/value head `g = h / (num_q_heads / num_kv_heads)` (integer division), so consecutive query heads share a key/value head. For query position `i` and key position `j`:
<img width="1179" height="168" alt="image" src="https://github.com/user-attachments/assets/fb1eb13b-2515-4487-a795-18813a04bc91" />
 
 
 
The `1/sqrt(head_dim)` scale is applied before the cap. Keys outside the window (later tokens, or tokens more than `window_size - 1` positions back) get zero weight. 
If `window_size ≥ seq_len` this is ordinary causal attention with soft-capping.

 - Use only native features of the framework you choose (no external libraries); keep the function signature unchanged
 - Write the result into `output`; every entry of `output` must be written
 - Accumulate dot products and the softmax in `float32` or better

## Example
Input: `num_q_heads = 2`, `num_kv_heads = 1`, `seq_len = 3`, `head_dim = 2`, `window_size = 2`, `softcap = 1.0`: 
 
<img width="1171" height="283" alt="image" src="https://github.com/user-attachments/assets/ff295839-8fb5-4844-9467-6fd9ae295011" />
 
 
Row 2 of each head sees only keys 1 and 2, because key 0 is outside the window of size 2.

## Constraints
 - All tensors are `float32`
 - 1 ≤ `num_kv_heads` ≤ `num_q_heads` ≤ 32, and `num_q_heads` is divisible by `num_kv_heads`
 - 1 ≤ `seq_len` ≤ 4,096; 1 ≤ `head_dim` ≤ 128
 - 1 ≤ `window_size` ≤ 8,192 (it may exceed `seq_len`)
 - 0 < `softcap` ≤ 50.0
 - Performance is measured with `num_q_heads = 16`, `num_kv_heads = 4`, `seq_len = 4,096`, `head_dim = 64`, `window_size = 1,024`, `softcap = 10.0`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim, int window_size,
                      float softcap) {}

```
