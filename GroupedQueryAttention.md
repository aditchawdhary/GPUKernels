#### Grouped Query Attention

Implement Grouped Query Attention (GQA), the attention mechanism used in modern large language models such as LLaMA-3, Mistral, and Gemma. GQA reduces the KV-cache memory footprint during inference by sharing key and value heads across groups of query heads. Given query tensor Q with num_q_heads heads and key/value tensors K, V each with num_kv_heads heads, compute scaled dot-product attention where every group of num_q_heads / num_kv_heads consecutive query heads attends to the same key and value head. All tensors use float32.

<img width="705" height="269" alt="image" src="https://github.com/user-attachments/assets/632cb8f4-3123-4e82-91d8-944f3947b74d" />


#### Implementation Requirements
 - Implement the function `solve(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim)`.
 - Do not change the function signature or use external libraries beyond the standard GPU frameworks.
 - Write the result into the provided `output` buffer.
 - `num_q_heads` is always divisible by `num_kv_heads`.
 - Use scaled dot-product attention with scale factor `1 / sqrt(head_dim)` and a softmax over the key dimension.

#### Example
With `num_q_heads = 4`, `num_kv_heads = 2` (groups of 2), `seq_len = 3`, `head_dim = 4`:

<img width="778" height="781" alt="image" src="https://github.com/user-attachments/assets/8665e467-c2a0-4beb-b837-f5e5ca4dbfd1" />

<img width="786" height="305" alt="image" src="https://github.com/user-attachments/assets/287a1eda-c4ee-434a-b7eb-5998dd0ef4ec" />



Output (values rounded to 2 decimal places):

<img width="762" height="523" alt="image" src="https://github.com/user-attachments/assets/d6549123-1bbf-4ebc-964f-72360749fbc4" />



#### Constraints
 - 1 ≤ num_kv_heads ≤ num_q_heads ≤ 64
 - num_q_heads is divisible by num_kv_heads
 - 1 ≤ seq_len ≤ 4,096
 - 8 ≤ head_dim ≤ 256; head_dim is a multiple of 8
 - All tensor values are float32
 - Performance is measured with num_q_heads = 32, num_kv_heads = 8, seq_len = 1,024, head_dim = 128


```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {}

```
