## Shared-Prefix Cascade Attention

Implement shared-prefix (cascade) attention, used when many sequences in a batch share the same prompt prefix (e.g. a long system prompt or few-shot examples). 
All `batch_size` sequences share one cached key/value prefix of prefix_len tokens, stored only once, and each sequence also has its own `suffix_len` suffix tokens. 
Each query token attends to every prefix token and causally to its own sequence's suffix tokens. All tensors are `float32`. 
A naive kernel re-reads the shared prefix once per sequence. A fast kernel reads it once for the whole batch and merges the prefix and suffix softmax partial results.

For sequence `b`, head `h` and suffix query `i`:
<img width="986" height="80" alt="image" src="https://github.com/user-attachments/assets/2993f8e7-7254-457f-8a90-ee4912132fe9" />

 

## Implementation Requirements
 - Implement `solve(Q, K_prefix, V_prefix, K_suffix, V_suffix, output, batch_size, num_heads, prefix_len, suffix_len, head_dim)`.
 - Do not change the function signature or use external libraries beyond the standard GPU frameworks.
 - Write the result into output.
 - Shapes (row-major, contiguous): `Q`, `K_suffix`, `V_suffix`, `output` are `(batch_size, num_heads, suffix_len, head_dim)`; `K_prefix`, `V_prefix` are `(num_heads, prefix_len, head_dim)` and are shared by every sequence.
 - Use scale `1 / sqrt(head_dim)`. Suffix query i sees all prefix keys and suffix keys `0..i` (inclusive) of its own sequence only; it never sees other sequences' suffixes.
 - `prefix_len` may be `0`, in which case this is plain causal attention over the suffix.

## Example
<img width="1014" height="359" alt="image" src="https://github.com/user-attachments/assets/272605e3-8e7f-4524-ae02-541723ac35c1" />


## Constraints
 - 1 ≤ `batch_size` ≤ 64
 - 1 ≤ `num_heads` ≤ 64
 - 0 ≤ `prefix_len` ≤ 8,192
 - 1 ≤ `suffix_len` ≤ 1,024
 - 1 ≤ `head_dim` ≤ 128
 - All values are `float32`; tolerance is `1e-4` absolute and relative
 - Performance is measured with `batch_size = 16`, `num_heads = 32`, `prefix_len = 4,096`, `suffix_len = 256`, `head_dim = 128`

## `solve.cu`
```cuda
#include <cuda_runtime.h>

// Q, K_prefix, V_prefix, K_suffix, V_suffix, output are device pointers
extern "C" void solve(const float* Q, const float* K_prefix, const float* V_prefix,
                      const float* K_suffix, const float* V_suffix, float* output, int batch_size,
                      int num_heads, int prefix_len, int suffix_len, int head_dim) {}

```
