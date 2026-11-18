## Block-Sparse KV Selection Attention

Implement one decode step of block-sparse attention with query-aware KV selection, the technique used by sparse-attention inference kernels such as NSA and Quest. 
Reading the whole KV cache for every generated token is bandwidth-bound, so the cache is split into contiguous blocks of `block_size` positions, each block is summarized by its mean-pooled key, and only the `num_selected` blocks whose summary looks most relevant to the current query are read at full resolution. 
Attention is then computed over the positions of the selected blocks only.

For each head `h`, with `num_blocks = ceil(seq_len / block_size)` and block `b` covering positions `[b * block_size, min((b + 1) * block_size, seq_len))`:

<img width="1086" height="357" alt="image" src="https://github.com/user-attachments/assets/25846bbd-adb2-4e89-b893-1b4d3fc8f1d9" /> 

Each head selects its own blocks. The tail block may be partial when `seq_len` is not a multiple of `block_size`, and its mean-pooled key must average only the positions that exist. 
If num_selected is at least `num_blocks` every block is kept and the attention becomes dense. Selection must be deterministic: when two blocks have equal scores, the block with the smaller index is selected. All tensors are `float32`.

<img width="1298" height="535" alt="image" src="https://github.com/user-attachments/assets/4a0374a7-bed4-4756-b84f-230d723644ff" />


## Implementation Requirements
 - Implement the function `solve(Q, K, V, output, num_heads, seq_len, head_dim, block_size, num_selected)`.
 - Do not change the function signature or use external libraries beyond the standard GPU framework.
 - `Q` has shape `(num_heads, head_dim)`: a single decode query per head.
 - `K` and V have shape `(num_heads, seq_len, head_dim)`.
 - Score block `b` of head `h` with the unscaled dot product `Q[h] · mean(K[h, t] for t in block b)`, and keep the `min(num_selected, num_blocks)` highest-scoring blocks, breaking ties toward the smaller block index.
 - Compute a single softmax attention per head over the positions of the selected blocks, using the scale factor `1 / sqrt(head_dim)`.
 - Write the result into the provided output buffer of shape `(num_heads, head_dim)`.

## Example
With `num_heads = 2`, `seq_len = 4`, `head_dim = 2`, `block_size = 2`, `num_selected = 1`:

<img width="1284" height="547" alt="image" src="https://github.com/user-attachments/assets/516bee4d-0ffe-4a77-91d5-a24e303d5131" />
 
The pooled keys are `[1, 0]` for block 0 (positions 0–1) and `[0, 1]` for block 1 (positions 2–3). Head 0 scores them `[1, 0]` and keeps block 0, so it attends to positions 0 and 1 with equal weights and returns `([1, 2] + [3, 4]) / 2 = [2, 3]`. 
Head 1 scores them `[0, 1]` and keeps block 1, returning `([5, 6] + [7, 8]) / 2 = [6, 7]`.

## Constraints
 - 1 ≤ `num_heads` ≤ `64`
 - 1 ≤ `seq_len` ≤ `8,192`
 - 1 ≤ `head_dim` ≤ `128`
 - 1 ≤ `block_size` ≤ `256`
 - 1 ≤ `num_selected`; if `num_selected ≥ num_blocks` every block is selected
 - All tensor values are `float32`
 - Performance is measured with `num_heads = 32`, `seq_len = 8,192`, `head_dim = 128`, `block_size = 64`, `num_selected = 32`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int num_heads,
                      int seq_len, int head_dim, int block_size, int num_selected) {}

```
