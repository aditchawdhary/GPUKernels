## Paged Chunked Prefill Attention

Implement the attention step of a chunked-prefill LLM serving batch over a paged KV cache with grouped-query attention `(GQA)`. 
Each sequence in the batch contributes a chunk of its newest `q_len` query tokens: a decode step `(q_len = 1)`, a fresh prompt, or the next chunk of a long prompt. 
The keys and values for all `context_lens[s]` tokens of sequence s (including the chunk itself, already appended) live in fixed-size blocks scattered through a shared cache pool and are located through block_table. 
Every query token attends causally to its own sequence's cache. All tensors are `float32` except the index tensors, which are `int32`.

<img width="1205" height="290" alt="image" src="https://github.com/user-attachments/assets/9ade869b-c09b-422f-9b3b-99ac30d88410" />


## Implementation Requirements
Implement `solve(Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, output, num_seqs, total_q, num_q_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq)`:

`Q`: shape `(total_q, num_q_heads, head_dim)`, the query chunks of all sequences packed back to back
`K_cache`, `V_cache`: shape `(num_blocks, block_size, num_kv_heads, head_dim)`, the shared paged cache pool
`block_table`: shape `(num_seqs, max_blocks_per_seq)`; `block_table[s, j]` is the physical block holding logical block `j` of sequence `s`
`context_lens`: shape `(num_seqs,)`, the number of cached tokens of each sequence, including its chunk
`cu_seqlens_q`: shape `(num_seqs + 1,)`, with `cu_seqlens_q[0] = 0`; rows cu_seqlens_q[s] to cu_seqlens_q[s + 1] - 1 of Q and output belong to sequence s, so q_len[s] = cu_seqlens_q[s + 1] - cu_seqlens_q[s]
`output`: shape (`total_q`, `num_q_heads`, `head_dim`)

<img width="1217" height="253" alt="image" src="https://github.com/user-attachments/assets/fc6627dd-87c9-4322-b9f8-4312fe303f7f" />


Use only native features of the framework you choose (no external libraries); keep the function signature unchanged
Write the result into output; every row of output must be written
Compute scores and accumulate in float32 or better; reduced-precision (TF32/FP16) products of Q and K exceed the tolerance

## Example
Input: `num_seqs = 1`, `total_q = 2`, `num_q_heads = 2`, `num_kv_heads = 1`, `head_dim = 2`, `block_size = 2`, `max_blocks_per_seq = 2`, `context_lens = [3]`, `cu_seqlens_q = [0, 2]`, `block_table = [[2, 0]]`

The chunk is the last 2 of 3 tokens, so query 0 is at position 1 and query 1 is at position 2. Logical block 0 (positions 0, 1) is physical block 2 and logical block 1 (position 2) is physical block 0. Physical block 1 and slot 1 of block 0 are unused and hold 9s. Both query heads share KV head 0:
 
<img width="1223" height="321" alt="image" src="https://github.com/user-attachments/assets/b77f5912-17b7-49bf-9a08-d16bd7e53e97" />
 
 

## Constraints
 - 1 ≤ `num_seqs` ≤ 64
 - 1 ≤ `q_len[s]` ≤ `context_lens[s]` ≤ 8,192 for every sequence; `total_q` = `cu_seqlens_q[num_seqs]` ≤ 8,192
 - 1 ≤ `num_kv_heads` ≤ `num_q_heads` ≤ 64, and `num_q_heads` is a multiple of `num_kv_heads`
 - 2 ≤ `head_dim` ≤ 256, `head_dim` is even
 - 1 ≤ `block_size` ≤ 64, `block_size` is a power of 2
 - `max_blocks_per_seq = ceil(max(context_lens) / block_size)`; entries `block_table[s, j]` with `j ≥ ceil(context_lens[s] / block_size)` are padding and must be ignored, and slots past `context_lens[s]` in a sequence's last block hold unrelated data
 - Physical blocks are scattered arbitrarily through the pool, and the pool holds blocks that belong to no sequence
 - Inputs are drawn from a standard normal distribution; Q may be scaled so that scores reach about ±100
 - Performance is measured with `num_seqs = 32` (4 prefill chunks with `q_len = 512` and `context_lens = 512, 1,536, 2,560, 4,096`, plus 28 decode sequences with `q_len = 1` and `context_lens = 1,024 to 4,480` in steps of 128), `total_q = 2,076`, `num_q_heads = 32`, `num_kv_heads = 8`, `head_dim = 128`, `block_size = 16`

```
#include <cuda_runtime.h>

// Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, output are device pointers
extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, const int* cu_seqlens_q,
                      float* output, int num_seqs, int total_q, int num_q_heads, int num_kv_heads,
                      int head_dim, int block_size, int max_blocks_per_seq) {}
```
