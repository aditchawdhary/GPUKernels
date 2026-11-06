### Paged KV-Cache Attention

Implement decode-phase attention over a **paged KV cache**. In LLM serving systems (e.g., vLLM), the key and value tensors for each sequence are stored in fixed-size memory blocks (pages) that may be scattered non-contiguously across a shared GPU memory pool. 
A `block_table` maps each sequence's logical block indices to physical block indices in the cache pool. 
Given a single query vector per sequence (one new token being generated), compute the attention output by gathering the relevant K/V blocks via the block table and computing scaled dot-product attention over the full context.

<img width="943" height="463" alt="image" src="https://github.com/user-attachments/assets/c4ea1ce2-641f-4077-91be-08c536c7047f" />


#### Implementation Requirements
Implement the function `solve(Q, K_cache, V_cache, block_table, context_lens, output, batch_size, num_heads, head_dim, block_size, max_blocks_per_seq)` that computes paged decode-phase attention:

 - `Q`: query tensor of shape `(batch_size, num_heads, head_dim)`, dtype `float32` — one query per sequence
 - `K_cache`: paged key cache of shape `(num_blocks, block_size, num_heads, head_dim)`, dtype `float32`
 - `V_cache`: paged value cache of shape `(num_blocks, block_size, num_heads, head_dim)`, dtype `float32`
 - `block_table`: physical block indices of shape `(batch_size, max_blocks_per_seq)`, dtype `int32`
 - `context_lens`: number of valid KV tokens per sequence, shape `(batch_size,)`, dtype `int32`
 - `output`: result of shape `(batch_size, num_heads, head_dim)`, dtype `float32`

For each sequence `s`and each attention head `h`, compute:

<img width="971" height="224" alt="image" src="https://github.com/user-attachments/assets/9eb5c7b8-c1c5-4d89-b12c-b72ae312b651" />

Do not use external libraries beyond the framework you select. Keep the function signature unchanged. Write results directly into output.

#### Example


<img width="984" height="497" alt="image" src="https://github.com/user-attachments/assets/e4b716dd-00fd-4eaf-b910-8c8e7fd367bb" />


### Constraints
 - 1 ≤ `batch_size` ≤ 32
 - 1 ≤ `num_heads` ≤ 64
 - 1 ≤ `head_dim` ≤ 256; `head_dim` is a multiple of 4
 - 1 ≤ `block_size` ≤ 64; `block_size` is a power of 2
 - 1 ≤ `context_lens[s]` ≤ 8,192 for all sequences `s`
 - `max_blocks_per_seq` = `ceil(max(context_lens) / block_size)`
 - All input tensors are on the GPU and in `float32` (except `block_table` and `context_lens` which are `int32`)
 - `block_table[s, i]` is a valid index into the first dimension of `K_cache` for all `i < ceil(context_lens[s] / block_size);` physical blocks are scattered arbitrarily through the pool, and entries at or beyond that index are padding that must be ignored
 - The cache pool may contain blocks that belong to no sequence; only blocks reachable through block_table may be read
 - Performance is measured with `batch_size` = 8, `num_heads` = 32, `head_dim` = 128, `block_size` = 16, `context_lens[s]` = 2,048 for all sequences
