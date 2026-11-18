## Multi-Head Latent Attention Decode

Implement one decode step of Multi-Head Latent Attention (MLA), the attention variant used by DeepSeek-V2/V3. 
Instead of caching a full key and value vector per head, MLA caches a single low-rank latent vector `c` of size `kv_lora_rank` per position, shared by every head, plus a small rotary key `k_pe` of size `rope_dim` that is also shared. 
Keys and values are reconstructed on the fly from the latent through the per-head up-projections `W_UK` and `W_UV`.

Materializing every key and value would defeat the purpose of the compressed cache, so production kernels use weight absorption: fold W_UK into the query and run the whole attention in latent space. 
For a single decode query with per-head query `q = [q_nope | q_pe]` and cache row `kv_cache[s] = [c[s] | k_pe[s]]`:

<img width="1157" height="184" alt="image" src="https://github.com/user-attachments/assets/69bf6d1e-81e9-4f83-adfb-ea2064dc1b96" />
 
 

Note that `c` plays the role of both key and value, and that the rotary term uses the same `k_pe` for every head. All tensors are `float32`.

<img width="1147" height="491" alt="image" src="https://github.com/user-attachments/assets/7bf1dc17-e10c-420b-9549-44a7a30698a5" />

## Implementation Requirements
 - Implement the function `solve(q, kv_cache, W_UK, W_UV, output, num_heads, seq_len, kv_lora_rank, head_dim, rope_dim)`.
 - Do not change the function signature or use external libraries beyond the standard GPU framework.
 - `q` has shape `(num_heads, head_dim + rope_dim)`: the first `head_dim` entries of each row are `q_nope`, the remaining rope_dim entries are `q_pe`.
 - `kv_cache` has shape `(seq_len, kv_lora_rank + rope_dim)`: the first `kv_lora_rank` entries of each row are the latent `c`, the remaining `rope_dim` entries are `k_pe`.
 - `W_UK` has shape `(num_heads, head_dim, kv_lora_rank)` and `W_UV` has shape `(num_heads, kv_lora_rank, head_dim)`.
 - Use the scale factor `1 / sqrt(head_dim + rope_dim)` and a softmax over the `seq_len` cached positions.
 - Write the result into the provided output buffer of shape `(num_heads, head_dim)`.

## Example
With `num_heads` = 2, `seq_len` = 2, `kv_lora_rank` = 2, `head_dim` = 2, `rope_dim` = 1:

<img width="1154" height="468" alt="image" src="https://github.com/user-attachments/assets/feec0564-9896-441d-bcb5-0340e8e96908" />

Output `(2×2, values rounded to 4 decimals)`:
<img width="1121" height="95" alt="image" src="https://github.com/user-attachments/assets/70e4f745-cf02-44a8-b437-264798fb58aa" />
 

Head 0 scores are equal, so its attention is uniform and the latent result is `[0.5, 0.5]`. Head 1 sees scores `[0.5774, 0]`, giving weights `[0.6405, 0.3595]`, which pass through the identity `W_UV` unchanged.

## Constraints
 - 1 ≤ `num_heads` ≤ 128
 - 1 ≤ `seq_len` ≤ 8,192
 - 1 ≤ `kv_lora_rank` ≤ 512
 - 1 ≤ `head_dim` ≤ 128
 - 1 ≤ `rope_dim` ≤ 64
 - All tensor values are `float32`
 - Performance is measured with `num_heads` = 128, `seq_len` = 4,096, `kv_lora_rank` = 512, `head_dim` = 128, `rope_dim` = 64
