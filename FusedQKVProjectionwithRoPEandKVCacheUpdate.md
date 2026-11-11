## Fused QKV Projection with RoPE and KV Cache Update

Implement the fused **QKV projection + rotary embedding + KV-cache** update that every LLaMA-style attention layer runs at the start of a decoding step. 
A batch of `B` sequences each contribute a single new token, whose hidden state is row `b` of `x` (shape `[B, d_model]`). One fused weight matrix W_qkv (shape `[d_model, (H_q + 2 × H_kv) × D]`) produces the `query`, `key`, and `value` projections in a single matrix multiply. 
Rotary positional embeddings are applied to the query and key (never to the value), and the rotated key and the value are written into the slot `positions[b]` of the paged KV cache. All tensors are `float32` except positions, which is `int32`.

For each sequence b, let `qkv = x[b] × W_qkv`. The first `H_q × D` values of `qkv` are the query heads, the next `H_kv × D` values are the key heads, and the last `H_kv × D` values are the value heads (grouped-query attention uses `H_kv` ≤ `H_q`). Each group is reshaped row-major into `[heads, D]`.

Rotation uses the precomputed table `cos_sin_cache` of shape `[S_max, D]`, whose row `p` stores `D/2 cosines` followed by `D/2 sines`. With `p = positions[b]`, `c = cos_sin_cache[p, 0 : D/2]` and `s = cos_sin_cache[p, D/2 : D]`, the rotate-half transform of a head vector `t` of length `D` is:

<img width="938" height="608" alt="image" src="https://github.com/user-attachments/assets/c3d3c13c-e07a-4a0c-a428-3334a1cd56b1" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The `solve` function signature must remain unchanged
 - The rotated queries must be stored in `Q_out`, and the rotated keys and the values must be written in place into `K_cache` and `V_cache`

## Example
<img width="943" height="714" alt="image" src="https://github.com/user-attachments/assets/c9b10f1b-db20-42a1-8121-a830efcf3d13" />


## Constraints
 - 1 ≤ `B` ≤ 1,024
 - 1 ≤ `d_model` ≤ 8,192
 - 1 ≤ `H_kv` ≤ `H_q` ≤ 128, and `H_q` is a multiple of `H_kv`
 - `D` is even, 2 ≤ `D` ≤ 128
 - 1 ≤ `S_max` ≤ 4,096
 - 0 ≤ `positions[b]` < `S_max` for every `b`
 - `positions` is `int32`; every other `tensor` is `float32`
 - Values of `x`, `W_qkv`, `K_cache`, and `V_cache` are in the range `[-10, 10]`
 - Performance is measured with `B = 64`, `d_model = 4,096`, `H_q = 32`, `H_kv = 8`, `D = 128`, `S_max = 2,048`

## References
 - Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864) — introduces rotary positional embeddings.
 - Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (arXiv:2305.13245) — the grouped-query layout that makes H_kv smaller than H_q.

## `solve.cu`
```
#include <cuda_runtime.h>

// x, W_qkv, cos_sin_cache, positions, K_cache, V_cache, Q_out are device pointers
extern "C" void solve(const float* x, const float* W_qkv, const float* cos_sin_cache,
                      const int* positions, float* K_cache, float* V_cache, float* Q_out, int B,
                      int d_model, int H_q, int H_kv, int D, int S_max) {}

```
