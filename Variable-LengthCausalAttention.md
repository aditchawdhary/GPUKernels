## Variable-Length Causal Attention

Implement variable-length causal self-attention over a packed batch of sequences. The tensors `Q`, `K`, and `V` each have shape `T × d`, where `T` is the total number of tokens across all sequences concatenated end-to-end. 
The 1D tensor `cu_seqlens` of shape `S + 1` stores cumulative sequence lengths, so sequence `s` occupies rows `[cu_seqlens[s], cu_seqlens[s+1])`.

For each token at packed position `t` belonging to sequence `s` with in-sequence position `p = t − cu_seqlens[s]`, attention is restricted to tokens of the same sequence at positions `0, 1, …, p (causal)`. Concretely:

<img width="963" height="164" alt="image" src="https://github.com/user-attachments/assets/043c29f0-9875-4033-b506-7c96f2366d11" /> 
 

This is the packed-sequence attention kernel used by inference engines such as vLLM and FlashAttention’s `varlen` path. 
By packing variable-length sequences into a single tensor and indexing through `cu_seqlens`, a serving stack avoids padding waste and amortizes kernel launch overhead across requests with very different lengths.

<img width="947" height="282" alt="image" src="https://github.com/user-attachments/assets/ac7cde3b-8f0a-4bb8-885b-ab4fd5b3908c" />

## Implementation Requirements
 - Implement the solve function; do not change its signature.
 - External libraries are not permitted.
 - Write the result into output.

## Example
<img width="967" height="438" alt="image" src="https://github.com/user-attachments/assets/fc32b332-8c3f-4e75-8c5b-6a5d45752f1a" />



## Constraints
 - 1 ≤ `S` ≤ 256
 - 1 ≤ `sequence length` ≤ 2,048 (each individual sequence)
 - 1 ≤ `T` ≤ 16,384 (total packed tokens)
 - 1 ≤ `d` ≤ 256
 - cu_seqlens is monotonically non-decreasing with `cu_seqlens[0] = 0` and `cu_seqlens[S] = T`.
 - `Q`, `K`, `V`, output are `float32`; `cu_seqlens` is `int32`.
 - Performance is measured with `T = 8,192`, `d = 64`, `S = 32` (`sequence lengths` between `128` and `384`).
