## Speculative Decoding Verification

Implement the token verification step of speculative decoding. A draft model proposes `T` tokens; the target model evaluates them in one forward pass and accepts or rejects each. Given `B` 
sequences, produce the verified output tokens. Probability tensors are `float32`; token tensors are `int32`.

Notation for each sequence `b`, at each draft position `i = 0,...,T-1`:
<img width="1029" height="132" alt="image" src="https://github.com/user-attachments/assets/3a084bb1-e034-4db6-845a-5f98b656c8b3" />

<img width="1005" height="433" alt="image" src="https://github.com/user-attachments/assets/53c15624-7449-46e6-94c9-d4ed2a822766" />


For each sequence `b`, process positions `i = 0, 1, ..., T-1` left-to-right:
<img width="1015" height="327" alt="image" src="https://github.com/user-attachments/assets/7449ec51-83aa-4e4a-aa08-3427fdbc68d5" />

## Implementation Requirements
 - Implement solve(draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, B, T, V).
 - Do not change the function signature or use external libraries beyond the standard GPU frameworks.
 - Write results into the provided output_tokens buffer (shape [B, T+1], int32).
 - Memory layout is row-major: draft_probs[b, i, v] is at offset b*T*V + i*V + v.
 - <img width="998" height="54" alt="image" src="https://github.com/user-attachments/assets/fe742edd-0d61-46cb-a49d-58f6289ebee1" />
 - If the adjusted distribution is all zeros (i.e., <img width="62" height="22" alt="image" src="https://github.com/user-attachments/assets/25dbea5b-9e02-4fa9-ab06-f3e0cc4f11f0" /> everywhere), fall back to the uniform distribution over `V` tokens.

## Example
<img width="1024" height="619" alt="image" src="https://github.com/user-attachments/assets/b1df0642-7701-4503-80a7-641596c1eee8" />


## Constraints
 - 1 ≤ `B` ≤ 256
 - 1 ≤ `T` ≤ 16
 - 2 ≤ `V` ≤ 131,072
 - `draft_probs[b, i, :]` and `target_probs[b, i, :]` are valid probability distributions (non-negative, sum to 1)
 - `draft_probs[b, i, draft_tokens[b, i]]` > 0 for all b, i
 - `uniform_samples` values are in `[0,1)`
 - All floating-point tensors use float32; token tensors use `int32`
 - Performance is measured with `B = 64`, `T = 8`, `V = 32,768`
