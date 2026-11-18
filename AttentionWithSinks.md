## Attention with Sinks

Implement Attention with Sinks for a given set of matrices. This is the attention pattern used in StreamingLLM to enable long-context inference: each query attends to a small number of leading sink tokens plus a sliding window of the most recent tokens. Combining a fixed prefix (sinks) with a moving window keeps the KV cache bounded while preserving the anchor tokens that stabilize the softmax distribution.

Given query matrix `Q`, key matrix `K`, and value matrix `V`, each of shape `M × d`, compute for every query position `i`:
<img width="1132" height="106" alt="image" src="https://github.com/user-attachments/assets/caa8a094-97d8-43ac-918d-6411fc79fe17" />

 

The set of allowed key positions for query `i` is:
<img width="1082" height="62" alt="image" src="https://github.com/user-attachments/assets/21ee9481-7b1d-40db-91a8-36a8d4c94352" />


Softmax is applied only over these allowed positions and the output is the weighted sum of the corresponding values:
<img width="1032" height="643" alt="image" src="https://github.com/user-attachments/assets/a263b02c-a589-43f6-aff8-745443ac149f" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output matrix output

## Example 1:
<img width="970" height="624" alt="image" src="https://github.com/user-attachments/assets/42546e39-6777-4616-ad4c-99ae4c7fcb03" />

Output:
`output (4×4)`:
 <img width="997" height="150" alt="image" src="https://github.com/user-attachments/assets/a4cdb590-0c06-4e50-93a1-4486ca78cac5" />


`Query 3` attends to positions `{0, 2, 3}`: position `0` is a sink, positions `2` and `3` fall inside the causal window `[i - window_size + 1, i]` = `[2, 3]`, and position 1 is neither a sink nor in the window, so it is masked out.

## Constraints
 - Matrices `Q`, `K`, and `V` are all of size `M × d`
 - 1 ≤ `M` ≤ 10,000
 - 1 ≤ `d` ≤ 128
 - 1 ≤ `num_sinks` ≤ 16, and `num_sinks` ≤ M
 - 1 ≤ `window_size` ≤ M
 - All elements in `Q`, `K`, and `V` are sampled from `[-1.0, 1.0]`
 - Data type for all matrices is `float32`
 - Performance is measured with `M = 5,000`, `d = 128`, `num_sinks = 4`, `window_size = 1,024`

## References
Xiao et al., Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453, ICLR 2024) — introduces the attention sink phenomenon and the efficient LLM streaming mechanism. For a more in depth dive, please read the paper.
https://arxiv.org/abs/2309.17453 
