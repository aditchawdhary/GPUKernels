## Beam Search Step

Implement one decoding step of beam search for autoregressive language models. Given the cumulative log-probabilities of K active beams in each of B independent sequences and the next-token log-probabilities for every token in the vocabulary from each beam, expand all K × V candidate continuations and select the top K candidates per sequence to form the next set of beams.

For each batch `b` and candidate `(k, v)`, the score is
<img width="907" height="53" alt="image" src="https://github.com/user-attachments/assets/35951ec4-395d-4941-b3c1-32e60f10b9fd" />

Per batch, the top `K` candidates by score are selected and written, sorted in descending order by score, into new_beam_scores. 
For each selected candidate, the originating beam index `k` is written to `parent_beam_indices` and the chosen vocabulary token `v` to `next_tokens`.

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The `solve` function signature must remain unchanged
 - The final results must be stored in new_beam_scores, parent_beam_indices, and next_tokens
 - Within each batch, the `K` outputs must be sorted in descending order by score; `parent_beam_indices` and `next_tokens` must correspond positionally to `new_beam_scores`
 - All `K × V` candidate scores within a batch are distinct in the test data, so no tie-breaking is required; the behavior on exact ties is unspecified

## Example 1:
```
Input:
  beam_scores = [[-0.5, -1.0]]
  token_logprobs = [[[-0.3, -1.2, -2.0, -2.5],
                     [-0.4, -0.1, -1.6, -3.2]]]
  B = 1, K = 2, V = 4

Output:
  new_beam_scores = [[-0.8, -1.1]]
  parent_beam_indices = [[0, 1]]
  next_tokens = [[0, 1]]

Explanation:
Candidate scores (cand[k, v] = beam_scores[k] + token_logprobs[k, v]):
  from beam 0: [-0.8, -1.7, -2.5, -3.0]
  from beam 1: [-1.4, -1.1, -2.6, -4.2]
The two highest scores are -0.8 (beam 0, token 0) and -1.1 (beam 1, token 1).
```


## Constraints
 - 1 ≤ `B` ≤ 64 (batch size)
 - 1 ≤ `K` ≤ 32 (beam width)
 - 1 ≤ `V` ≤ 100,000 (vocabulary size)
 - `beam_scores`, `token_logprobs`, and `new_beam_scores` are `32-bit` floats
 - `parent_beam_indices` and `next_tokens` are `32-bit` integers
 - All tensors are stored on the GPU in row-major order
 - Performance is measured with `B = 16`, `K = 8`, `V = 50,000`
