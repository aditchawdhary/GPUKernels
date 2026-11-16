## Min-P Sampling

Implement batched *min-p sampling*, a logit-filtering primitive used in modern LLM serving stacks such as vLLM, Hugging Face TGI, and llama.cpp. 
Given a batch of `logits` logits of shape `B x V` and a scalar min_p in `[0, 1]`, produce a batch of filtered probabilities probs of the same shape. For each row `b`:

1. Convert logits to probabilities with softmax: <img width="210" height="24" alt="image" src="https://github.com/user-attachments/assets/c0f29532-cf2f-4894-a3c9-cf87b29d3d48" />

2. Find the maximum probability in the row: <img width="107" height="24" alt="image" src="https://github.com/user-attachments/assets/737632bc-0f38-453d-be11-a49b4c2fe19f" />

3. Mask every token whose probability is below <img width="93" height="24" alt="image" src="https://github.com/user-attachments/assets/8e5a138a-fcaa-4478-aa14-806cec9e8170" /> (set it to 0).
 
4. Renormalize the surviving probabilities so they sum to 1.

Unlike top-k or top-p sampling, min-p does not require any sort: the cutoff is defined relative to the row's most likely token, which makes it well-suited to GPU parallelization while still adapting to how peaked or flat each distribution is.

## Implementation Requirements
 - Implement `solve(logits, probs, min_p, B, V)`; do not change the signature or use external libraries beyond the standard GPU frameworks.
 - Write the result into the provided probs buffer. Every row must sum to exactly `1`; masked positions must be exactly `0`.
 - Apply `min-p` independently per row of the batch.
 - Use the numerically stable softmax (subtract the per-row max before exponentiating).

## Example
<img width="1137" height="360" alt="image" src="https://github.com/user-attachments/assets/bb0a37c8-27f8-4dcc-aa85-688e82023b28" />

 

## Constraints
 - 1 ≤ `B` ≤ 256
 - 1 ≤ `V` ≤ 200,000
 - 0.0 ≤ `min_p` ≤ 1.0
 - -50.0 ≤ `logits[b, i]` ≤ 50.0
 - Inputs and outputs are `float32`
 - Performance is measured with `B` = 64, `V` = 128,000, `min_p` = 0.05
