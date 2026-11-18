## Fused Logit Penalties

Implement the fused logit penalty stage that LLM serving stacks such as vLLM, SGLang and Hugging Face TGI run on every decoding step, right before sampling. 
Given a batch of `B` in-flight requests, a logit row of `V` vocabulary entries per request, the tokens each request has already seen, and per-request penalty coefficients, produce the penalized logits in `output`. 
Every tensor is `float32` except the token id tensors, which are `int32`.

<img width="1360" height="397" alt="image" src="https://github.com/user-attachments/assets/649cf39e-d86a-43c9-9bd7-4ac638bb0479" />

<img width="1261" height="551" alt="image" src="https://github.com/user-attachments/assets/f6ca347b-f10a-4710-a674-edb5ed958049" />


## Implementation Requirements
 - Implement `solve(logits, prompt_tokens, output_tokens, presence_penalty, frequency_penalty, repetition_penalty, output, B, V, P, G)`; do not change the signature or use external libraries beyond the standard GPU frameworks.
 - Write the penalized logits into the provided output buffer; leave logits unmodified.
 - Each request has its own presence_penalty, frequency_penalty and repetition_penalty value.
 - Apply the penalties in the order listed above; the repetition penalty must be applied before the additive penalties.
 - Token ids equal to -1 are padding slots and contribute to no count; padding may appear at any position in a row.

## Example
With `B = 2`, `V = 6`, `P = 3`, `G = 4`:
<img width="1384" height="704" alt="image" src="https://github.com/user-attachments/assets/b1a0cfa9-3c80-42c8-92ac-163783b5c851" />



## Constraints
 - 1 ≤ `B` ≤ 256
 - 1 ≤ `V` ≤ 131,072
 - 1 ≤ `P` ≤ 1,024
 - 1 ≤ `G` ≤ 512
 - Token ids are -1 (padding) or in the range 
 - -50.0 ≤ `logits[b][v]` ≤ 50.0
 - 0.0 ≤ `presence_penalty[b]`, `frequency_penalty[b]` ≤ 2.0
 - 0.5 ≤ `repetition_penalty[b]` ≤ 2.0
 - `logits`, the penalty vectors and output are `float32`; `prompt_tokens` and `output_tokens` are `int32`
 - Performance is measured with `B = 256`, `V = 128,256`, `P = 1,024`, `G = 512`

## `solve.cu`
```
#include <cuda_runtime.h>

// logits, prompt_tokens, output_tokens, presence_penalty, frequency_penalty, repetition_penalty,
// output are device pointers
extern "C" void solve(const float* logits, const int* prompt_tokens, const int* output_tokens,
                      const float* presence_penalty, const float* frequency_penalty,
                      const float* repetition_penalty, float* output, int B, int V, int P, int G) {}

```
