## Decaying Causal Attention

Implement decaying causal attention. Given query matrix `Q`, key matrix `K`, and value matrix `V`, each of shape `seq_len × d_model`, and a scalar decay factor `gamma ∈ (0, 1]`, compute the unnormalized causal attention output where position `n` attends to all past positions `m ≤ n` with weight `gamma^(n−m)`:
<img width="986" height="77" alt="image" src="https://github.com/user-attachments/assets/1eac4271-2b1d-4aa5-a2c7-f6101bdf00a4" />

 

Unlike standard softmax attention, there is no normalization — the weights decay geometrically from the current position backward. This is the parallel form of the Retention mechanism (RetNet), used as a recurrence-friendly alternative to attention in sequence models.
<img width="983" height="270" alt="image" src="https://github.com/user-attachments/assets/3d25a222-60c2-4f7c-b418-d35b81881e22" />


## Implementation Requirements
 - Implement the solve function; do not change its signature.
 - Do not use external libraries beyond those provided.
 - Write the result into output.

## Example
<img width="999" height="583" alt="image" src="https://github.com/user-attachments/assets/4b8f607e-b729-442c-ac35-473ed1cc0775" />


## Constraints
 - 1 ≤ `seq_len` ≤ 8,192
 - 1 ≤ `d_model` ≤ 256
 - 0 < `gamma` ≤ 1
 - All tensors are `float32` on GPU.
 - Performance is measured with `seq_len = 4,096`, `d_model = 64`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma) {}

```
