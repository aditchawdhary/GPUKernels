## TurboQuant KV Cache Attention

Implement attention score computation against a TurboQuant-compressed KV cache. TurboQuant compresses each key vector to `uint8` codebook indices plus a 1-bit residual correction (QJL), reducing KV cache memory by up to 6x. 
Your task: dequantize the compressed keys and compute dot-product attention scores against full-precision queries.

<img width="1072" height="800" alt="image" src="https://github.com/user-attachments/assets/8da4fd51-9b88-434d-9ac6-cb0245aa6aba" />

## Implementation Requirements
 - The solve function signature must remain unchanged.
 - Use only native features (no external libraries).
 - Store the result in scores as `float32`.

## Example

<img width="1030" height="453" alt="image" src="https://github.com/user-attachments/assets/2bbec585-4113-40cf-b93b-b6854d47c319" />

## Constraints

<img width="604" height="308" alt="image" src="https://github.com/user-attachments/assets/189767ea-2b4a-497d-a444-ce5c1fed17bc" />

```
#include <cuda_runtime.h>

// Q, K_idx, qjl_signs, gamma, Pi, S_mat, codebook, scores are device pointers
extern "C" void solve(const float* Q, const unsigned char* K_idx, const signed char* qjl_signs,
                      const float* gamma, const float* Pi, const float* S_mat,
                      const float* codebook, float* scores, int B, int S, int D, int C) {}

```
