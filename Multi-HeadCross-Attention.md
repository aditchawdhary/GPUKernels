## Multi-Head Cross-Attention

Implement multi-head cross-attention, the building block used in encoder-decoder transformers such as T5, BART, and Whisper, as well as the text-conditioning layers of diffusion models like Stable Diffusion. 
Given decoder queries `Q` of shape `(M, H, D)` and encoder keys `K` / values `V` of shape `(N, H, D)`, compute, for each head `h`, scaled dot-product attention where every one of the `M` queries attends to all `N` encoder positions:

<img width="1130" height="108" alt="image" src="https://github.com/user-attachments/assets/04ee2e30-bd02-4c4a-83ae-4b9587215735" />

There is no causal mask — encoder positions are fully visible to every decoder query — and the query length `M` may differ from the key/value length `N`. All tensors use `float32`.

<img width="1130" height="364" alt="image" src="https://github.com/user-attachments/assets/76a85afc-53cf-455b-aea0-fb2191aa9ddc" />

## Implementation Requirements
 - Implement the function `solve(Q, K, V, output, M, N, H, D)`.
 - Do not change the function signature or use external libraries beyond the standard GPU framework.
 - Write the result into the provided `output` buffer of shape `(M, H, D)`.
 - Use scaled dot-product attention with scale factor `1 / sqrt(D)` and a softmax over the key dimension.
 - No causal mask is applied — every query attends to all `N` encoder positions.

## Example
With `M = 2`, `N = 3`, `H = 2`, `D = 2`. Tensors are laid out as (position, head, dim). Per-head views:

<img width="1137" height="584" alt="image" src="https://github.com/user-attachments/assets/9955da85-fe7c-4eed-a6ce-5d5b563e99b6" />

 

## Constraints
 - 1 ≤ `M` ≤ 4,096
 - 1 ≤ `N` ≤ 4,096
 - 1 ≤ `H` ≤ 64
 - 1 ≤ `D` ≤ 256
 - All tensor values are `float32`
 - Performance is measured with `M = 1,024`, `N = 2,048`, `H = 16`, `D = 128`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int H, int D) {}

```
