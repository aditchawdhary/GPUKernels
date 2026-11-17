## Fused Linear Cross Entropy

Compute the mean cross-entropy loss of a language-model output head without materializing the full logits matrix. Given hidden states `X` of shape `N × D`, an output projection W of shape `V × D` (one row per vocabulary token) and integer class targets targets of length `N`, compute

 <img width="1019" height="92" alt="image" src="https://github.com/user-attachments/assets/6a2f6db9-f340-43a5-a87f-6388a0ffe09b" />
 
 
All tensors are `32-bit` floats except targets, which is `32-bit` integer. With a realistic vocabulary the `N × V` logits matrix is far larger than the inputs, so a fast solution fuses the matmul with an online (streaming) `log-sum-exp` over vocabulary tiles — the trick behind Liger/Apple "cut cross-entropy" kernels.

## Implementation Requirements
 - Do not change the function signature.
 - Do not use external libraries beyond what is available in the starter.
 - Write the scalar mean loss to `loss[0]`.
 - The computation must be numerically stable for large logits (subtract the row maximum).

## Example 1:
<img width="999" height="326" alt="image" src="https://github.com/user-attachments/assets/5ba71b16-a9d0-4e9a-8350-2baff28b9b2d" />

## Constraints
 - 1 ≤ `N` ≤ 8,192
 - 1 ≤ `D` ≤ 4,096
 - 2 ≤ `V` ≤ 65,536
 - 0 ≤ `targets[i]` < `V`
 - Logit magnitudes may reach several hundred
 - Performance is measured with `N = 4,096`, `D = 1,024`, `V = 32,768`

## `solve.cu`
```
#include <cuda_runtime.h>

// X, W, targets, loss are device pointers
extern "C" void solve(const float* X, const float* W, const int* targets, float* loss, int N, int D,
                      int V) {}

```
