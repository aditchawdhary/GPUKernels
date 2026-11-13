## MoE Top-K Gating

Implement a GPU program that performs Top-K Gating for Mixture of Experts (MoE) models. Given a logit matrix of shape [M, E] where M is the number of tokens and E is the number of experts, identify the k largest values in each row, extract their indices, and apply softmax to get mixing weights.

For each row `i`, the operation computes:
<img width="733" height="104" alt="image" src="https://github.com/user-attachments/assets/d1dfa3b1-fc5a-462b-9f5c-5498e6d691b3" />


The selected experts must remain ordered by descending logit value, matching the order returned by `topk`. The `topk_weights` array must correspond positionally to `topk_indices` in that same order.

## Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - The final result must be stored in the topk_weights and topk_indices arrays

## Example 1:
```
Input:
  logits = [[1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0]]
  M = 2, E = 4, k = 2

Output:
  topk_weights = [[0.7311, 0.2689],
                  [0.7311, 0.2689]]
  topk_indices = [[3, 2],
                  [0, 1]]

Explanation:
Row 0: Top-2 values are 4.0 and 3.0 at indices 3 and 2.
       Softmax([4.0, 3.0]) = [0.7311, 0.2689]
Row 1: Top-2 values are 4.0 and 3.0 at indices 0 and 1.
       Softmax([4.0, 3.0]) = [0.7311, 0.2689]
```

## Constraints
 - 1 ≤ M ≤ 10,000 (number of tokens)
 - 1 ≤ E ≤ 256 (number of experts)
 - 1 ≤ k ≤ E (top-k selection, typically k=2)
 - All tensors are stored on GPU
 - Logits are 32-bit floats
 - Indices are 32-bit integers
 - Performance is measured with M = 1,024, k = 2

```
#include <cuda_runtime.h>

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {}

```
