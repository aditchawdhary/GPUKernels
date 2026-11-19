## Sliding Window Self-Attention

Implement Sliding Window Self-Attention for a given set of matrices. Before introducing the sliding window version, let's first recall standard Self-Attention.

1. Standard Softmax Attention
Given query matrix `Q`, key matrix `K`, and value matrix `V`, each position `i` attends to all positions `j` using a softmax-weighted sum:

 <img width="543" height="118" alt="image" src="https://github.com/user-attachments/assets/cceefb01-3c46-4c9c-8700-15454aa5d20d" />

In other words, each query computes similarity with all keys, applies a softmax to get attention weights, and then computes a weighted sum of values.

2. Sliding Window Self-Attention
Sliding Window Attention modifies standard attention by restricting each query to attend only to a local window around its position.

 - For each position `i`, only consider the keys and values within a window of size `window_size` around `i` (positions `[i-window_size, ..., i+window_size]`).
 - Compute similarity scores between `Qi` and the keys in this window:
 <img width="131" height="41" alt="image" src="https://github.com/user-attachments/assets/6a16df0c-0043-414b-85f3-655bb315445e" />


 - Apply `softmax` over these local scores to obtain attention weights.
 - Use the weights to compute a weighted average of the values in the same window:
<img width="567" height="51" alt="image" src="https://github.com/user-attachments/assets/ca8a8b48-31c7-41cb-bcb6-71db8e12c511" />

In short, each query only attends to its nearby neighbors.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/7fe20427-f6b8-4505-ba7a-e3eb4f4b64e0" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output matrix output

## Example 1:
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/9fa61c30-4950-4280-8a58-65ec4559866c" />


## Example 2:

<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/29ced11b-5b79-490c-8fa1-6b596d9a6e8c" />

 

## Constraints
 - Matrix `Q`, `K`, and `V` are all of size `M×d`
 - 1 ≤ `M` ≤ 10000
 - 1 ≤ `d` ≤ 128
 - 1 ≤ `window_size` ≤ 32
 - All elements in `Q`, `K`, and `V` are in `[-100.0, 100.0]`
 - Data type for all matrices is `float32`
 - Performance is measured with `M = 5,000` and elements of `Q`, `K` and `V` sampled from `[-1.0, 1.0]`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int window_size) {}

```
