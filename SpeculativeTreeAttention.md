## Speculative Tree Attention

Implement the verification attention step of tree-based speculative decoding (Medusa, EAGLE, SpecInfer). A draft model proposes a tree of T candidate tokens that continue a context of P tokens whose keys and values are already in `K_cache` and `V_cache` (shape `P × H × D`). 
The target model scores every tree node in a single forward pass: `node i` (`query Q[i]`, `key K_tree[i]`, `value V_tree[i]`) attends to all `P` context tokens plus every node on its own root-to-node path, and to nothing else.

The tree is given by parents: `parents[i]` is the index of node i's parent, with `parents[i] < i`, or -1 if node i directly follows the context. Let `A(i)` be node i together with all of its ancestors. For each head `h`:

 <img width="1032" height="92" alt="image" src="https://github.com/user-attachments/assets/ce709169-8214-450f-a505-ea815da50020" />

where context keys/values come from the caches and tree keys/values from K_tree/V_tree. All tensors are 32-bit floats except parents, which is `32-bit` integer.

## Implementation Requirements
 - Do not change the function signature.
 - Do not use external libraries beyond what is available in the starter.
 - Write the result to output (shape T × H × D).
 - The softmax must be numerically stable for large scores.

## Example 1:
Input: `P` = 1, `T` = 3, `H` = 1, `D` = 2, `parents` = `[-1, 0, 0]` (node 0 follows the context; nodes 1 and 2 are siblings under node 0)
<img width="1034" height="206" alt="image" src="https://github.com/user-attachments/assets/9ec3f8cc-0038-48d4-b65d-ed7d45b46ff7" />
 
  
 
 
 
Node 0 attends to {context, 0}; node 1 to {context, 0, 1}; node 2 to {context, 0, 2}.
Output:
<img width="1008" height="109" alt="image" src="https://github.com/user-attachments/assets/3c526594-984a-4496-8204-0944d98682c4" />
 

## Constraints
 - 1 ≤ `P` ≤ 16,384
 - 1 ≤ `T` ≤ 128
 - 1 ≤ `H` ≤ 64
 - 1 ≤ `D` ≤ 256
 - -1 ≤ `parents[i]` < i
 - Input values are in `[-6, 6]`
 - Performance is measured with `P = 8,192`, `T = 64`, `H = 32`, `D = 128`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K_cache, V_cache, K_tree, V_tree, parents, output are device pointers
extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const float* K_tree, const float* V_tree, const int* parents, float* output,
                      int P, int T, int H, int D) {}

```
