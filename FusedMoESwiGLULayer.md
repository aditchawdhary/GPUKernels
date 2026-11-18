## Fused MoE SwiGLU Layer

Implement the expert computation of a Mixture-of-Experts (MoE) feed-forward layer, as used in Mixtral, DeepSeek and Qwen-MoE. 
Each of the `T` tokens in `x` `(T × H)` has already been routed to K distinct experts: expert indices are in topk_ids (`T × K`, `32-bit` int) and combine weights in topk_weights `(T × K)`. 
Every expert e is a SwiGLU MLP with weights `w_gate[e]`, `w_up[e]` `(H × I)` and `w_down[e]` `(I × H)`. Compute

<img width="1163" height="102" alt="image" src="https://github.com/user-attachments/assets/78abc50f-c77f-4e73-ba83-b48a3dd8b9d5" />

where <img width="152" height="20" alt="image" src="https://github.com/user-attachments/assets/aaa6d52c-d292-41e2-a243-a04a56f73e14" />. 
All floating-point tensors are `32-bit`. A fast solution groups tokens by expert and fuses gate/up GEMMs, the SwiGLU activation, the down projection and the weighted scatter-combine instead of running a dense layer per expert.

## Implementation Requirements
 - Do not change the function signature.
 - Do not use external libraries beyond what is available in the starter.
 - Write the result to output.
 - Use topk_weights exactly as given (they are not guaranteed to be normalized or positive).
 - The K expert ids of a token are distinct; some experts may receive no tokens.

## Example 1:
Input: `T` = 2, `H` = 2, `I` = 2, `E` = 2, `K` = 1
<img width="1197" height="423" alt="image" src="https://github.com/user-attachments/assets/82db5276-097d-450f-aaf8-fb23778410af" />
  
  

## Constraints
 - 1 ≤ `T` ≤ 8,192
 - 1 ≤ `H`, `I` ≤ 4,096
 - 1 ≤ `K` ≤ `E` ≤ 64
 - 0 ≤ `topk_ids[t][k]` < `E`, distinct within a row
 - Performance is measured with `T = 4,096`, `H = 1,024`, `I = 1,408`, `E = 16`, `K = 2`

## `solve.cu`
```
#include <cuda_runtime.h>

// x, topk_ids, topk_weights, w_gate, w_up, w_down, output are device pointers
extern "C" void solve(const float* x, const int* topk_ids, const float* topk_weights,
                      const float* w_gate, const float* w_up, const float* w_down, float* output,
                      int T, int H, int I, int E, int K) {}

```
