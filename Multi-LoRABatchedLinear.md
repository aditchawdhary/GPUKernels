## Multi-LoRA Batched Linear

Implement the batched multi-adapter LoRA projection used by multi-tenant LLM servers (Punica / S-LoRA, vLLM multi-LoRA). A batch of `T` tokens `x` (`T × K`) shares one base weight `W` (`K × N`), 
but each token may belong to a different fine-tuned adapter selected from a pool of `L` low-rank adapters `A` (`L × K × R`) and `B` (`L × R × N`). With `a = adapter_ids[t]`:

<img width="986" height="64" alt="image" src="https://github.com/user-attachments/assets/f33c3a74-1ba5-40c3-ad19-f37d4c24a712" />
 

Tokens are not grouped by adapter. All tensors are 32-bit floats except `adapter_ids`, which is 32-bit integer.

## Implementation Requirements
 - Do not change the function signature.
 - Do not use external libraries beyond what is available in the starter.
 - Write the result to y (shape T × N).
 - Tokens with `adapter_ids[t] = -1` use only the base weight.

## Example 1:
Input: `T = 3`, `K = 2`, `N = 2`, `R = 1`, `L = 2`, `scale = 0.5`, `adapter_ids = [0, 1, -1]`
 
<img width="989" height="104" alt="image" src="https://github.com/user-attachments/assets/d1747936-9ed9-48d7-9c86-8dfe8e5707b2" />  
  
Output:
<img width="982" height="105" alt="image" src="https://github.com/user-attachments/assets/e54065af-fb04-4b0a-b948-28f1695ad426" />


## Constraints
 - 1 ≤ `T` ≤ 4,096
 - 1 ≤ `K`, `N` ≤ 8,192
 - 1 ≤ `R` ≤ 64
 - 1 ≤ `L` ≤ 64
 - -1 ≤ `adapter_ids[t]` < `L`
 - Performance is measured with `T = 1,024`, `K = 4,096`, `N = 4,096`, `R = 16`, `L = 8`

## `solve.cu`
```
#include <cuda_runtime.h>

// x, W, A, B, adapter_ids, y are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      const int* adapter_ids, float* y, int T, int K, int N, int R, int L,
                      float scale) {}

```
