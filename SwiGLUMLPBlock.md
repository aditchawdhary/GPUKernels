#### SwiGLU MLP Block

Implement the SwiGLU MLP block — the feedforward network used in LLaMA, Mistral, Gemma, and most modern large language models. 

Given an input matrix `x` of shape `[M, d_model]` and three weight matrices `W_gate`, `W_up` (each `[d_model, d_ffn]`), and `W_down` (`[d_ffn, d_model]`), compute: 
`output = (SiLU(x × W_gate) ⊙ (x × W_up)) × W_down`, where `SiLU(z) = z × sigmoid(z)` and `⊙` denotes element-wise multiplication. All tensors are `float32`.

<img width="1405" height="446" alt="image" src="https://github.com/user-attachments/assets/d22001c4-3b27-4011-b4e7-cdb9de907cfa" />


#### Implementation Requirements
 - Implement the solve function with the signature unchanged.
 - Do not use external libraries beyond the framework provided.
 - Write the result into output in-place.

#### Example
Input: M = 2, d_model = 2, d_ffn = 4

<img width="1234" height="740" alt="image" src="https://github.com/user-attachments/assets/2cbdef0d-251a-4323-b6be-6c771e265bd1" />


#### Constraints
 - 1 ≤ M ≤ 65,536
 - 1 ≤ d_model ≤ 8,192
 - 1 ≤ d_ffn ≤ 32,768
 - All tensors are float32 on the GPU.
 - Input values are in the range [-10, 10].
 - Performance is measured with M = 512, d_model = 4,096, d_ffn = 14,336

```
#include <cuda_runtime.h>

// x, W_gate, W_up, W_down, output are device pointers
extern "C" void solve(const float* x, const float* W_gate, const float* W_up, const float* W_down,
                      float* output, int M, int d_model, int d_ffn) {}

```
