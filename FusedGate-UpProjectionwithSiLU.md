## Fused Gate-Up Projection with SiLU

Implement the fused gate-up projection with SiLU used in every LLaMA / Mistral SwiGLU MLP block. 
A packed weight `W_gate_up` of shape `[H, 2I]` produces both the gate and up projections in one matrix multiply; the gate half is passed through SiLU and multiplied element-wise by the up half. 
Production CuTe / CUTLASS kernels fuse the SiLU-and-mul into the GEMM epilogue so the `2I-wide` intermediate never lands in HBM. All tensors are `float32`.

<img width="1129" height="365" alt="image" src="https://github.com/user-attachments/assets/d62ef5ef-aa3a-469a-9717-245ec21edcfd" />

## Implementation Requirements
Implement `solve(x, W_gate_up, output, M, H, I)`:

 - `x`: shape `(M, H)`, the token hidden states
 - `W_gate_up`: shape `(H, 2I)`, packed gate and up weights laid out row-major so columns `[0, I)` are `W_gate` and columns `[I, 2I)` are `W_up`
 - `output`: shape `(M, I)`
For every token `m` and channel `i`:
<img width="1172" height="204" alt="image" src="https://github.com/user-attachments/assets/62ea9f27-da98-45a2-a301-3aba6ac09bb7" />
 

 - Use only native features of the framework you choose (no external libraries); keep the function signature unchanged
 - Write the result into output; every entry of `output` must be written
 - Accumulate the GEMM in `float32` or better; reduced-precision `(TF32/FP16)` products of `x` and `W_gate_up` exceed the tolerance on the larger tests

## Example
Input: `M = 2`, `H = 2`, `I = 2`, `x = identity`, `W_gate_up = identity` on both the gate and up halves:
  
<img width="1165" height="273" alt="image" src="https://github.com/user-attachments/assets/f5aaefaf-fac2-419d-97fd-86e920bad58b" /> 

## Constraints
 - 1 ≤ `M` ≤ 8,192
 - 1 ≤ `H` ≤ 8,192
 - 1 ≤ `I` ≤ 16,384
 - Inputs are drawn from a normal distribution; `x` and `W_gate_up` may be scaled so gate values reach about `±20`
 - Performance is measured with `M = 4,096`, `H = 4,096`, `I = 11,008` (Llama-3-8B-style gate-up widths)

## `solve.cu`
```
#include <cuda_runtime.h>

// x, W_gate_up, output are device pointers
extern "C" void solve(const float* x, const float* W_gate_up, float* output, int M, int H, int I) {}

```
