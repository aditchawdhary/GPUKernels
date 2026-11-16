## Gated Attention

Implement gated attention (Qiu et al., 2025). In addition to standard multi-head causal attention, every head applies its own element-wise sigmoid gate to its `output`. 
The gate is not given: it is computed from the layer input `X` through a learned projection `W_g`, so the gate projection is a full matrix multiplication that an efficient kernel should fuse with the attention computation instead of materializing it separately. 
All tensors are `float32`.

<img width="241" height="224" alt="image" src="https://github.com/user-attachments/assets/5e041129-8d82-4cd2-bede-d2291e0a0498" />


## Implementation Requirements
Implement `solve(Q, K, V, X, W_g, output, M, d_model, h)`:

 - Q, K, V: shape (M, d_model), query, key, and value matrices
 - X: shape (M, d_model), the layer input the gate is computed from
 - W_g: shape (d_model, d_model), gate projection weights
 - output: shape (M, d_model), the gated attention output
 - M: number of tokens; d_model: model dimension, divisible by h; h: number of attention heads, head dimension d_k = d_model / h
For head `j = 0..h-1`, let `Q_j`, `K_j`, `V_j` be the column blocks `[j·d_k, (j+1)·d_k)` of `Q`, `K`, `V`, and let `G_j` be the same column block of the gate-logit matrix `X W_g`:

<img width="1160" height="125" alt="image" src="https://github.com/user-attachments/assets/7c3c01cb-faea-4d3f-8985-abb86ab858f3" />

 
where causal_mask sets every position `(i, j)` with `j > i` to `-∞` before the softmax, so a token can only attend to itself and earlier tokens.

 - Use only native features of the framework you choose (no external libraries); keep the function signature unchanged
 - Write the result into `output`; every entry of output must be written
 - Accumulate the GEMM in `float32` or better; compute the `softmax` in `float32` with max subtraction

## Example
<img width="1172" height="426" alt="image" src="https://github.com/user-attachments/assets/89543d11-87f3-4367-8994-87fa13a9c721" />


## Constraints
 - All tensors are float32
 - 1 ≤ `M` ≤ 4,096
 - `d_model` up to 512, divisible by `h`; `h` up to 8
 - Inputs `Q`, `K`, `V`, `X` are drawn from signed ranges; `W_g` is scaled so gate values stay in a range where the `sigmoid` is not saturated
 - Performance is measured with `M = 4,096`, `d_model = 512`, `h = 8`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, X, W_g, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* X,
                      const float* W_g, float* output, int M, int d_model, int h) {}

```
