### Fused LayerNorm Linear GELU

Implement the first half of a transformer MLP block as a single fused operation: layer normalization of each row of `x` `(M × K)`, followed by a linear projection with `W` `(K × N)` and `bias, followed by the exact (erf-based) GELU activation. 
All tensors are 32-bit floats.

<img width="756" height="84" alt="image" src="https://github.com/user-attachments/assets/42b275fa-1407-4827-a135-0f6b05b89b76" />

where `mu_i` and `sigma_i^squared` are the mean and (biased, divide-by-K) variance of row `i`. 
A fast solution computes the row statistics once and applies the normalization while loading `x` tiles into the GEMM, with bias and GELU in the epilogue, so neither the normalized activations nor the pre-activation are written to memory.

### Implementation Requirements
 - Do not change the function signature.
 - Do not use external libraries beyond what is available in the starter.
 - Write the result to y.
 - Use the exact erf-based GELU, not the tanh approximation.

### Example 1:
<img width="741" height="328" alt="image" src="https://github.com/user-attachments/assets/8909f11c-e53d-4e21-b370-dcdd7d840df2" />
 

### Constraints
 - 1 ≤ `M` ≤ 8,192
 - 1 ≤ `K`, `N` ≤ 8,192
 - 0 < `eps` ≤ 1
 - Input values may carry a large common offset (e.g. around -1,000)
 - Performance is measured with `M` = 4,096, `K` = 4,096, `N` = 4,096

```cuda
#include <cuda_runtime.h>

// x, gamma, beta, W, bias, y are device pointers
extern "C" void solve(const float* x, const float* gamma, const float* beta, const float* W,
                      const float* bias, float* y, int M, int K, int N, float eps) {}

```
