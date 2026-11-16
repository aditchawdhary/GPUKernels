## Linear Recurrence

Given two matrices a and `x`, each of shape `[B, L]` (`batch size` × `sequence length`), compute the linear recurrence h of shape `[B, L]` defined by: `h[b, 0] = x[b, 0]` and `h[b, t] = a[b, t] × h[b, t−1] + x[b, t]` for `t ≥ 1`. 
All values are float32. This operation is the core computational primitive of State Space Models (SSMs) such as Mamba, S4, and H3.

<img width="1062" height="249" alt="image" src="https://github.com/user-attachments/assets/7e3d62ee-f76a-4ccb-8bf2-8023ef1686c9" />

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The result must be stored in the output tensor `h`

## Examples
<img width="1081" height="445" alt="image" src="https://github.com/user-attachments/assets/aa09ae9c-8213-4ba9-a2b0-568dd4f6dadb" />

 
## Constraints
 - `1` ≤ `B` ≤ `256` (batch size)
 - `1` ≤ `L` ≤ `65,536` (sequence length)
 - All values in `a` and `x` are `float32`
 - Performance is measured with `B` = `64`, `L` = `16,384`

## `solve.cu`
```
#include <cuda_runtime.h>

// a, x, h are device pointers
extern "C" void solve(const float* a, const float* x, float* h, int B, int L) {}

```
