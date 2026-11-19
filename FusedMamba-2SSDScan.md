## Fused Mamba-2 SSD Scan

Implement the fused Mamba-2 selective state space (SSD) layer used for prefill in Mamba-2 and hybrid LLMs such as Codestral Mamba, Nemotron-H and Granite 4. 
Each head keeps a `head_dim × state_dim` state that decays by a data-dependent scalar every token, takes a rank-1 update from the token, and is read out by the token's `C` vector. 
Fuse the time-step softplus, the scan, the `D` skip connection and the `silu(z)` output gate into one pass. All tensors are `float32`.

<img width="1006" height="325" alt="image" src="https://github.com/user-attachments/assets/d8be0210-b96f-42ec-bf21-2c4f3923df76" />

## Implementation Requirements
Implement `solve(x, dt, dt_bias, A, B, C, D, z, y, batch, seq_len, num_heads, head_dim, num_groups, state_dim)`:

 - `x`, `z`: shape `(batch, seq_len, num_heads, head_dim)`, the SSM input and the gate input
 - `dt`: shape `(batch, seq_len, num_heads)`, the raw time step
 - `dt_bias`, `A`, `D`: shape `(num_heads,)`; every `A[h]` is negative
 - `B`, `C`: shape `(batch, seq_len, num_groups, state_dim)`, shared by the heads of a group
 - `y`: shape `(batch, seq_len, num_heads, head_dim)`, the output

Head `h` reads group `g = h / (num_heads / num_groups)` (integer division), so consecutive heads share `B` and `C`. For each sequence `b` and head `h`, the state <img width="200" height="25" alt="image" src="https://github.com/user-attachments/assets/55b6f7d2-8a40-42da-b362-408f998275e2" /> 
starts at zero and, for <img width="200" height="25" alt="image" src="https://github.com/user-attachments/assets/6014135a-5907-49d2-bd1f-ee1ccf34c53c" />
:

<img width="1176" height="133" alt="image" src="https://github.com/user-attachments/assets/2ff2de78-4869-4bd2-ae27-9e5a6602a313" />

 
The state decays before the new token is added, and the new token's own update is included in its output. Sequences in the batch are independent.

 - Use only native features of the framework you choose (no external libraries); keep the function signature unchanged
 - Write the result into y; every entry of y must be written, and the inputs must not be modified
 - Accumulate in float32 or better. A sequential scan is correct but slow; the chunked SSD form (a masked quadratic form inside each chunk plus states carried between chunks) maps onto matrix multiplies

## Example
Input: `batch = 1`, `seq_len = 3`, `num_heads = 1`, `head_dim = 2`, `num_groups = 1`, `state_dim = 2`, `dt = 0`, `dt_bias = [0]`, `A = [-1]`, `D = [0.5]`, and every entry of `z is 1`. Rows are time steps:
 
<img width="1005" height="317" alt="image" src="https://github.com/user-attachments/assets/27057b0f-1d8e-41be-8602-7ae901df40c4" />
   

## Constraints
 - All tensors are `float32`
 - 1 ≤ `batch` ≤ 4; 1 ≤ `seq_len` ≤ 4,096
 - 1 ≤ `num_groups` ≤ `num_heads` ≤ 32, and `num_heads` is divisible by `num_groups`
 - 1 ≤ `head_dim` ≤ 128; 1 ≤ `state_dim` ≤ 128
 - -16 ≤ `A` < 0; -30 ≤ `dt + dt_bias` ≤ 30
 - Performance is measured with `batch = 2`, `seq_len = 4,096`, `num_heads = 32`, `head_dim = 64`, `num_groups = 4`, `state_dim = 128`

## `solve.cu`
```
#include <cuda_runtime.h>

// x, dt, dt_bias, A, B, C, D, z, y are device pointers
extern "C" void solve(const float* x, const float* dt, const float* dt_bias, const float* A,
                      const float* B, const float* C, const float* D, const float* z, float* y,
                      int batch, int seq_len, int num_heads, int head_dim, int num_groups,
                      int state_dim) {}

```
