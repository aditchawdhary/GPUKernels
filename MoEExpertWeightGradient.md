## MoE Expert Weight Gradient

Compute the weight gradients of a Mixture-of-Experts (MoE) expert projection in the backward pass. Routed tokens have already been sorted by expert: the activations `X` (`M_total × K`) and the upstream gradients `dY` (`M_total × N`) are partitioned into `G` contiguous row groups whose boundaries are given by `group_offsets` (`length G + 1`). For every expert `g`, with `s = group_offsets[g]` and `e = group_offsets[g+1]`, compute `dW[g] = X[s:e]T × dY[s:e]`, a `K × N` matrix. All floating-point tensors are `32-bit`.

<img width="726" height="90" alt="image" src="https://github.com/user-attachments/assets/bb2dfb4d-a8a2-48ab-9829-5300ed9cf529" />

Unlike the forward grouped GEMM, the ragged dimension (tokens per expert) is the reduction dimension: every expert's output has the same fixed size, but the amount of work behind it depends on how many tokens the router sent to that expert. An expert that received no tokens has an all-zero gradient.

<img width="727" height="287" alt="image" src="https://github.com/user-attachments/assets/e8615f3c-8dc3-4477-a28f-057def53b68d" />

## Implementation Requirements
 - Implement the solve function; do not change its signature.
 - External libraries are not permitted.
 - Write the result into `dW`, laid out as `G` consecutive row-major `K × N` matrices.
 - `dW` is not zero-initialized; every element must be written, including the gradients of experts that received no tokens (which must be `0`).
 - Any router / gate weighting is already folded into `dY`; do not apply gate weights.
 - Use `float32` operands and accumulate in `float32` or better; rounding `X` or `dY` to `TF32`, `FP16` or `BF16` is outside the tolerance.

## Example
Input: `G = 3`, `M_total = 3`, `K = 2`, `N = 2`, `group_offsets = [0, 2, 2, 3]`

<img width="724" height="323" alt="image" src="https://github.com/user-attachments/assets/72712cb4-e0d0-4dc1-afa3-4ce967fb8b2e" />

 
## Constraints
 - 1 ≤ `G` ≤ 128 (number of experts)
 - 1 ≤ `M_total` ≤ 65,536 (total routed rows across all experts)
 - 1 ≤ `K`, `N` ≤ 4,096
 - `group_offsets[0]` = 0, `group_offsets[G]` = `M_total`, and `group_offsets` is non-decreasing; a group may be empty
 - `X`, `dY` and `dW` are `32-bit` floats; `group_offsets` is `32-bit` integer
 - Values of `X` and `dY` are in the range `[-1, 1]`
 - Performance is measured with `G = 64`, `M_total = 32,768`, `K = 1,024`, `N = 512` (skewed expert loads, two experts empty)

```
#include <cuda_runtime.h>

// X, dY, group_offsets, dW are device pointers
extern "C" void solve(const float* X, const float* dY, const int* group_offsets, float* dW, int G,
                      int M_total, int K, int N) {}

```
