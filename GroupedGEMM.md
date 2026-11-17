## Grouped GEMM

Implement a Grouped General Matrix Multiplication. The input matrix `A` of shape `M_total × K` is partitioned into `G` contiguous row groups whose boundaries are given by the cumulative array group_offsets of length `G + 1`. 
Each group `g` has its own weight matrix `B[g]` of shape `K × N`. For every group, multiply the rows assigned to it by that group's weight matrix and write the result into the corresponding rows of `C`.

Formally, for every group `g` in `0, 1, ..., G-1`, let `s = group_offsets[g]` and `e = group_offsets[g+1]`. Then `C[s:e] = A[s:e] × B[g]`. A group may be empty `(s == e)`, in which case no rows are written for that group.

This pattern is the core building block of Mixture-of-Experts (MoE) inference: after a router has dispatched tokens to experts, each expert holds a variable number of tokens that all share the same weight matrix. 
A single grouped GEMM computes every expert's projection in one launch.

<img width="1235" height="301" alt="image" src="https://github.com/user-attachments/assets/418a3451-cd87-490f-bdd4-0c3199a8460e" />


## Implementation Requirements
 - Implement the solve function; do not change its signature.
 - External libraries are not permitted.
 - Write the result into C.

## Examples
<img width="1237" height="346" alt="image" src="https://github.com/user-attachments/assets/18326c30-74a0-446a-8761-b09888797387" />

## Constraints
 - 1 ≤ `G` ≤ 64 (number of groups)
 - 1 ≤ `M_total` ≤ 16,384 (total rows across all groups)
 - 1 ≤ `K`, `N` ≤ 4,096
 - `group_offsets[0] = 0`, `group_offsets[G] = M_total`, and group_offsets is non-decreasing
 - A group may be empty (`group_offsets[g+1]` = `group_offsets[g]`); in that case nothing is written for that group
 - `A`, `B`, and `C` are `32-bit` floats; group_offsets is `32-bit` integer
 - Values of `A` and `B` are in the range `[-1, 1]`
 - All tensors are stored on the GPU
 - Performance is measured with `G = 8`, `M_total = 8,192`, `K = 1,024`, `N = 2,048`

## `solve.cu`
```
#include <cuda_runtime.h>

// A, B, group_offsets, C are device pointers
extern "C" void solve(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {}

```
