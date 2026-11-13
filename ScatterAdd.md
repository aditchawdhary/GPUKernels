## Scatter Add

Implement a row-wise scatter-add (the operation behind torch.index_add_, used to combine Mixture-of-Experts outputs and to aggregate messages in graph neural networks). 
Given a source matrix src of shape N × D and an integer vector index of length N, add each source row i into row index[i] of an `M × D` output matrix that starts at zero:

<img width="930" height="73" alt="image" src="https://github.com/user-attachments/assets/453a16ed-73b1-4d04-a74e-3729d7f4f5b8" />
 
All values are 32-bit floating point numbers; index holds 32-bit integers.

## Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - The final result must be stored in output; its initial contents are undefined, so every row must be written (rows no source row maps to are 0)

## Example:
Input (`N` = 4, `D` = 2, `M` = 3):
<img width="924" height="277" alt="image" src="https://github.com/user-attachments/assets/8c027d6e-4864-4316-bc6d-46177e59c226" />

## Constraints
 - 1 ≤ `N` ≤ 2,000,000
 - 1 ≤ `D` ≤ 1,024
 - 1 ≤ `M` ≤ 1,000,000
 - 0 ≤ `index[i]` < `M`
 - Source values are in the range `[-10, 10]`
 - Performance is measured with `N = 1,000,000`, `D = 128`, `M = 50,000`

```
#include <cuda_runtime.h>

// src, index, output are device pointers
extern "C" void solve(const float* src, const int* index, float* output, int N, int D, int M) {}

```
