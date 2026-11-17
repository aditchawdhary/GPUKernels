## Softmax Attention Backward

Given the query matrix `Q` of size `M×d`, key matrix `K` of size `N×d`, value matrix `V` of size `N×d`, and upstream gradient `dO` of size `M×d`, compute the gradients of the softmax attention operation with respect to `Q`, `K`, and `V`. 
The forward pass computes
<img width="1156" height="82" alt="image" src="https://github.com/user-attachments/assets/371f0ec4-e5f5-43e4-8637-c456fe6834d0" />

where softmax is applied row-wise. The backward pass produces `dQ` `(M×d)`, `dK` `(N×d)`, and `dV` `(N×d)` via:
<img width="1147" height="213" alt="image" src="https://github.com/user-attachments/assets/f22bd541-eb0f-4ed6-9997-9d5dcb36b94e" />
 

## Implementation Requirements
 - Use only GPU native features (external libraries are not permitted)
 - The `solve` function signature must remain unchanged
 - The final results must be stored in `dQ`, `dK`, and `dV`

## Example
Input:

<img width="1152" height="508" alt="image" src="https://github.com/user-attachments/assets/43e2602a-c710-4cfa-acd7-ee8a42c4959a" />


Output:
<img width="1153" height="391" alt="image" src="https://github.com/user-attachments/assets/8f914983-851b-4c4a-aab9-00f94bb03b6b" />

 

## Constraints
 - Matrix `Q` and `dO` are of size `M × d`; matrices `K`, `V` are of size `N × d`
 - 1 ≤ `M`, `N` ≤ 100,000
 - 1 ≤ `d` ≤ 128
 - All input values are `float32`
 - Performance is measured with `M = 8,192`, `N = 4,096`, `d = 128`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, dO, dQ, dK, dV are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* dO, float* dQ,
                      float* dK, float* dV, int M, int N, int d) {}
```
