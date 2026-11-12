## Multi-Head Attention

Implement a program for multi-head self-attention. Given three input matrices `Q` (queries), `K` (keys), and `V` (values) of size `N x d_model`, compute:

<img width="726" height="56" alt="image" src="https://github.com/user-attachments/assets/62242017-4155-4213-a8c8-e664d15aa9cb" />

where each head computes:

<img width="726" height="137" alt="image" src="https://github.com/user-attachments/assets/ac572391-68e6-4ac1-8788-15ca2afcfef2" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output array

## Example 1:
<img width="726" height="440" alt="image" src="https://github.com/user-attachments/assets/765ecaf8-3968-46cd-b11b-524befab514b" />


## Example 2:
<img width="726" height="320" alt="image" src="https://github.com/user-attachments/assets/bc5c4c2f-945a-4bea-9875-7a3906b08920" />
 

## Constraints
 - `1 ≤ N ≤ 10000`
 - `2 ≤ d_model ≤ 1024`
 - `1 ≤ h ≤ d_model`
 - `d_model % h == 0`
 - `-10.0 ≤ values ≤ 10.0`
 - Performance is measured with `N = 1,024`, `d_model = 1,024`, and values sampled from `[-1.0, 1.0]`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {}

```
