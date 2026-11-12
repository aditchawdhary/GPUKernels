## Causal Self-Attention

Implement Causal (masked) Self-Attention for a given set of matrices. Given the query matrix `Q` of size `M×d`, key matrix `K` of size `M×d`, and value matrix `V` of size `M×d`, your program should compute the output matrix using the formula:
 
<img width="900" height="90" alt="image" src="https://github.com/user-attachments/assets/61630f68-445b-4815-85c4-9ebf840468ab" />

where mask is a causal mask that sets all positions corresponding to keys after the current query to negative infinity.

i.e., for query `i` and key `j`:

<img width="898" height="79" alt="image" src="https://github.com/user-attachments/assets/8ea9ff4a-6678-4fe9-a5d6-a6cb0c3921ed" />

 
The softmax function is applied row-wise. `Q`, `K`, `V`, and `output` are all of data type `float32`; `M`, and `d` are of data type `int32`.

<img width="894" height="379" alt="image" src="https://github.com/user-attachments/assets/020065fa-0df4-46b3-b348-a9b69160f527" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output matrix output

## Example 1:
<img width="895" height="515" alt="image" src="https://github.com/user-attachments/assets/64ff0358-96e6-443f-8e5a-cbf2c8b4bb4a" />

## Example 2:
<img width="913" height="512" alt="image" src="https://github.com/user-attachments/assets/ab4d2f7b-512b-4f0d-9ebf-94a4b854189a" />

## Constraints
 - Matrix `Q`, `K`, and `V` are all of size M×d
 - 1 ≤ `M` ≤ 10000
 - 1 ≤ `d` ≤ 128
 - All elements in `Q`, `K`, and `V` are sampled from `[-1.0, 1.0]`
 - Data type for all matrices is `float32`
 - Performance is measured with `M = 5,000`

## solve.cu
```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {

}

```
