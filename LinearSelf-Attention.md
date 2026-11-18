## Linear Self-Attention

Implement Linear Attention for a given set of matrices, following the method described in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" . 
Given the query matrix `Q` of size `M×d`, key matrix `K` of size `M×d`, and value matrix `V` of size `M×d`, your program should compute the output matrix using the formula:

<img width="1120" height="107" alt="image" src="https://github.com/user-attachments/assets/6952530e-1a51-42d7-b688-e171c43e0701" />


where <img width="57" height="26" alt="image" src="https://github.com/user-attachments/assets/564b4698-c8fc-4a13-ab4f-59d8cae5c559" /> is a feature map applied element-wise, for example:
 <img width="1132" height="91" alt="image" src="https://github.com/user-attachments/assets/678dff1a-b32e-4831-bd2a-a80703eeb184" />

All matrices `Q`, `K`, `V`, and `output` are of type `float32`, and `M` and `d` are of type `int32`.

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output matrix output

## Example 1:
### Input:
<img width="1114" height="378" alt="image" src="https://github.com/user-attachments/assets/650986d3-4980-413f-bf1c-23d0ecf8f204" />
 

### Output:
<img width="1117" height="137" alt="image" src="https://github.com/user-attachments/assets/79959c4c-5837-4ae8-8ebb-ca39b82c6706" />
 

## Example 2:
### Input:
<img width="1115" height="385" alt="image" src="https://github.com/user-attachments/assets/a3acdeb0-ab48-44d9-9202-31746b1bdd35" />

### Output:
output (2×2):
<img width="1117" height="122" alt="image" src="https://github.com/user-attachments/assets/e9398994-aba5-4e66-974c-4bd89588ff84" />
 

## Constraints
 - Matrix `Q`, `K`, and `V` are all of size `M×d`
 - 1 ≤ `M` ≤ 10000
 - 1 ≤ `d` ≤ 128
 - All elements in `Q`, `K`, and `V` are sampled from `[-100.0, 100.0]`
 - Data type for all matrices is `float32`
 - Performance is measured with `M = 10,000`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {

}

```
