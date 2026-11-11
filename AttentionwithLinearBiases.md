## Attention with Linear Biases

Implement Attention with Linear Biases (ALiBi), following the method described in "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" , for a given set of matrices. Given the query matrix `Q` of size `M×d`, key matrix `K` of size `N×d`, and value matrix `V` of size `N×d`, your program should compute the output matrix using the formula:

<img width="736" height="65" alt="image" src="https://github.com/user-attachments/assets/a2870596-d6c0-4a7c-900c-aa8e8ded70b1" />
 
where `α` is a slope controlling the linear bias and `Δ = i - j` represents the relative position between query `i` and key `j`. The softmax function is applied row-wise. 
`Q`, `K`, `V`, `output`, and `α` are all of data type `float32`; `M`, `N`, `d` are of data type `int32`.

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output matrix output

## Example 1:
<img width="720" height="561" alt="image" src="https://github.com/user-attachments/assets/99fc6541-af2e-4eb3-af2e-8acb3f531738" />
 

## Example 2:
<img width="724" height="454" alt="image" src="https://github.com/user-attachments/assets/dcac26dd-de80-4f24-8d8d-bdd786dcc9d1" />


## Constraints
 - Matrix `Q` is of size `M×d` and matrices `K` and `V` are of size `N×d`
 - 1 ≤ `M, N` ≤ 2048
 - 1 ≤ `d` ≤ 1024
 - -1.0 ≤ `α` ≤ 1.0
 - Performance is measured with `M = 2,048`, `N = 2,048`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) {}

```
