## Hadamard Transform

Implement a GPU program that applies the orthonormal Walsh–Hadamard transform to every row of a batched input matrix. Given an input tensor of shape `B × N`, where `N` is a power of two, compute the transform of each row independently and write the result to the output tensor. 
This kernel appears in modern LLM inference pipelines (e.g. QuaRot, QuIP#), where activations are rotated by a Hadamard matrix before low-bit quantization to spread out outlier values.

<img width="904" height="256" alt="image" src="https://github.com/user-attachments/assets/a9735251-7b76-4b9a-8ae7-bde33fd398a9" />


## Implementation Requirements
 - External libraries are not permitted
 - The `solve` function signature must remain unchanged
 - The final result must be stored in the `output` tensor
 - The kernel must be entirely GPU-resident—no host-side fallback
 - `N` is guaranteed to be a power of two

## Example
```
Input:  B = 1, N = 4
        input = [1.0, 2.0, 3.0, 4.0]

Output: output = [5.0, -1.0, -2.0, 0.0]
```

<img width="905" height="75" alt="image" src="https://github.com/user-attachments/assets/9214993c-afea-4023-974c-814273fecbc0" />


## Constraints
 - 1 ≤ `B` ≤ 16,384
 - 1 ≤ `N` ≤ 4,096 and `N` is a power of two
 - All values are `32-bit` floating point numbers
 - Absolute error ≤ `1e-4` and relative error ≤ `1e-4`
 - Performance is measured with `B` = 8,192, `N` = 4,096

## `solve.cu`
```
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int B, int N) {}

```
