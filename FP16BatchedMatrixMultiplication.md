## FP16 Batched Matrix Multiplication

Implement a batched matrix multiplication in FP16. Given a batch of matrices A of shape [B, M, K] and a batch of matrices B of shape [B, K, N], compute the output batch C of shape [B, M, N] such that for each batch index b:
<img width="731" height="42" alt="image" src="https://github.com/user-attachments/assets/cef9987e-e5b4-4bab-848e-8c89fcbb1e8a" />


All matrices are stored in row-major order and use 16-bit floating point numbers (FP16/half). Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16.

## Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16
 - The final result must be stored in the C array as half

## Example 1:
```
Input:
B = 2, M = 2, K = 3, N = 2
A = [
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
]
B = [
  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
  [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]
]
Output:
C = [
  [[22.0, 28.0], [49.0, 64.0]],
  [[92.0, 68.0], [128.0, 95.0]]
]
```

## Constraints
 - 1 ≤ B ≤ 128
 - 1 ≤ M, N, K ≤ 1024
 - Performance is measured with K = 256, M = 256, N = 256


```
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {}

```
