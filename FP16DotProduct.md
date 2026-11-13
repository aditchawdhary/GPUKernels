## FP16 Dot Product

Implement a GPU program that computes the dot product of two vectors containing 16-bit floating point numbers (FP16/half). The dot product is the sum of the products of the corresponding elements of two vectors.

Mathematically, the dot product of two vectors `A` and `B` of length `n` is defined as:
<img width="715" height="90" alt="image" src="https://github.com/user-attachments/assets/75843012-58d7-4127-ad59-5d2547e91ac8" />


All inputs are stored as `16-bit` floating point numbers (`FP16/half`). For best precision, accumulation during multiplication should use `FP32` before converting the final result to `FP16`.

## Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - Accumulation during multiplication should use `FP32` for better precision before converting the final result to `FP16`
 - The final result must be stored in the output variable as `half`

## Example 1:
```
Input:  A = [1.0, 2.0, 3.0, 4.0]
               B = [5.0, 6.0, 7.0, 8.0]
       Output: result = 70.0  (1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0)
```
## Example 2:
```
Input:  A = [0.5, 1.5, 2.5]
               B = [2.0, 3.0, 4.0]
       Output: result = 15.5  (0.5*2.0 + 1.5*3.0 + 2.5*4.0)
```
## Constraints
 - `A` and `B` have identical lengths
 - 1 ≤ `N` ≤ 100,000,000
 - Performance is measured with `N = 100,000,000`

## `solve.cu`
```

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {}

```
