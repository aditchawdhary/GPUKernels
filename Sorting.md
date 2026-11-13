## Sorting

Write a program that sorts an array of `32-bit` floating-point numbers in ascending order. You are free to choose any sorting algorithm.

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The sorted result must be stored back in the input data array

## Example
Input: `data = [5.0, 2.0, 8.0, 1.0, 9.0, 4.0]`, `N = 6`
Output: `data = [1.0, 2.0, 4.0, 5.0, 8.0, 9.0]`

## Constraints
 - 1 ≤ `N` ≤ 1,000,000
 - Performance is measured with `N = 1,000,000`

## `solve.cu`
```
#include <cuda_runtime.h>

// data is device pointer
extern "C" void solve(float* data, int N) {}

```
