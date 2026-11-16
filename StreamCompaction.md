## Stream Compaction
Given a 1D array `A` of `N` 32-bit floating point numbers, compact all positive elements (`A[i] > 0`) to the front of the output array out, preserving their original relative order. 
Fill any remaining positions with `0.0`. Stream compaction is a fundamental GPU primitive used throughout rendering, sparse computation, and collision detection.

## Implementation Requirements
 - Use only native GPU features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The first k positions of out must contain the k elements of A where A[i] > 0, in their original order
 - Positions k through N−1 of out must be 0.0
 - Elements where A[i] = 0.0 are not selected

## Example
```
Input:  A = [1.0, -2.0, 3.0, 0.0, -1.0, 4.0]
Output: out = [1.0, 3.0, 4.0, 0.0, 0.0, 0.0]
```

## Constraints
 - 1 ≤ `N` ≤ 100,000,000
 - −1000.0 ≤ `A[i]` ≤ 1000.0
 - `out` is pre-allocated with `N` elements, initialised to `0.0
 - Performance is measured with `N = 50,000,000`

## `solve.cu`
```
#include <cuda_runtime.h>

// A, out are device pointers
extern "C" void solve(const float* A, int N, float* out) {}

```
