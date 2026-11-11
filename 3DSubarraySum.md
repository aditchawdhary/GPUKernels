## 3D Subarray Sum

Implement a program that computes the sum of a 3D subarray of 32-bit integers. You are given an input 3D array input of length `N x M x K`, and two depth indices `S_DEP` and `E_DEP` and two row indices `S_ROW` and `E_ROW` and two column indices `S_COL` and `E_COL`. `S_DEP`, `E_DEP`, `S_ROW`, `E_ROW`, `S_COL` and `E_COL` are inclusive, 0-based start and end indices — compute the sum of `input[S_DEP..E_DEP][S_ROW..E_ROW][S_COL..E_COL]`.

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output variable

## Example 1:
```
Input:  input = [[[1, 2, 3],
                  [4, 5, 1]],
                 [[1, 1, 1],
                  [2, 2, 2]]]
        N = 2, M = 2, K = 3
        S_DEP = 0, E_DEP = 1, S_ROW = 0, E_ROW = 0, S_COL = 1, E_COL = 2
Output: output = 7
```

## Example 2:
```
Input:  input = [[[5, 10],
                  [5, 2],
                  [2, 2]]]
        N = 1, M = 3, K = 2
        S_DEP = 0, E_DEP = 0, S_ROW = 0, E_ROW = 2, S_COL = 1, E_COL = 1
Output: output = 14
```

## Constraints
 - 1 ≤ `N`, `M`, `K` ≤ 500
 - 1 ≤ `input[i]` ≤ 10
 - 0 ≤ `S_DEP` ≤ `E_DEP` ≤ `N - 1`
 - 0 ≤ `S_ROW` ≤ `E_ROW` ≤ `M - 1`
 - 0 ≤ `S_COL` ≤ `E_COL` ≤ `K - 1`
 - Performance is measured with `K = 500`, `M = 500`, `N = 500`

## `solution.cu`
```
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP,
                      int S_ROW, int E_ROW, int S_COL, int E_COL) {}
```
