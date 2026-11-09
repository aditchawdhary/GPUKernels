2D Subarray Sum

Implement a program that computes the sum of a 2D subarray of 32-bit integers. You are given an input 2D array input of length `N x M`, and two row indices `S_ROW` and `E_ROW` and two column indices `S_COL` and `E_COL`. 
`S_ROW`, `E_ROW`, `S_COL` and `E_COL` are inclusive, 0-based start and end indices — compute the sum of `input[S_ROW..E_ROW][S_COL..E_COL]`.

### Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output variable

#### Example 1:
```
Input:  input = [[1, 2, 3],
                 [4, 5, 1]]

        S_ROW = 0, E_ROW = 1, S_COL = 1, E_COL = 2
Output: output = 11
```

#### Example 2:
```
Input:  input = [[5, 10],
                 [5, 2]]
        S_ROW = 0, E_ROW = 0, S_COL = 1, E_COL = 1
Output: output = 10
```

#### Constraints
 - 1 ≤ `N`, `M` ≤ 10,000
 - 1 ≤ `input[i]` ≤ 10
 - 0 ≤ `S_ROW` ≤ `E_ROW` ≤ `N - 1`
 - 0 ≤ `S_COL` ≤ `E_COL` ≤ `M - 1`
 - Performance is measured with `M = 10,000`, `N = 10,000`
