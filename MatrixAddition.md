## Matrix Addition

Implement a program that performs element-wise addition of two `N x N` matrices containing 32-bit floating point numbers on a GPU. 
The program should take two input matrices of equal dimensions and produce a single output matrix containing their element-wise sum.

## Implementation Requirements
 - External libraries are not permitted
 - The `solve` function signature must remain unchanged
 - The final result must be stored in matrix `C`

## Example 1:
```
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
        B = [[5.0, 6.0],
             [7.0, 8.0]]
Output: C = [[6.0, 8.0],
             [10.0, 12.0]]
```

## Example 2:
```
Input:  A = [[1.5, 2.5, 3.5],
             [4.5, 5.5, 6.5],
             [7.5, 8.5, 9.5]]
        B = [[0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5]]
Output: C = [[2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0],
             [8.0, 9.0, 10.0]]
```

## Constraints
 - Input matrices `A` and `B` have identical dimensions
 - `1 ≤ N ≤ 4096`
 - All elements are 32-bit floating point numbers
 - Performance is measured with `N = 4,096`
