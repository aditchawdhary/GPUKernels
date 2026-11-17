## BFS Shortest Path

Implement a program that finds the shortest path in an unweighted 2D grid using Breadth-First Search (BFS). Given a grid with obstacles and start/end positions, return the minimum number of steps needed to reach the destination.

<img width="994" height="320" alt="image" src="https://github.com/user-attachments/assets/d7d1db30-1dfc-449d-983b-22a95b591c9e" />

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - Return the shortest path length, or -1 if no path exists
 - Grid cells with value 0 are free, cells with value 1 are obstacles
 - Movement is allowed in 4 directions: up, down, left, right

## Example 1:
```
Input:
  grid (4x4) = [
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
  ]
  start_row = 0, start_col = 0
  end_row = 3, end_col = 3

Output: 6

Explanation: One possible shortest path:
(0,0) → (0,1) → (0,2) → (1,2) → (2,2) → (2,3) → (3,3)
```

## Example 2:
```
Input:
  grid (3x3) = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 0]
  ]
  start_row = 0, start_col = 0
  end_row = 0, end_col = 2

Output: -1

Explanation: No path exists due to obstacles completely blocking the way.
```

## Constraints
 - 1 ≤ `rows`, `cols` ≤ 1000
 - Grid values are either 0 (free) or 1 (obstacle)
 - Start and end positions are guaranteed to be within bounds and on free cells (value 0)
 - Start and end positions may be the same (return 0 in this case)
 - Performance is measured with `cols = 500`, `rows = 500`

## `solve.cu`
```
#include <cuda_runtime.h>

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row,
                      int start_col, int end_row, int end_col) {}

```
