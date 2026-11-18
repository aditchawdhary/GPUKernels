## All-Pairs Shortest Paths

Given a weighted directed graph of `N` vertices represented as an `N × N` distance matrix, compute the shortest path distance between every pair of vertices using the Floyd-Warshall algorithm. 
The matrix is stored as a flat array in row-major order: `dist[i * N + j]` is the weight of the directed edge from vertex `i` to vertex `j`. 
A value of `+infinity` means no direct edge exists. The diagonal is always zero. For each intermediate vertex `k` from `0 to N - 1` (in order), update all pairs:

<img width="740" height="51" alt="image" src="https://github.com/user-attachments/assets/70aa23e5-2944-45dc-ad08-ba2797a976d3" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The `solve` function signature must remain unchanged
 - The final result must be stored in `output`

## Example:
```
Input: N = 4
dist = [
  0,   5, inf,  10,   // row 0: edges from vertex 0
  inf, 0,   3, inf,   // row 1: edges from vertex 1
  inf, inf, 0,   1,   // row 2: edges from vertex 2
  inf, inf, inf, 0    // row 3: edges from vertex 3
]

Output:
output = [
  0,   5,   8,   9,   // shortest paths from vertex 0
  inf, 0,   3,   4,   // shortest paths from vertex 1
  inf, inf, 0,   1,   // shortest paths from vertex 2
  inf, inf, inf, 0    // shortest paths from vertex 3
]

Explanation:
- output[0][2] = 8   (path 0 -> 1 -> 2, cost 5 + 3 = 8)
- output[0][3] = 9   (path 0 -> 1 -> 2 -> 3, cost 5 + 3 + 1 = 9, beats direct 0 -> 3 = 10)
- output[1][3] = 4   (path 1 -> 2 -> 3, cost 3 + 1 = 4)
```

## Constraints
 - 1 ≤ `N` ≤ 4,096
 - Edge weights are finite `float32` values or `+infinity` (no edge)
 - The `input` contains no negative cycles
 - The diagonal satisfies `dist[i * N + i] = 0` for all `i`
 - `dist` and `output` are flat arrays of `N × N` floats in row-major order
 - Performance is measured with `N = 2,048`
