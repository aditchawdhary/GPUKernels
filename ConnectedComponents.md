## Connected Components

Given an undirected graph with `N` vertices (numbered `0 to N−1`) and `M` edges represented as two arrays `src` and `dst` of `32-bit integers`, compute the connected component label for every `vertex`. 
The label for `vertex i` must equal the minimum `vertex ID` in the connected component containing `i`.

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - Store the label for vertex `i` in `labels[i]`
 - Each edge (`src[e]`, `dst[e]`) is undirected; vertex IDs satisfy `0 ≤ src[e], dst[e] < N`
 - The label for every vertex must be the minimum vertex ID in its component

## Example
```
Input:
  N = 5, M = 3
  src = [0, 1, 3]
  dst = [1, 2, 4]

Graph edges: 0-1, 1-2, 3-4

Output:
  labels = [0, 0, 0, 3, 3]
```

Explanation:
  Vertices 0, 1, 2 are connected (min ID = 0).
  Vertices 3, 4 are connected (min ID = 3).

## Constraints
 - 1 ≤ `N` ≤ 1,000,000
 - 1 ≤ `M` ≤ 5,000,000
 - 0 ≤ `src[e]`, `dst[e]` < `N` for all edges
 - The graph may contain self-loops and duplicate edges
 - All values are `32-bit integers (int32)`
 - Performance is measured with `N = 1,000,000`, `M = 5,000,000`

```
#include <cuda_runtime.h>

// src, dst, labels are device pointers
extern "C" void solve(const int* src, const int* dst, int* labels, int N, int M) {}

```
