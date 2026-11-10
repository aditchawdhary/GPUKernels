### Weight Dequantization

Implement a GPU program that "dequantizes" a weight matrix on the GPU. You are given an input matrix `X` of shape `[M, N]` containing quantized values and a scale matrix `S` of shape `[ceil(M/T), ceil(N/T)]`, where `T` is the tile size.

For each element `X_i,j`, the corresponding scale factor is `S_row,col` where `row = floor(i/T)` and `col=floor(j/T)`. The output `Y_i,j` should be computed as: 
```
Y_i,j = X_i,j * S_row,col
```

### Implementation Requirements
 - External libraries are not permitted
 - The `solve` function signature must remain unchanged
 - The final result must be stored in the output buffer `Y`

### Example 1:
```
Input:
M = 4, N = 4, TILE_SIZE = 2
X = [
  [10, 10,  5,  5],
  [10, 10,  5,  5],
  [ 2,  2,  8,  8],
  [ 2,  2,  8,  8]
]
S = [
  [0.5, 2.0],
  [4.0, 0.25]
]

Output:
Y = [
  [ 5.0,  5.0, 10.0, 10.0],
  [ 5.0,  5.0, 10.0, 10.0],
  [ 8.0,  8.0,  2.0,  2.0],
  [ 8.0,  8.0,  2.0,  2.0]
]
Explanation:
Tile (0,0) of X is multiplied by S[0,0] (0.5).
Tile (0,1) of X is multiplied by S[0,1] (2.0).
Tile (1,0) is multiplied by S[1,0] (4.0).
Tile (1,1) is multiplied by S[1,1] (0.25).
```

### Constraints
 - 1 ≤ `M`, `N` ≤ 8192
 - `TILE_SIZE ∈ {16, 32, 64, 128}`
 - Performance is measured with `M = 8,192`, `N = 8,192`, `TILE_SIZE = 128`

```
#include <cuda_runtime.h>

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {}

```
