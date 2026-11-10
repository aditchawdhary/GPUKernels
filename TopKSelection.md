### Top K Selection

Implement a GPU program that, given a `1D` array input of `32-bit` floating point numbers of length `N`, selects the `k` largest elements and writes them in descending order to the output array of length `k`.

### Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - The final result must be stored in the output array

### Example 1:
```
  Input:
  input = [1.0, 5.0, 3.0, 2.0, 4.0]
  N = 5
  k = 3

  Output:
  output = [5.0, 4.0, 3.0]
```
  
### Example 2:
```
  Input:
  input = [7.2, -1.0, 3.3, 8.8, 2.2]
  N = 5
  k = 2

  Output:
  output = [8.8, 7.2]
```
  
### Constraints
 - 1 ≤ `N` ≤ 100,000,000
 - 1 ≤ `k` ≤ `N`
 - All values in input are `32-bit floats`
 - Performance is measured with `N` = 50,000,000`, `k = 100`

```
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {}
```
