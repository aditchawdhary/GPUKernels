## 2D Jacobi Stencil

Given a 2D grid of 32-bit floating point values, apply one iteration of the 5-point Jacobi stencil: each interior cell of the output is set to the average of its four cardinal neighbors (top, bottom, left, right) from the input grid. 
Boundary cells (first/last row and column) are copied unchanged from the input to the output.

<img width="1232" height="345" alt="image" src="https://github.com/user-attachments/assets/60c8fbc8-2ebc-47a3-8f19-b6e58ebea387" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The `solve` function signature must remain unchanged
 - The final result must be stored in `output`
 - Read exclusively from `input` and write exclusively to `output` (do not update input)

## Example:
<img width="1224" height="473" alt="image" src="https://github.com/user-attachments/assets/cc4ccfc1-9076-42dd-ae4b-eebc8b5b820a" />
 

## Constraints
 - 1 ≤ `rows, cols` ≤ 16,384
 - Input values are in the range `[-100, 100]`
 - All values are `32-bit` floats
 - Performance is measured with `rows = 8,192`, `cols = 8,192`

## `solve.cu`
```
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {}

```
