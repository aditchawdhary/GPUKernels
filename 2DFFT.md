## 2D FFT

Compute the 2D Discrete Fourier Transform (2D DFT) of a complex-valued signal stored on the GPU. 
Given a 2D complex input signal of shape `M × N`, compute its 2D DFT spectrum using the row-column decomposition: apply a 1D DFT along each row, then a 1D DFT along each column of the result. 
All values are `32-bit` floating point.

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in `spectrum`
 - The input and output are stored as 1D arrays of interleaved real and imaginary parts in row-major order: element `x[m, n]` has its real part at index `2*(m*N + n)` and imaginary part at index `2*(m*N + n) + 1`

## Example
<img width="738" height="522" alt="image" src="https://github.com/user-attachments/assets/a822d916-0bee-4177-a8e4-c89a37cda782" />

## Constraints
 - 1 ≤ `M`, `N` ≤ 4096
 - Signal values are `32-bit` floating point (real and imaginary parts)
 - Performance is measured with `M` = 2,048, `N` = 2,048

## `solve.cu`
```
#include <cuda_runtime.h>

// signal, spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int M, int N) {}

```
