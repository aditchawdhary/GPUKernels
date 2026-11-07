### 3D Convolution

Implement a program that performs a 3D convolution operation. Given a 3D input volume and a 3D kernel (filter), compute the convolved output. The convolution should use a "valid" boundary condition (no padding).

For a 3D convolution, the output at position `(i, j, k)` is given by:
<img width="756" height="81" alt="image" src="https://github.com/user-attachments/assets/074588ec-e1a3-4956-a4f6-82ac6565bc20" />


The input consists of: 
 - `input:` A 3D volume of 32-bit floats, as a 1D array (row-major, then depth).
 - `kernel:` A 3D kernel of 32-bit floats, as a 1D array (row-major, then depth).
 - `input_depth`, `input_rows`, `input_cols`: Dimensions of the input.
 - `kernel_depth`, `kernel_rows`, `kernel_cols`: Dimensions of the kernel.

Output:
 - `output:` A 1D array (row-major, then depth) storing the result.

Output dimensions:
 - `output_depth = input_depth - kernel_depth + 1`
 - `output_rows = input_rows - kernel_rows + 1`
 - `output_cols = input_cols - kernel_cols + 1`

#### Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in output

#### Examples

<img width="762" height="583" alt="image" src="https://github.com/user-attachments/assets/142276a9-ce43-4da6-81df-3ff465b1e387" />

<img width="757" height="432" alt="image" src="https://github.com/user-attachments/assets/299b95dc-5f8c-4072-b11e-6b48ea6b64fa" />


#### Constraints
 - 1 ≤ input_depth, input_rows, input_cols ≤ 256
 - 1 ≤ kernel_depth, kernel_rows, kernel_cols ≤ 5
 - kernel_depth ≤ input_depth
 - kernel_rows ≤ input_rows
 - kernel_cols ≤ input_cols
 - Performance is measured with input_cols = 128, input_rows = 128, kernel_cols = 5, kernel_rows = 5

```
#include <cuda_runtime.h>

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {}
```
