#### 2D Convolution

Write a program that performs a 2D convolution operation on the GPU. Given an input matrix and a kernel (filter), compute the convolved output. 
The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

<img width="753" height="262" alt="image" src="https://github.com/user-attachments/assets/0cdfd486-0041-4bcd-89c7-b8ac6b862586" />


The input consists of:
 - `input`: A 2D matrix of 32-bit floating-point numbers, represented as a 1D array in row-major order.
 - `kernel`: A 2D kernel (filter) of 32-bit floating-point numbers, also represented as a 1D array in row-major order.

The output should be written to the output matrix (also a 1D array in row-major order). The output matrix will have dimensions:
 - `output_rows = input_rows - kernel_rows + 1`
 - `output_cols = input_cols - kernel_cols + 1`

The convolution operation is defined as:
<img width="743" height="31" alt="image" src="https://github.com/user-attachments/assets/64955af1-ef1e-4d22-b128-108b4c059278" />


#### Implementation Requirements

Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the array output

#### Example 1:

<img width="725" height="511" alt="image" src="https://github.com/user-attachments/assets/51dbe634-efec-4430-b3e6-638e44f2b8b1" />


#### Example 2:

<img width="729" height="560" alt="image" src="https://github.com/user-attachments/assets/b10e8e93-efbb-4cf5-a579-46afb3096fda" />
 

#### Constraints
 - 1 ≤ `input_rows`, `input_cols` ≤ 3072
 - 1 ≤ `kernel_rows`, `kernel_cols` ≤ 31
 - `kernel_rows` ≤ `input_rows`
 - `kernel_cols` ≤ `input_cols`
 - Performance is measured with `input_cols` = 3,072, `input_rows` = 3,072, `kernel_cols` = 15, `kernel_rows` = 15

```
#include <cuda_runtime.h>

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {}

```
