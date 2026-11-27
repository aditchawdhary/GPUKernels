# GPUKernels

## Built-in Variables

- CUDA provides built-in variables automatically inside device code. They are read-only (you cannot assign/write values to them).
- `blockDim` and `gridDim` are fixed for a given launch. They come from:
  - The `<<<blocks, threads>>>` configuration, so the same kernel launched with different `blocks` and `threads` values will see them differently.
  - `blockIdx` and `threadIdx` are ***different for every thread***. That's the whole point: it's the same code everywhere, but each thread sees its own values.
  - There are **four** variables, each with `.x`, `.y`, `.z`: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`.
- The launch configuration is what sets them:

  ```cuda
  dim3 threads(16, 16);   // blockDim = (16, 16, 1)
  dim3 blocks(64, 32);    // gridDim  = (64, 32, 1)
  myKernel<<<blocks, threads>>>(...);
  ```

- Inside `myKernel`, every thread sees `blockDim.x == 16` and `gridDim.y == 32`. Each one gets its own `threadIdx` and `blockIdx`.

- `blockDim` and `gridDim` are fixed for a given launch. There are **four** built-in variables in total, each with `.x`, `.y`, `.z`:

| Variable    | Meaning                               | Type    |
|-------------|---------------------------------------|---------|
| `threadIdx` | thread's index within its block       | `uint3` |
| `blockIdx`  | block's index within the grid         | `uint3` |
| `blockDim`  | size of each block (threads per block) | `dim3`  |
| `gridDim`   | size of the grid (blocks per grid)     | `dim3`  |

- If you launch with plain integers, like `<<<10, 256>>>`, then `.y` and `.z` default to 1 for the dimensions and 0 for the indices.
- There is also `warpSize` (32 on current NVIDIA GPUs).
- They only exist in device code, i.e. `__global__` and `__device__` functions. Host/CPU code cannot see them. If a helper `__device__` function needs the index, it can read `threadIdx` directly or take it as a parameter.
- Launch 3 blocks with 4 threads each `<<<3, 4>>>`

```cuda
int idx = blockDim.x * blockIdx.x + threadIdx.x;
```

<img width="752" height="400" alt="image" src="https://github.com/user-attachments/assets/39854243-610c-471e-ab5b-6015629e16c6" />

## Matrix Multiplication
