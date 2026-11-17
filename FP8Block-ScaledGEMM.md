## FP8 Block-Scaled GEMM

Implement an **FP8 block-scaled GEMM**, the low-precision matrix multiply used in DeepSeek-V3, NVIDIA Transformer Engine and DeepGEMM for LLM inference on Hopper and Blackwell. 
Activations and weights are stored as OCP `E4M3` `float8` values with a shared `float32` scale per contiguous block of 128 elements along the reduction dimension. 
Given quantized activations `x_q` of shape `M × K`, quantized weights `w_q` of shape `N × K` and their block scales, compute `y = x × wT` of shape `M × N` in `float16`.

E4M3 encoding: each byte of `x_q` / `w_q` is an IEEE-like 8-bit float with 1 sign bit, 4 exponent bits and 3 mantissa bits (PyTorch `float8_e4m3fn` / CUDA `__nv_fp8_e4m3`). Decode each byte to a real value, then multiply by its block scale:

```
x[m, k] = e4m3_decode(x_q[m, k]) * x_scales[m, k // 128]
w[n, k] = e4m3_decode(w_q[n, k]) * w_scales[n, k // 128]
y[m, n] = sum_k x[m, k] * w[n, k]
```
x_scales has shape M × (K/128) and w_scales has shape N × (K/128), both float32. K is always divisible by 128.

## Implementation Requirements
 - Do not change the function signature.
 - Do not use external libraries beyond what is available in the starter.
 - Write the result to y as float16.

## Example
Input (`M = 2`, `N = 2`, `K = 128`, all block scales = 1): every entry of `x_q` row 0 is `E4M3` `1.0` `(0x38)`, row 1 is `E4M3` `0.5` `(0x30)`; every entry of `w_q` row 0 is `E4M3` `2.0` `(0x40)`, row 1 is `E4M3` `-1.0` `(0xB8)`.

Output (`float16`):
<img width="1082" height="83" alt="image" src="https://github.com/user-attachments/assets/24ab5b48-9791-46a2-9230-118c73c229ba" />
 

## Constraints
 - 1 ≤ `M`, `N` ≤ 16,384
 - 128 ≤ `K` ≤ 16,384 and `K` is divisible by 128
 - `E4M3` payloads are finite (no NaN); block scales are positive `float32` in [0.01, 4]
 - Performance is measured with `M = 2,048`, `N = 4,096`, `K = 4,096`

## `solve.cu`
```
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// x_q, x_scales, w_q, w_scales, y are device pointers
extern "C" void solve(const uint8_t* x_q, const float* x_scales, const uint8_t* w_q,
                      const float* w_scales, __half* y, int M, int N, int K) {}

```
