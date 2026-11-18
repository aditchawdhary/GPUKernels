## NVFP4 Matrix Multiplication

Implement an NVFP4 matrix multiplication, the low-precision GEMM that powers state-of-the-art LLM inference on Hopper and Blackwell GPUs. 
Both operands are stored in 4-bit floating point (FP4 E2M1) with per-block FP8 (E4M3) scales along the reduction dimension, plus a single per-tensor FP32 scale. 
Given packed activations `x_q` of shape `M × K`, packed weights `w_q` of shape `N × K`, and their respective block scales, compute `y = alpha × (x × wT)` of shape `M × N` in `float16`.

FP4 E2M1 encoding: Each weight is 4 bits [sign | exp (2 bits) | mantissa (1 bit)] representing one of sixteen values: {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. The nibble-to-value mapping is:
```
  0x0 =  0.0    0x8 = -0.0
  0x1 =  0.5    0x9 = -0.5
  0x2 =  1.0    0xA = -1.0
  0x3 =  1.5    0xB = -1.5
  0x4 =  2.0    0xC = -2.0
  0x5 =  3.0    0xD = -3.0
  0x6 =  4.0    0xE = -4.0
  0x7 =  6.0    0xF = -6.0
```
**Packing**: Each byte of `x_q / w_q` stores two `FP4` values. The high nibble (bits 7–4) holds the even-index value and the low nibble (bits 3–0) holds the odd-index value.

**Block scales**: Each contiguous block of 16 `FP4` values along the K dimension shares one `E4M3` (float8) scale. The scale tensors x_scales and w_scales are passed as raw uint8 bytes holding the E4M3 bit patterns. Dequantization is:

```
x[m, k] = fp4_decode(x_q_nibble[m, k]) * e4m3_decode(x_scales[m, k // 16])
w[n, k] = fp4_decode(w_q_nibble[n, k]) * e4m3_decode(w_scales[n, k // 16])
y[m, n] = alpha * sum_k x[m, k] * w[n, k]
```
## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in y as float16

## Example
<img width="1151" height="525" alt="image" src="https://github.com/user-attachments/assets/4fa1c1b7-0a28-4ad6-8be3-9849520365b9" />


## Constraints
 - 1 ≤ `M`, `N` ≤ 32,768
 - 16 ≤ `K` ≤ 32,768
 - `K` is divisible by 16 (the NVFP4 block size)
 - All tensors are stored in row-major order
 - `Inputs:` `x_q`, `w_q`, `x_scales`, `w_scales` are `uint8`; `alpha` is `float32`
 - `Output:` `y` is `float16`
 - Performance is measured with `M = 2,048`, `N = 18,432`, `K = 3,072`
