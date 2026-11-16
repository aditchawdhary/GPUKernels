## INT4 Weight-Only Quantized MatMul

Implement a weight-only INT4 quantized matrix multiplication `(W4A16)`, a core kernel used in modern LLM inference. Given a `float16` activation matrix x of shape `M × K` and a weight matrix stored in packed `INT4` format, 
compute the output matrix `y = x × WT` of shape `M × N`, where `W` is the dequantized `float16` weight matrix of shape `N × K`.

**Packing format:** Each byte of `w_q` stores two `INT4` weights. The high nibble (bits 7–4) holds weight `w[n, 2i]` and the low nibble (bits 3–0) holds `w[n, 2i+1]`. 
INT4 values are stored unsigned in the range `[0, 15]` with an offset of 8, so the signed weight is nibble − 8, giving values in [−8, 7].

**Dequantization:** Weights are dequantized group-wise. Each contiguous block of group_size weights along the K dimension shares one float16 scale:
```
W[n, k] = (w_q_nibble[n, k] - 8) * scales[n, k // group_size]
```

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in y

## Example
<img width="1089" height="624" alt="image" src="https://github.com/user-attachments/assets/6cee6552-5c0d-4e54-90e2-4fcd4370b212" />

 

## Constraints
 - 1 ≤ `M`, `N` ≤ 8,192
 - 1 ≤ `K` ≤ 8,192
 - `K` is divisible by `2` and by `group_size`
 - `group_size` ∈ `{2, 4, 8, 16, 32, 64, 128}`
 - All tensors are stored in row-major order
 - Input dtype: `x` and `scales` are `float16`; `w_q` is `uint8`
 - Output dtype: `y` is `float16`
 - Performance is measured with `M` = `4,096`, `N` = `4,096`, `K` = `4,096`, `group_size` = `128`
