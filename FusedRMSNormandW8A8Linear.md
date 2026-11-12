## Fused RMSNorm and W8A8 Linear

Implement the fused **RMSNorm + W8A8 linear layer** used by int8 LLM inference (SmoothQuant-style W8A8). Each token row of `x` is RMS-normalized and scaled by `gamma`, then quantized on the fly to `int8` with its own symmetric per-token scale. 
The `int8` activations are multiplied by an `int8` weight matrix that was quantized offline with one scale per output channel, the products are accumulated as integers, and the result is dequantized to `float32`.

<img width="873" height="266" alt="image" src="https://github.com/user-attachments/assets/a0e3ee6c-3ab2-4523-a465-8867ab04000b" />

## Implementation Requirements
Implement `solve(x, gamma, W, w_scale, output, M, N, K)`:

 - `x`: shape `(M, K)`, `float32`, one row per token
 - `gamma`: shape `(K,)`, `float32`, the RMSNorm weight
 - `W`: shape `(N, K)`, `int8` with values in `[-127, 127]`
 - `w_scale`: shape `(N,)`, `float32`, one scale per output channel
 - `output`: shape `(M, N)`, `float32`

<img width="884" height="255" alt="image" src="https://github.com/user-attachments/assets/e6f17501-e732-4a3b-8a89-3cebf89e96cf" />
  
where round is round-to-nearest (ties to even). If `s_i = 0`, every `q_i,k = 0`. Then
 
<img width="874" height="70" alt="image" src="https://github.com/user-attachments/assets/49ad3c4b-b379-40d7-acc4-a79c51e26650" />
 

 - Use only native features of the framework you choose (no external libraries); keep the function signature unchanged
 - Write the result into output; every element must be written
 - The activations must really be quantized to `int8` levels; the sums <img width="105" height="29" alt="image" src="https://github.com/user-attachments/assets/45adc928-9f22-465c-888f-aa004caed8e1" />
 are integers and should be accumulated exactly (for example in `int32`)
 - Floating-point rounding may move an activation that lies right on a rounding boundary to the neighbouring level; the tolerance allows for this

## Example
<img width="877" height="351" alt="image" src="https://github.com/user-attachments/assets/f6595a81-cb7a-43ca-95be-14a2019fcd24" />

 

## Constraints
 - 1 ≤ `M` ≤ 8,192
 - 1 ≤ `N` ≤ 16,384
 - 1 ≤ `K` ≤ 8,192
 - `x` is drawn from a standard normal distribution, possibly scaled, with a few outlier channels or tokens up to about 60× larger; rows may be all zero
 - `gamma` is in `[0.5, 1.5]`; `w_scale` is about `1 / (127 · √K)`
 - Performance is measured with `M = 4,096`, `N = 4,096`, `K = 4,096`

## `solve.cu`

```
#include <cuda_runtime.h>

// x, gamma, W, w_scale, output are device pointers
extern "C" void solve(const float* x, const float* gamma, const int8_t* W, const float* w_scale,
                      float* output, int M, int N, int K) {}

```
