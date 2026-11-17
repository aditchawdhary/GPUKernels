## Flash Decoding Attention

Implement flash decoding attention: the long-context decode step of LLM inference (Flash-Decoding, Split-KV, Parallel Attention). 
A small batch of `T` queries (typically 1, the token being generated) attends to a KV cache of length `S` that is often tens of thousands of tokens. For each head `h`:

<img width="1266" height="125" alt="image" src="https://github.com/user-attachments/assets/341ec1de-e22c-425a-8718-adb6662782ba" />
 
 

`Q` has shape `T × H × D`, `K` and `V` have shape `S × H × D`, and output has shape `T × H × D`. All tensors are `32-bit` floats. Unlike prefilling, `T ≪ S`, so a fast solution splits the KV cache across thread blocks, runs an online softmax per split and combines the partial outputs (flash decoding) rather than materializing a full `T × S` attention matrix.

## Implementation Requirements
 - Do not change the function signature.
 - Do not use external libraries beyond what is available in the starter.
 - Write the result to output.
 - The softmax must be numerically stable for large scores.

## Example
Input: `T = 2`, `S = 3`, `H = 1`, `D = 2`
<img width="1305" height="97" alt="image" src="https://github.com/user-attachments/assets/369f10ac-ce49-4c65-8d35-2674a4dc068c" />
  
 
Output (rounded):
<img width="1286" height="82" alt="image" src="https://github.com/user-attachments/assets/8a147833-9af0-465b-a134-0183097b1ff3" />
 

## Constraints
 - 1 ≤ `T` ≤ 16
 - 1 ≤ `S` ≤ 131,072
 - 1 ≤ `H` ≤ 64
 - 1 ≤ `D` ≤ 256
 - Input values are in `[-6, 6]`
 - Performance is measured with `T = 1`, `S = 65,536`, `H = 32`, `D = 128`

## `solve.cu`
```
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int T, int S,
                      int H, int D) {}

```
