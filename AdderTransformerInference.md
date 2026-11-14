## Adder Transformer Inference

Run batched autoregressive inference for a 10-parameter transformer that adds two 10-digit numbers. 
Given prompts of shape `[batch_size, 31]` (int32) and a 10-float weight buffer, write output logits of shape `[batch_size, 11, 10]` — one logit row per decode step over the 10-digit vocabulary (0–9). 
All tensors are float32 except the int32 prompts.

The model comes from the AdderBoard competition for the smallest autoregressive transformer that adds 10-digit numbers at `≥99%` accuracy. 
It encodes carry propagation in 10 learned parameters via RoPE geometry, tied embeddings, and SwiGLU gating.
<img width="1104" height="610" alt="image" src="https://github.com/user-attachments/assets/fc432a4c-222a-46c7-b701-b45e9c9e3728" />



## Model Architecture
Single-layer pre-norm transformer. Hidden dim 2, 1 head, head dim 2, vocab 10 (digits 0–9), tied input/output embeddings.

Each step runs the full sequence `[batch_size, seq_len, 2]` through:

1. **Token Embedding** (2 parameters: `w0`, `w1`)
<img width="1130" height="87" alt="image" src="https://github.com/user-attachments/assets/8e9b7fa5-0751-4710-a89c-6b968a2b688a" />

 

2. **Unit RMSNorm** (no parameters)
<img width="1123" height="77" alt="image" src="https://github.com/user-attachments/assets/69ddf0cf-dc3b-4746-80ef-6d3c790700c2" />

 

3. **Self-Attention** (3 parameters: q0, q1, v0)

Projections applied to the normed hidden state h with shape [*, 2]:
<img width="1118" height="82" alt="image" src="https://github.com/user-attachments/assets/0665a094-0f82-4c22-808b-5afb784a493c" />  
 
After projection, Q and K are each normalized with Unit RMSNorm, then RoPE is applied with angular frequency ω = 2π/19:
<img width="1124" height="76" alt="image" src="https://github.com/user-attachments/assets/80c14c29-73f5-4591-851c-42488efa0da1" /> 

Scaled dot-product attention with causal mask uses scale factor:
<img width="1095" height="68" alt="image" src="https://github.com/user-attachments/assets/bdd825c8-c35d-41df-a3c0-2e0c6db2ff61" />

 
where <img width="63" height="22" alt="image" src="https://github.com/user-attachments/assets/192ae3cf-0147-492a-9881-8f250326eeaf" /> is the head dimension and <img width="31" height="24" alt="image" src="https://github.com/user-attachments/assets/c4e3d43f-5090-49c6-ace9-56b4ae382137" /> is the QK-norm scale constant (see weight table below for exact value).

The output projection maps [attn_0, attn_1] → [0, attn_0] (no parameters), followed by a residual connection.

4. **MLP** (3 parameters: a, c, carry)

Applied to the unit-RMSNorm of the post-attention hidden state:

<img width="1099" height="191" alt="image" src="https://github.com/user-attachments/assets/e3b5754c-3ed8-4bd5-9c93-83ff6cff23ab" />
 
followed by a residual connection.

5. Final **RMSNorm** (2 parameters: n0, n1)

Standard RMSNorm with learned weight:
<img width="1106" height="75" alt="image" src="https://github.com/user-attachments/assets/bb912a9c-f2af-463e-bb8a-2445a0a91450" />

 

6. Output **Logits** (tied with embedding)
<img width="1125" height="59" alt="image" src="https://github.com/user-attachments/assets/cf1316cc-f89b-4a1d-af93-3ebb381f367f" />


## Autoregressive Decoding
Starting from the 31-token prompt, repeat 11 times:

1. Run the full forward pass on the current sequence
2. Extract logits at the last position → store in output
3. Append `argmax(logits)` as the next token

The sequence grows from length 31 to 42 over the 11 decode steps.

## Weight Layout

| Offset | Size   | Name  | Description                 |
|--------|--------|-------|-----------------------------|
|   0 	 | 2 |	embed | Embedding: `e(d) = [w0 - w1*d², -d]` |
|   2	  | 2 | q_proj | Q projection weights `[q0, q1]`      |
|   4	  | 1 | v_proj | V projection weight `v0`             |
|   5	  | 2 | gate   |	MLP gate weights `[a, c]`              |
|   7	  | 1 | carry  |	MLP carry weight                   |
|   8	  | 2 | norm   |	Final RMSNorm weight `[n0, n1]`        |

## Token Encoding
Each input pair `(a, b)` of 10-digit numbers is encoded as a 31-token sequence:

```
[0, a_rev_0, ..., a_rev_9, 0, 0, 0, 0, 0, 0, 0, 0, 0, b_rev_0, ..., b_rev_9, 0]
```
where `a_rev` and `b_rev` are the digits in least-significant-first order, zero-padded to 10 digits. The model then generates 11 output tokens (digits of the sum, also least-significant-first).

## Implementation Requirements
 - Implement `solve(prompts, output, weights, batch_size)` with the exact signature shown (JAX exception: `solve(prompts, weights, batch_size)` returns the output tensor directly)
 - Do not use any external libraries beyond what the framework provides
 - The function must write logits into the output buffer (except JAX, which returns it)
 - Architecture constants are fixed: `vocab_size = 10`, `hidden_dim = 2`, `head_dim = 2`, `num_heads = 1`, `prompt_len = 31`, `decode_steps = 11`
 - RMSNorm `epsilon = 10−6`
 - RoPE angular frequency `ω = 2π/19`
 - Attention scale = `(1/√2) · S² where S²` = `ln(10) / (√2 · (cos(0.3ω) − cos(0.7ω)))`
 - SiLU activation: `silu(x) = x · sigmoid(x)`

## Example
```
With batch_size = 2 and pairs (3, 5), (99, 1):

Input prompts (shape [2, 31]):
  [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  [0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Output logits shape: [2, 11, 10]
  (logits at each of 11 decode steps over 10 digit classes)

Expected decoded tokens (via argmax):
  Pair (3, 5):   sum = 8       → [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  Pair (99, 1):  sum = 100     → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
```

## Constraints
 - `batch_size`: 1 ≤ `batch_size` ≤ 100,000
 - `prompts`: `32-bit` integer tensor, values in `[0, 9]`
 - `weights`: `32-bit` float tensor with exactly 10 elements
 - `output`: `32-bit` float tensor of shape `[batch_size, 11, 10]`
 - Input numbers are in range `[0, 9,999,999,999]` (10-digit unsigned integers)
 - Performance is measured with `batch_size = 100,000`

## `solve.cu`
```
#include <cuda_runtime.h>

// prompts, output, weights are device pointers
extern "C" void solve(const int* prompts, float* output, const float* weights, int batch_size) {}

```
