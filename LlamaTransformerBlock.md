## Llama Transformer Block

Implement a single Llama-style transformer decoder block. Given an input tensor 
 of shape `(seq_len, 512)`, a packed weight buffer, and precomputed RoPE tables, compute the output using pre-norm architecture with Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), and a SwiGLU feed-forward network.

<img width="955" height="745" alt="image" src="https://github.com/user-attachments/assets/09b64f6b-a456-49db-8cfb-b7584dbb0d97" />


The block follows Llama's pre-norm architecture. Unlike GPT-2, it uses RMSNorm (no mean subtraction, no additive bias), Grouped Query Attention with 8 query heads and 2 key/value heads, Rotary Position Embeddings applied to Q and K, and a SwiGLU feed-forward network. None of the linear projections have bias terms.

<img width="947" height="83" alt="image" src="https://github.com/user-attachments/assets/501a93cf-a15f-41f0-a26d-91ebd6fd36ed" />

 
The sub-operations in detail:
<img width="957" height="538" alt="image" src="https://github.com/user-attachments/assets/553bb6df-8fad-4a9e-854e-12f3a87ce377" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output tensor
 - RMSNorm uses <img width="51" height="24" alt="image" src="https://github.com/user-attachments/assets/b8f792bb-5679-4f3f-a16e-087d2d1bb858" />, no additive bias
 - Apply causal masking: position attends only to positions 
 - Repeat K and V heads (GQA groups) before computing attention
 - `cos` and `sin` have shape `(seq_len, 32)` — apply them to both `Q` and `K` heads independently

## Weight Layout
All parameters are packed into a single contiguous weights buffer (2,819,072 floats) in the order below. All 2-D matrices are stored row-major with shape (out_dim, in_dim). There are no bias terms.

| Parameter | Shape | Size | 	Offset | 
|-----------|-------|------|---------|
| w1 (RMSNorm 1 scale)	 | (512,) | 	512 |	0 | 
| W_Q | (512, 512)	 | 262,144 | 	512 | 
| W_K | (128, 512)	 | 65,536 | 	262,656 | 
| W_V | (128, 512)	 | 65,536 | 	328,192 | 
| W_O | (512, 512)	 | 262,144 | 	393,728 | 
| w2 (RMSNorm 2 scale) | 	(512,) | 	512 | 	655,872 | 
| W_gate | (1408, 512)	 | 720,896 | 	656,384 | 
| W_up | (1408, 512)	 | 720,896 | 	1,377,280 | 
| W_down | (512, 1408)	 | 720,896 | 	2,098,176 | 

## Example
With `seq_len = 4`, x drawn uniformly from `[−1, 1]`, and randomly initialized weights:

```
Input:  x.shape       = (4, 512)       # 4 token hidden states
        weights.shape = (2,819,072,)   # packed weight buffer
        cos.shape     = (4, 32)        # precomputed RoPE cosines
        sin.shape     = (4, 32)        # precomputed RoPE sines
        seq_len       = 4
Output: output.shape  = (4, 512)       # transformed token hidden states
```

## Constraints
 - `d_model` = 512, `n_q_heads` = 8, `n_kv_heads` = 2, `head_dim` = 64, `ffn_hidden` = 1,408
 - 1 ≤ `seq_len` ≤ 4,096
 - All tensors use `32-bit` floating point
 - Performance is measured with `seq_len = 2,048`
