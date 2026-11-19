## Diffusion Transformer Block

Implement a single DiT (Diffusion Transformer) block, the layer that Stable Diffusion 3, Flux, and the original DiT stack dozens of times to denoise image latents. 
Given a batch of patch token sequences `x` of shape (batch_size, seq_len, 512), a per-sample conditioning vector `c` of shape (batch_size, 512) (the timestep plus class/text embedding), and a packed weight buffer, compute the block output. 
Unlike a language model block, normalization here has no learned affine parameters: 
the scale, shift, and residual gate are predicted from `c` by an adaLN-Zero modulation layer, so every sample in the batch is normalized with different parameters.

<img width="1268" height="820" alt="image" src="https://github.com/user-attachments/assets/083a81ff-ee2a-4c6f-922f-1bbdbc874d1b" />

<img width="1252" height="752" alt="image" src="https://github.com/user-attachments/assets/f2756cc2-1d43-4a2b-a150-63de5e5769ee" />

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output tensor
 - LayerNorm uses <img width="68" height="22" alt="image" src="https://github.com/user-attachments/assets/6ac594a8-c84f-495e-ac95-0718833edee6" /> and has no learned weight or bias
 - Modulation scales by <img width="110" height="22" alt="image" src="https://github.com/user-attachments/assets/22f6b43b-b0b6-40e3-aa06-80437534f777" />

 - Self-attention is bidirectional — do not apply a causal mask
 - Use the GELU tanh approximation:
<img width="379" height="35" alt="image" src="https://github.com/user-attachments/assets/76383055-136d-428f-8857-63e920d2cb58" />

 

## Weight Layout
All block parameters are packed into a single contiguous weights buffer `(4,726,272 floats)` in the order below. All 2-D matrices are stored row-major with shape `(out_dim, in_dim)`, so <img width="85" height="27" alt="image" src="https://github.com/user-attachments/assets/649ea3b6-937d-44a2-89a2-4c134f112a17" /> lives at `weights[1575936 + i * 512 + j]`.

 | Parameter | 	Shape | 	Size | 	Offset | 
 |-----------|--------|--------|---------|     
 | <img width="40" height="26" alt="image" src="https://github.com/user-attachments/assets/dbf91260-0eb9-4325-850e-fa5f596bea2c" /> (adaLN modulation) | 	(3072, 512) | 	1,572,864 | 	0 | 
 | <img width="40" height="26" alt="image" src="https://github.com/user-attachments/assets/52a8f8a7-f40d-4405-ac70-985867b989d5" /> | (3072,) | 	3,072 | 	1,572,864
 | <img width="40" height="26" alt="image" src="https://github.com/user-attachments/assets/852e5237-003f-4e11-b5f3-ce2bccc6bac2" /> | (1536, 512) | 	786,432 | 	1,575,936
 | <img width="40" height="26" alt="image" src="https://github.com/user-attachments/assets/8ed6c089-00bb-464c-bae0-d5ab83943feb" /> | (1536,)	 | 1,536 | 	2,362,368
 | <img width="40" height="26" alt="image" src="https://github.com/user-attachments/assets/d20317e8-b254-4833-a6cc-a17acf86cbcf" /> | (512, 512)	 | 262,144 | 	2,363,904
 | <img width="40" height="26" alt="image" src="https://github.com/user-attachments/assets/8ae63fb6-3e2e-4f59-b01c-ec6c78478e0c" /> | (512,) | 	512 | 	2,626,048
 | <img width="40" height="26" alt="image" src="https://github.com/user-attachments/assets/a5acf8e7-d412-4f9f-924a-2874e72c5112" /> | (2048, 512) | 	1,048,576 | 	2,626,560
 | <img width="40" height="26" alt="image" src="https://github.com/user-attachments/assets/a81809d0-b522-4aa6-97a7-59a04be89f35" /> | (2048,) | 	2,048 | 	3,675,136
 | <img width="40" height="26" alt="image" src="https://github.com/user-attachments/assets/93a4c326-20de-4791-b87c-2985793ad407" /> | (512, 2048) | 	1,048,576 | 	3,677,184
 | <img width="40" height="26" alt="image" src="https://github.com/user-attachments/assets/45b2063a-c8b8-47cc-81a5-f679453c09f8" /> | (512,) | 	512 | 	4,725,760



## Example
With `batch_size = 1`, `seq_len = 4`, `x` drawn uniformly from `[−1, 1]`, `c` drawn uniformly from `[−2, 2]`, and randomly initialized weights:

```
Input:  x.shape       = (1, 4, 512)     # 4 patch tokens
        c.shape       = (1, 512)        # timestep + class conditioning
        weights.shape = (4,726,272,)    # packed weight buffer
        batch_size    = 1
        seq_len       = 4
Output: output.shape  = (1, 4, 512)     # denoised patch tokens
```

## Constraints
 - `d_model = 512`, `n_heads = 8`, `head_dim = 64`, `mlp_hidden = 2,048`
 - 1 ≤ `batch_size` ≤ 16
 - 1 ≤ `seq_len` ≤ 4,096
 - All tensors use `32-bit` floating point
 - Performance is measured with `batch_size = 4`, `seq_len = 1,024`
