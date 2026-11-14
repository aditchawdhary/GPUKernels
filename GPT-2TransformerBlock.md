## GPT-2 Transformer Block

Implement a single GPT-2 transformer decoder block. Given an input tensor 
 of shape (seq_len, 768) and a packed weight buffer containing all block parameters, compute the output using pre-norm architecture with multi-head self-attention and a feed-forward network with GELU activation.

<img width="1143" height="687" alt="image" src="https://github.com/user-attachments/assets/c9391851-8698-45aa-a860-c9e10ce7bd17" />


The block uses GPT-2's pre-norm architecture: LayerNorm is applied before each sub-layer (attention and feed-forward), not after. At a high level:
<img width="1207" height="91" alt="image" src="https://github.com/user-attachments/assets/226ca0d4-17ef-425e-bd26-bf93fc7acbf4" />

 
where the sub-layers are defined as:
<img width="1175" height="272" alt="image" src="https://github.com/user-attachments/assets/2d79f349-94ca-4fc1-aa3a-f0ae4c2b4bc2" /> 
 
Expanding into individual steps:
<img width="1213" height="267" alt="image" src="https://github.com/user-attachments/assets/b8ce7487-cbea-47b1-a088-2a2c90603d73" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output tensor
 - LayerNorm uses <img width="42" height="21" alt="image" src="https://github.com/user-attachments/assets/e2e714da-7b49-43e6-942a-17b47cc1fc63" />

 - Use the GELU tanh approximation: <img width="200" height="36" alt="image" src="https://github.com/user-attachments/assets/962b7b11-cfc7-405e-9629-ef3313d1286e" />

 
## Weight Layout
All block parameters are packed into a single contiguous weights buffer (7,087,872 floats) in the following order. Index into the buffer using the offsets below (e.g. 
 is at weights[1536 + i * 2304 + j]). All 2D matrices are stored in row-major order.

 | Parameter | 	Shape | 	Size | 	Offset | 
 |-----------|--------|--------|---------|     
 | <img width="25" height="21" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/e2f3e695-9374-4f0b-bdec-4e8d89d28660" /> (LN1 weight) | 	(768,) | 	768 | 	0 | 
 | <img width="30" height="23" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/e7901b75-cd9d-4b31-89d6-ddae152f5add" /> (LN1 bias)  	| (768,) | 	768	 | 768 | 
 | <img width="49" height="27" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/d18f6a94-5f67-4e9f-bab7-6270b6d06b03" /> | (768, 2304) | 	1,769,472 |	1,536  | 
 | <img width="44" height="29" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/47ca462e-937c-4659-91a7-bafdac73a8e7" /> | (2304,) | 	2,304 | 	1,771,008 | 
 | <img width="53" height="26" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/5a418eba-ec5c-4059-ba63-dea33c6319c4" /> | (768, 768) | 	589,824 | 	1,773,312 | 
 | <img width="44" height="29" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/90e4b3b9-2dc3-4450-89b7-cfa699a6cbfe" /> | (768,) | 	768 | 	2,363,136 | 
 | <img width="25" height="21" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/f215c2a9-acbb-4c9c-b706-aaaca23f4e5a" /> (LN2 weight) | 	(768,) | 	768 | 	2,363,904 | 
 | <img width="31" height="24" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/b3a46ba8-3306-4295-b6fd-094c05fcca57" /> (LN2 bias) | 	(768,) | 	768 | 	2,364,672 | 
 | <img width="47" height="23" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/876f4db3-2dfd-4ec0-a736-bb1940bd3748" /> | (768, 3072) | 	2,359,296 | 	2,365,440 | 
 | <img width="35" height="28" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/caa37798-73ca-4ccc-bd89-3f538a4afb0c" /> | (3072,) | 	3,072 | 	4,724,736 | 
 | <img width="58" height="28" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/75b9fbb5-53ff-4a02-9dc1-1ab932af7835" /> | (3072, 768) | 	2,359,296 | 	4,727,808 | 
 | <img width="45" height="29" scale="0.8" alt="image" src="https://github.com/user-attachments/assets/0c0cb221-596c-4c51-b782-651550a74c15" /> | (768,) | 	768 | 	7,087,104 | 

## Example
```
With seq_len = 4, x uniformly drawn from [−1, 1], and weights randomly initialized (see Weight Layout for the packing structure):

Input:  x.shape       = (4, 768)       # 4 token embeddings
        weights.shape = (7,087,872,)   # packed weight buffer
        seq_len       = 4
Output: output.shape  = (4, 768)       # transformed token embeddings
```

## Constraints
 - `d_model = 768`, `n_heads = 12`, `ffn_dim = 3,072` `(GPT-2 124M architecture)`
 - 1 ≤ `seq_len` ≤ 4,096
 - All tensors use `32-bit` floating point
 - Performance is measured with `seq_len = 1,024`
