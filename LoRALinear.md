#### LoRA Linear

Implement a LoRA (Low-Rank Adaptation) linear layer forward pass. Given an input matrix `x` of shape `batch × d_in`, a base weight matrix `W` of shape `d_out × d_in`, 
a LoRA down-projection matrix `A` of shape `rank × d_in`, and a LoRA up-projection matrix `B` of shape `d_out × rank`, compute `output = x × WT + lora_scale × (x × AT) × BT`. All tensors are `float32`.

<img width="1155" height="324" alt="image" src="https://github.com/user-attachments/assets/e377c682-ec56-4343-9dd5-72b41941f97a" />

#### Implementation Requirements
 - Implement the solve function; do not change its signature.
 - Do not use external libraries beyond those provided.
 - Write the result into output.

#### Examples
#### Example 1:

 <img width="1171" height="91" alt="image" src="https://github.com/user-attachments/assets/2cb13a16-9ee9-4ae9-b510-cc8bf37a3251" /> 

With lora_scale = 0.5:

<img width="1158" height="59" alt="image" src="https://github.com/user-attachments/assets/0d323e2e-5923-40da-abd9-91c5cfba349e" />
  
  
  
 

#### Constraints
 - 1 ≤ batch ≤ 1,024
 - 1 ≤ d_in, d_out ≤ 8,192
 - 1 ≤ rank ≤ 256; rank < min(d_in, d_out)
 - All tensors are float32 on GPU.
 - Performance is measured with batch = 256, d_in = 4,096, d_out = 4,096, rank = 64
