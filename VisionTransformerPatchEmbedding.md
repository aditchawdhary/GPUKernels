## Vision Transformer Patch Embedding

Implement the patch embedding stem of a Vision Transformer — the layer that turns a batch of images into a sequence of tokens for ViT, CLIP, SigLIP and DiT. 
Given a batch of B images of shape `(C, H, W)`, split every image into non-overlapping `P × P` patches, project each flattened patch to a `D-dimensional` embedding with `patch_weight` and `patch_bias`, prepend the learned `cls_token`, and add the learned positional embedding `pos_embed`. 
All tensors are `float32`, and `H` and `W` are always divisible by `P`.

<img width="770" height="296" alt="image" src="https://github.com/user-attachments/assets/d32efe25-30c2-43d9-a2a5-e59d56cbd240" />

<img width="1410" height="305" alt="image" src="https://github.com/user-attachments/assets/18f5c9fb-df6a-4b53-855d-338da1ed6c5e" />


## Implementation Requirements
 - Implement the `solve` function with the signature unchanged.
 - Do not use external libraries beyond the framework provided.
 - Write the result into `output` in-place.
 - All tensors are contiguous and row-major: `images` is `(B, C, H, W)`, `patch_weight` is `(D, C, P, P)`, `patch_bias` and `cls_token` are `(D,)`, `pos_embed` is `(N+1, D)`, and `output` is `(B, N+1, D)`.

## Example
<img width="1414" height="751" alt="image" src="https://github.com/user-attachments/assets/f14fad55-4ea8-4aec-b6e0-b37bd9571205" />


## Constraints
 - 1 ≤ `B` ≤ 128
 - 1 ≤ `C` ≤ 4
 - 1 ≤ `H`, `W` ≤ 1,024, both divisible by `P`
 - 1 ≤ `P` ≤ 32
 - 1 ≤ `D` ≤ 1,024
 - All tensors are `float32` on the GPU
 - Image values are in the range `[-10, 10]`
 - Performance is measured with `B = 64`, `C = 3`, `H = 224`, `W = 224`, `P = 16`, `D = 768`
