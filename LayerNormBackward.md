## Layer Norm Backward

Implement the backward pass of Layer Normalization, the foundational normalization layer used in GPT-2, BERT, ViT, and standard transformer architectures. 
The forward pass normalized each row (token) of the float32 input `x` of shape `(N, C)` using per-row mean and reciprocal standard deviation `rstd`, and applied per-feature learned scale gamma and shift `beta`. 
Given `x`, `gamma`, `mean`, `rstd`, and the upstream gradient `dy` of shape (N, C), compute the input gradient `dx` of shape `(N, C)` and the parameter gradients `dgamma` and `dbeta`, each of shape `(C,)`.

<img width="744" height="251" alt="image" src="https://github.com/user-attachments/assets/c5eabc96-24f5-4510-a345-d7eb05adf5d7" />

 
<img width="746" height="262" alt="image" src="https://github.com/user-attachments/assets/335569ae-cafb-4274-b226-862b79ff4d25" /> 

#### Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The results must be stored in `dx`, `dgamma`, and `dbeta`
 - `dx`, `dgamma`, and `dbeta` are not zero-initialized; every element must be written

#### Example

<img width="744" height="357" alt="image" src="https://github.com/user-attachments/assets/b38f3fae-ce42-4da3-98a6-e9b39d574698" />

 

#### Constraints
 - 1 ≤ `N` ≤ 65,536
 - 1 ≤ `C` ≤ 16,384
 - All tensors are `float32`
 - Values of `x` are in [−250, 250]; values of `dy` are in [−10, 10]; values of `gamma` are in [−2, 2]
 - Performance is measured with `N` = 16,384, `C` = 4,096
