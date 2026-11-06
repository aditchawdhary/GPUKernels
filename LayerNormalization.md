#### Layer Normalization

Implement the forward pass of layer normalization for a 2D input tensor. Given an input tensor of shape [N, C] where N is the batch size and C is the number of features, normalize each sample independently across its C features, then apply learnable scale (weight) and shift (bias) parameters. 
Layer normalization is a core building block of transformer architectures.

For each sample `i`, layer normalization computes:
 
 <img width="743" height="202" alt="image" src="https://github.com/user-attachments/assets/6920f2cc-0a9e-4afa-8aec-9294f4ec3ae5" />

 <img width="753" height="270" alt="image" src="https://github.com/user-attachments/assets/fa698d29-9f67-4e61-8c2d-3aa10570ba35" />

Each row is its own normalization group (here N=3, C=6)
μ₀, σ₀² over row 0
μ₁, σ₁² over row 1
μ₂, σ₂² over row 2
Statistics are reduced along the feature axis, never across rows.
weight and bias are per-feature, shared by every row.

#### Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output tensor

#### Example
<img width="752" height="476" alt="image" src="https://github.com/user-attachments/assets/7439a43a-aa7f-42c1-abfc-ca9475c2e4bb" />



#### Constraints
 - 1 ≤ N ≤ 65,536
 - 1 ≤ C ≤ 4,096
 - eps = 1e-5
 - Input values are in the range [-100.0, 100.0]
 - Weight values are in the range [0.1, 10.0]
 - Bias values are in the range [-10.0, 10.0]
 - Performance is measured with N = 65,536, C = 512
