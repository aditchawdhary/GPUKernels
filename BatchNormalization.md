#### Batch Normalization

Implement batch normalization forward pass for 2D input tensors. Given an input tensor of shape [N, C] where N is the batch size and C is the number of features, compute the normalized output using learnable scale (gamma) and shift (beta) parameters.

For each feature channel j, batch normalization computes:
<img width="756" height="246" alt="image" src="https://github.com/user-attachments/assets/b21e8ab5-98ab-4531-b31f-2e8df3f787ae" />
 
#### Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output tensor

#### Example 1:
```
Input:  input = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  (N=3, C=2)
        gamma = [1.0, 1.0]
        beta = [0.0, 0.0]
        eps = 1e-5
Output: output = [[-1.224, -1.224], [0.0, 0.0], [1.224, 1.224]]
```

#### Example 2:
```
Input:  input = [[0.0, 1.0], [2.0, 3.0]]  (N=2, C=2)
        gamma = [2.0, 0.5]
        beta = [1.0, -1.0]
        eps = 1e-5
Output: output = [[-1.0, -1.5], [3.0, -0.5]]
```

#### Constraints
 - 1 ≤ N ≤ 10,000
 - 1 ≤ C ≤ 1,024
 - eps = 1e-5
 - -100.0 ≤ input values ≤ 100.0
 - 0.1 ≤ gamma values ≤ 10.0
 - -10.0 ≤ beta values ≤ 10.0
 - Performance is measured with N = 5,000
