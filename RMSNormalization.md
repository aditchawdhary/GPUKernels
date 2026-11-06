### RMS Normalization

Implement RMS Normalization forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the normalized output using a scalar scale (gamma) and shift (beta) parameter.

RMS Normalization computes:
 
<img width="256" height="163" alt="image" src="https://github.com/user-attachments/assets/36881089-0b48-4dfa-a876-15a4cd4abf25" />
 
 

### Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output tensor


### Example 1:
```
Input:  input = [1.0, 2.0, 3.0, 4.0]  (N=4)
        gamma = 1.0
        beta = 0.0
        eps = 1e-5
Output: output = [0.36514813, 0.73029625, 1.0954444, 1.4605925 ]
```

### Example 2:
```
Input:  input = [1.0, 2.0, 3.0]  (N=3)
        gamma = 1.0
        beta = 0.0
        eps = 1e-5
Output: output = [0.46290955, 0.9258191, 1.3887286]
```

### Constraints
 - 1 ≤ N ≤ 100,000
 - eps = 1e-5
 - -100.0 ≤ input values ≤ 100.0
 - 0.1 ≤ gamma ≤ 10.0
 - -10.0 ≤ beta ≤ 10.0
 - Performance is measured with N = 100,000



 
