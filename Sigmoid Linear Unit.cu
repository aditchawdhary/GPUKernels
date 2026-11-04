Sigmoid Linear Unit
Easy
Implement the SiLU (Sigmoid Linear Unit) activation function forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the output using the elementwise formula.

SiLU is defined as:
 
 

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output tensor
Example 1:
Input:  input = [0.5, 1.0, -0.5]  (N=3)
Output: output = [0.3112295, 0.731059, -0.1887705]
Example 2:
Input:  input = [-1.0, -2.0, -3.0, -4.0, -5.0]  (N=5)
Output: output = [-0.26894143 -0.23840584 -0.14227763 -0.07194484 -0.03346425]
Constraints
1 ≤ N ≤ 10,000
-100.0 ≤ input values ≤ 100.0
Performance is measured with N = 50,000
