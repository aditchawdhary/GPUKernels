#### Softmax Attention

Implement a GPU program that computes the softmax attention operation for a given set of matrices. Given the query matrix Q of size M×d, key matrix K of size N×d, and value matrix V of size N×d, your program should compute the output matrix using the formula:

<img width="710" height="79" alt="image" src="https://github.com/user-attachments/assets/a4d724a7-ddec-4a1d-984d-79fb563f2f68" />

where the softmax function is applied row-wise.

#### Implementation Requirements
 - Use only GPU native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output matrix output

#### Example 1:
Input:

`Q (2×4):`

<img width="696" height="76" alt="image" src="https://github.com/user-attachments/assets/1f3b713b-225e-4f91-aae2-69731bd1612b" />


`K (3×4):`

<img width="694" height="95" alt="image" src="https://github.com/user-attachments/assets/095e75d9-7a60-4192-aa69-f221bfe0dcea" />


`V (3×4):`

<img width="706" height="93" alt="image" src="https://github.com/user-attachments/assets/6a06fec5-4472-4cea-846f-4ca3dd08628c" />


Output:

`output (2×4):`

<img width="702" height="70" alt="image" src="https://github.com/user-attachments/assets/cdff428f-9949-4957-b469-0cd7a1e5827e" />



#### Example 2:
Input:

`Q (1×2):`
<img width="702" height="48" alt="image" src="https://github.com/user-attachments/assets/b58b2bca-4889-4783-9775-6990b79d4930" />


`K (2×2):` 
<img width="706" height="69" alt="image" src="https://github.com/user-attachments/assets/ff7e6e25-2af3-4d7c-9cc4-6ad0caa2075a" />


`V (2×2):` 
<img width="702" height="70" alt="image" src="https://github.com/user-attachments/assets/9cf9e004-fa98-4ef3-bf89-55b3853ae834" />


Output:

`output (1×2):` 

<img width="900" height="35" alt="image" src="https://github.com/user-attachments/assets/a218238d-2ae2-4404-8cae-1385340d61e7" />


#### Constraints
 - Matrix Q is of size M×d and matrices K and V are of size N×d
 - 1 ≤ M, N ≤ 100,000
 - 1 ≤ d ≤ 128
 - Performance is measured with M = 512, N = 256
