#### Row-wise Argmax

Implement row-wise argmax over a matrix of 32-bit floating point logits, the core of greedy decoding in language models. 
Given logits of shape B × V (batch size by vocabulary size), write to output[b] the column index of the largest value in row b. 
If several columns share the maximum value, return the smallest such index.

#### Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - The final result must be stored in output as 32-bit integers

#### Example:
Input (B = 2, V = 4):

<img width="757" height="66" alt="image" src="https://github.com/user-attachments/assets/e417256d-ec0c-4a2a-8035-89b2447172a5" />

Output:

<img width="749" height="50" alt="image" src="https://github.com/user-attachments/assets/78d45564-6676-4796-af13-6b034d0f36e5" />

 
In the second row, columns 1 and 2 tie at -0.5, so the smaller index 1 is returned.

#### Constraints
 - 1 ≤ B ≤ 1,024
 - 1 ≤ V ≤ 262,144
 - Logits are finite 32-bit floating point numbers (no NaN)
 - Performance is measured with B = 256, V = 128,000

```
#include <cuda_runtime.h>

// logits, output are device pointers
extern "C" void solve(const float* logits, int* output, int B, int V) {}

```
