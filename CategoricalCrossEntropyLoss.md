#### Categorical Cross Entropy Loss

Implement a GPU program to calculate the categorical cross-entropy loss for a batch of predictions. Given a matrix of predicted logits `Z` of size `N x C` and a vector of true class labels true_labels of size `N`, compute the average cross-entropy loss over the batch. 
The loss for a single sample `j` with logits `z_j = [z_j1, ..., z_jC]` and true label `y_j` is calculated using the numerically stable formula:

<img width="761" height="87" alt="image" src="https://github.com/user-attachments/assets/339412ef-71b3-4d6f-963f-271f155ce718" />

The final output stored in the loss variable should be the average `loss` over the `N` samples:

<img width="760" height="66" alt="image" src="https://github.com/user-attachments/assets/7149d36f-5c42-4bbb-b120-3abf168ae50a" />
 
The input parameters are `logits`, `true_labels`, `N` (number of samples), and `C` (number of classes). The result should be stored in `loss` (a pointer to a single float).

#### Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - The final result (average loss) must be stored in loss

Example 1:
```
Input:  N = 2, C = 3
        logits = [[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]]
        true_labels = [1, 1]
Output: loss = [0.3548926]
```

Example 2:
```
Input:  N = 3, C = 4
        logits = [[-0.5, 1.5, 0.0, 1.0], [2.0, -1.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]
        true_labels = [3, 0, 1]
Output: loss = [0.98820376]
```

#### Constraints
 - 1 ≤ N ≤ 10,000
 - 2 ≤ C ≤ 1,000
 - -10.0 ≤ logits[i, j] ≤ 10.0
 - 0 ≤ true_labels[i] ≤ C
 - Performance is measured with N = 10,000

```
#include <cuda_runtime.h>

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {}

```





