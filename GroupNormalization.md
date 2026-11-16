## Group Normalization

Implement Group Normalization for 4D activation tensors, the normalization layer used by Stable Diffusion U-Nets and many ResNet variants. Given an input tensor `X` of shape `(N, C, H, W)`, the channels are split into `G` contiguous groups of `C/G` channels each. 
For every (batch, group) pair, the mean and variance are computed over all `(C/G) × H × W` elements, the activations are normalized, then scaled and shifted by per-channel parameters gamma and beta.

<img width="1157" height="598" alt="image" src="https://github.com/user-attachments/assets/031dff23-4390-4c32-b9d9-208266f9762f" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the Y tensor

## Example 1:
```
Input:  N=1, C=4, H=2, W=2, G=2, eps=1e-5
        X[0,0] = [[1, 1], [1, 1]]
        X[0,1] = [[3, 3], [3, 3]]
        X[0,2] = [[2, 2], [2, 2]]
        X[0,3] = [[6, 6], [6, 6]]
        gamma = [1, 1, 1, 1]
        beta  = [0, 0, 0, 0]
Output: Y[0,0] = [[-1, -1], [-1, -1]]
        Y[0,1] = [[ 1,  1], [ 1,  1]]
        Y[0,2] = [[-1, -1], [-1, -1]]
        Y[0,3] = [[ 1,  1], [ 1,  1]]
Note:   Group 0 = channels {0, 1}: mean = 2, var = 1, std = 1
        Group 1 = channels {2, 3}: mean = 4, var = 4, std = 2
```

## Example 2:
```
Input:  N=1, C=2, H=1, W=2, G=2, eps=1e-5
        X[0,0] = [[1, 3]]
        X[0,1] = [[2, 6]]
        gamma = [2, 1]
        beta  = [0, 0]
Output: Y[0,0] = [[-2,  2]]
        Y[0,1] = [[-1,  1]]
Note:   G=C, so each channel is its own group (Instance Norm).
        Channel 0: mean=2, var=1, std=1
        Channel 1: mean=4, var=4, std=2
```

## Constraints
 - 1 ≤ `N` ≤ 32
 - 1 ≤ `C` ≤ 1,024 and `C` is divisible by `G`
 - 1 ≤ `G` ≤ `C`
 - 1 ≤ `H`, `W` ≤ 128
 - `eps` = 1e-5
 - -100.0 ≤ input values ≤ 100.0
 - 0.1 ≤ `gamma` values ≤ 10.0
 - -10.0 ≤ `beta` values ≤ 10.0
 - Performance is measured with `N` = 8, `C` = 512, `H` = 64, `W` = 64, `G` = 32

## `solve.cu`
```
#include <cuda_runtime.h>

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N, int C,
                      int H, int W, int G, float eps) {}

```
