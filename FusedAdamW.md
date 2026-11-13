## Fused AdamW

Implement a single fused AdamW optimizer step. Given a flat parameter vector params, its gradient grad, and the optimizer's running first and second moment estimates `m` and `v`, update params, `m`, and `v` in place for one training step. AdamW is the optimizer behind most modern LLM training runs (GPT, LLaMA, and friends); fusing its handful of elementwise operations into a single kernel avoids several redundant read–write passes over what can be a very large parameter tensor, which is exactly what a fused=True optimizer does under the hood.

For `step t (1-indexed)`, the update is:
<img width="850" height="213" alt="image" src="https://github.com/user-attachments/assets/2a82144e-2ed4-4393-a108-63c64d72852a" />

 
 
 
 

where `θ` is params, `g` is grad, `η` is the learning rate `lr`, `λ` is `weight_decay`, and `β1`, `β2`, `ε` are `beta1`, `beta2`, `eps`. The weight decay is decoupled: it is applied directly to params rather than added into grad, which is the detail that distinguishes AdamW from plain Adam with L2 regularization.

## Implementation Requirements
 - Do not change the function signature.
 - Do not use external libraries beyond what is available in the starter.
 - params, m, and v are updated in place; there is no separate output buffer.
 - Apply the decoupled weight decay to params using its pre-update value, before computing the moment updates and the adaptive step, matching the order shown above.
 - t is always ≥ 1, so the bias-correction denominators are never zero.

## Example 1:
```
N = 2, t = 1
params = [1.0, -1.0]
grad   = [0.1, -0.1]
m      = [0.0,  0.0]
v      = [0.0,  0.0]
lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.0

m_new = 0.9*0 + 0.1*grad      = [0.01, -0.01]
v_new = 0.999*0 + 0.001*grad^2 = [1e-5,  1e-5]
m_hat = m_new / (1 - 0.9)      = [0.1,  -0.1]
v_hat = v_new / (1 - 0.999)    = [0.01,  0.01]
step  = lr * m_hat / (sqrt(v_hat) + eps) = [0.1, -0.1]

params -> [0.9, -0.9]
```

## Example 2:
```
N = 1, t = 1
params = [1.0]
grad   = [0.5]
m      = [0.0]
v      = [0.0]
lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01

params after decoupled decay: 1.0 * (1 - 0.1*0.01) = 0.999
m_new = 0.05, v_new = 0.00025
m_hat = 0.5,  v_hat = 0.25
step  = 0.1 * 0.5 / (0.5 + 1e-8) = 0.1

params -> 0.999 - 0.1 = 0.899
```

## Constraints
 - 1 ≤ `N` ≤ 16,777,216
 - 1 ≤ `t` ≤ 100,000
 - `params`, `grad`, `m`, and `v` are float32
 - Values in params and grad are in the range [−10, 10]
 - 0 < `lr` ≤ 1
 - 0 ≤ `beta1, beta2` < 1
 - 1e-12 ≤ `eps` ≤ 1e-2
 - 0 ≤ `weight_decay` ≤ 1
 - `v` is always non-negative; `m` may be negative
 - Performance is measured with `N = 16,777,216`, `t = 1,000`

## `solve.cu`
```
#include <cuda_runtime.h>

// params, grad, m, v are device pointers
extern "C" void solve(float* params, const float* grad, float* m, float* v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) {}

```


