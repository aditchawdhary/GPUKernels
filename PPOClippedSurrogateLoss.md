## PPO Clipped Surrogate Loss

Given precomputed advantages and log probabilities from the current and previous policies, compute the scalar PPO clipped-surrogate loss. 
For each response token, PPO compares the current policy with the policy that generated the data, limits how far that ratio can move from one, and averages the resulting surrogate objective across the batch.

## Implementation Requirements
 - Implement `solve(advantages, log_pi, log_pi_old, output, clip_eps, B, S)` with the signature unchanged
 - Write the single scalar loss into `output[0]`
 - Use only native features of the selected framework; external libraries are not permitted
 - All input tensors have shape `[B, S]` and the output has shape `[1]`
 - All `S` positions are valid response tokens; padding and token masks are outside the scope of this challenge
 - `advantages` and `log_pi_old` are fixed precomputed inputs; do not differentiate through them

## Loss Calculation
At position `[b, s]`, `log_pi` and `log_pi_old` are the log probabilities assigned to the same response token by the current and previous policies. Let `A` denote advantages, and let `epsilon` denote `clip_eps`.

First, compute the probability ratio between the current and previous policies:

<img width="289" height="43" alt="image" src="https://github.com/user-attachments/assets/df7d97ce-ba3c-43a3-8c67-2e4345d99b8b" />

Clamp that ratio to PPO's trust region:

<img width="256" height="36" alt="image" src="https://github.com/user-attachments/assets/2a8aa5ef-dd5e-4bba-8836-846ff43740aa" />

The per-token surrogate takes the less optimistic of the unclipped and clipped values. This min is intentional: it applies correctly to both positive and negative advantages.

<img width="342" height="62" alt="image" src="https://github.com/user-attachments/assets/b50a3198-a276-4840-9708-52c0048d862d" />


PPO maximizes this surrogate objective. Because this challenge returns a loss for minimization, negate its mean over every batch and sequence position:

<img width="339" height="87" alt="image" src="https://github.com/user-attachments/assets/f0cb9973-c2b0-470d-900d-edcdcdad10f3" />
 
## Examples
<img width="1301" height="386" alt="image" src="https://github.com/user-attachments/assets/b97f73fa-9ab7-4ead-aa3e-7a33af27058c" />

 
 
 
 
## Constraints
 - 1 ≤ `B` ≤ 256
 - 1 ≤ `S` ≤ 16,384
 - All tensors contain `32-bit` floating point values
 - `0 ≤ clip_eps < 1`
 - `log_pi - log_pi_old` lies in [-16, 16], so every exponential is finite in `float32`
 - Performance is measured with `B = 256`, `S = 16,384`

## Reference
Proximal Policy Optimization Algorithms, Schulman et al. (2017)

## `solve.cu`
```
#include <cuda_runtime.h>

// advantages, log_pi, log_pi_old, output are device pointers
extern "C" void solve(const float* advantages, const float* log_pi, const float* log_pi_old,
                      float* output, float clip_eps, int B, int S) {}

```
