## DPO Sequence Loss

Given precomputed sequence log probabilities for chosen and rejected responses under a current policy and a reference policy, compute the Direct Preference Optimization (DPO) loss. Each input is a one-dimensional tensor of shape `[B]`. The output is one value: the DPO loss averaged across all `B` response pairs.

Each batch entry represents one prompt and its chosen/rejected response pair. Token-level log probabilities have already been summed, and the same tokenization and masking convention was used for the current and reference policies.

Before training, a reviewer labels one response as preferred and one as rejected. Each DPO update then consumes that fixed example: prompt `x`, chosen response `y+`, and rejected response `y-`. The current and reference policies each score both responses, producing four sequence log probabilities.
<img width="1059" height="562" alt="image" src="https://github.com/user-attachments/assets/750597a2-ce61-42bc-ac17-15acbfb18869" />


## Implementation Requirements
 - Implement `solve(chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output, beta, B)` with the signature unchanged
 - Write the mean DPO loss across the batch into `output[0]`
 - Use only native features of the selected framework; external libraries are not permitted
 - All four log-probability inputs have shape `[B]`
 - Compute the loss stably as `softplus(-logits)` or an equivalent formulation
 - The reference-policy log probabilities are fixed precomputed inputs

## Loss Calculation
At batch index `i`, let `l+` and `l-` be the chosen and rejected log probabilities under the current policy. Let `l+_ref` and `l-_ref` be the corresponding values under the reference policy.
<img width="1073" height="99" alt="image" src="https://github.com/user-attachments/assets/a70409ae-a05a-4287-a615-f9a816a9f8a4" />

Subtract the reference margin from the current-policy margin, then scale by `beta`:
<img width="1058" height="51" alt="image" src="https://github.com/user-attachments/assets/c7fa695c-d985-420d-979a-7a50b78fe161" />


DPO maximizes the log-sigmoid of this preference logit. This challenge returns the equivalent loss for minimization, averaged across the batch:
<img width="1045" height="64" alt="image" src="https://github.com/user-attachments/assets/c7832aaf-f7b5-4597-87dc-5a2257d3cf44" />

 
The `softplus` form is mathematically equivalent and remains finite for very large logits.

## Examples
For `B = 4` and `beta = 0.1`: 
 
<img width="1059" height="166" alt="image" src="https://github.com/user-attachments/assets/66d2d416-06d5-42b3-a5bf-5cef016712ee" />
 
 
The reference margin is zero, so the preference logits and final loss are:
<img width="1065" height="79" alt="image" src="https://github.com/user-attachments/assets/665bfc3a-52c0-43a0-adf7-50806783dc29" />

 
## Constraints
 - 1 ≤ `B` ≤ 65,536
 - All tensors contain `32-bit` floating point values
 - 0 < `beta` ≤ 1
 - All input `log probabilities` are `finite`
 - Sequence reductions and token masks are outside the scope of this challenge
 - Performance is measured with `B = 65,536`

## Reference
Direct Preference Optimization: Your Language Model is Secretly a Reward Model, Rafailov et al. (2023)
https://arxiv.org/abs/2305.18290

## `solve.cu`
```
#include <cuda_runtime.h>

// chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output are device pointers
extern "C" void solve(const float* chosen_logps, const float* rejected_logps,
                      const float* chosen_ref_logps, const float* rejected_ref_logps, float* output,
                      float beta, int B) {}

```
