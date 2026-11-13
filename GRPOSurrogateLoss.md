## GRPO Surrogate Loss
For each of `B` prompts, GRPO compares the rewards of `G` sampled responses. It uses each response's reward relative to the rest of its group as an advantage, rather than learning a separate critic. 
That advantage is applied to every one of the response's `S` tokens in a clipped policy objective with a reference-policy KL regularizer.

At position `[b, g, s]`, `log_pi`, `log_pi_old`, and `log_ref` all score the same sampled token `y[b,g,s]`, conditioned on prompt b and the preceding response tokens.

<img width="1184" height="427" alt="image" src="https://github.com/user-attachments/assets/a78db76c-b825-4ae4-98f5-fa734c82de39" />


## Implementation Requirements
 - Implement `solve(rewards, log_pi, log_pi_old, log_ref, output, clip_eps, beta, B, G, S)` with the signature unchanged
 - Write the scalar mean loss into `output[0]`
 - Use only native features of the selected framework; external libraries are not permitted
 - Rewards have shape `[B, G]`; all log-probability tensors have shape `[B, G, S]`
 - Token masks and padding handling are outside the scope of this challenge; all S positions participate in the reduction
 - `Rewards`, `log_pi_old`, and `log_ref` are fixed precomputed inputs; do not differentiate through them

## Loss Calculation
For one prompt `b`, normalize the `G` response rewards using the group's population mean and standard deviation. Let `R` denote rewards.
<img width="1156" height="254" alt="image" src="https://github.com/user-attachments/assets/f553b281-2587-45d6-92a7-d1d47a92a64d" />

 
 
 
This is an outcome-level GRPO advantage. The same advantage is broadcast to every token of response g:
<img width="1153" height="58" alt="image" src="https://github.com/user-attachments/assets/7bff8924-056b-46db-9012-9bf562050e31" />



Compare the current policy with the old policy that sampled the response.
<img width="1176" height="38" alt="image" src="https://github.com/user-attachments/assets/f4143b60-d511-4f14-bcdf-cc692c62c593" />



Clamp that ratio around one, then take the less optimistic of the clipped and unclipped values. Let epsilon denote clip_eps. This min applies correctly to both positive and negative advantages.
<img width="1178" height="103" alt="image" src="https://github.com/user-attachments/assets/117a661f-2eb4-4690-a032-e5c742b863bc" />

GRPO also penalizes divergence from the frozen reference policy. Let beta be the KL coefficient. The non-negative k3 sample estimator is:
<img width="1163" height="90" alt="image" src="https://github.com/user-attachments/assets/c2bf6d56-e78b-4e9d-a99d-2a8519cd8234" />

GRPO maximizes the clipped objective minus this penalty. Because this challenge returns a loss for minimization, write the negated mean over all prompts, responses, and tokens to output[0]:
<img width="1192" height="70" alt="image" src="https://github.com/user-attachments/assets/4af7dd9f-7895-4c96-bade-0391b29ebedc" />

 
## Examples
For `B = 1`, `G = 2`, `S = 2`, `clip_eps = 0.2`, and `beta = 0.01`:

<img width="1201" height="288" alt="image" src="https://github.com/user-attachments/assets/2e6a4fb4-0c8d-4f43-a52b-8dcc09ee465f" />
 
 
## Constraints
 - 1 ≤ `B` ≤ 256
 - 2 ≤ `G` ≤ 32
 - 1 ≤ `S` ≤ 16,384
 - All tensors contain 32-bit floating point values
 - 0 ≤ `clip_eps` < 1 and 0 ≤ `beta` ≤ 1
 - Both `log_pi - log_pi_old` and `log_ref - log_pi` lie in `[-16, 16]`, so every exponential is finite in `float32`
 - Performance is measured with `B = 64`, `G = 16`, `S = 4,096`

## Reference
DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO), Shao et al. (2024)

## `solve.cu`
```
#include <cuda_runtime.h>

// rewards, log_pi, log_pi_old, log_ref, output are device pointers
extern "C" void solve(const float* rewards, const float* log_pi, const float* log_pi_old,
                      const float* log_ref, float* output, float clip_eps, float beta, int B, int G,
                      int S) {}

```
