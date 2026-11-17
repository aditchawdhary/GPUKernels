## Parallel Reverse Scan (GAE)

Given reward and value sequences with shape `[B, S]`, compute the Generalized Advantage Estimate (GAE) for every batch element and timestep. GAE combines one-step temporal-difference errors with a reverse discounted accumulation, creating a dependency that runs from the end of each sequence toward the beginning.

<img width="1131" height="275" alt="image" src="https://github.com/user-attachments/assets/043e81f4-63e6-4348-b4d7-33e20a6e7f8a" />

## How GAE Is Computed
The calculation has two stages. First, compute the one-step temporal-difference error at every position. The value after the final position is defined as zero:
<img width="1120" height="68" alt="image" src="https://github.com/user-attachments/assets/53cf6e5a-4b71-44ba-8eb9-3a236ac565c8" />

 
Next, let `c = gamma × lambda`. The advantage at position `t` is the current TD error plus the decayed advantage from the next position:
<img width="1125" height="46" alt="image" src="https://github.com/user-attachments/assets/64ff09ac-5289-449e-8d68-acacc0a56fdb" />

Therefore, implementations can compute all delta values in parallel, then perform a reverse scan over the sequence using the recurrence above. For long sequences, the reverse scan can be split into blocks: each block computes its local scan, then receives the decayed carry from the block to its right.

For the example below, `gamma = 0.9` and `lambda = 0.5`, so `c = 0.45`:
<img width="1127" height="86" alt="image" src="https://github.com/user-attachments/assets/a9e55cbc-dc73-4211-af09-d152c165f66c" />

 
## Implementation Requirements
 - Implement `solve(rewards, values, advantages, gamma, lam, B, S)` with the signature unchanged
 - Write the result into the preallocated advantages tensor
 - Use only native features of the selected framework; external libraries are not permitted
 - For the final timestep, use a bootstrap value of zero
 - Use `float32` arithmetic for all tensors
 - Every sequence terminates at `S`; padding, per-token terminal flags, and masks are outside the scope of this challenge

## Examples
For `B = 1`, `S = 4`, `gamma = 0.9`, and `lambda = 0.5`:

<img width="1124" height="96" alt="image" src="https://github.com/user-attachments/assets/76ebc0cd-5989-4e23-b528-ca67e18f87a5" />

 
The final timestep uses `next_value = 0`, so its temporal-difference error is `4.0 - 2.0 = 2.0`. Each earlier advantage includes the discounted contribution from later timesteps.

## Constraints
 - 1 ≤ `B` ≤ 256
 - 1 ≤ `S` ≤ 65,536
 - rewards, values, and advantages have shape `[B, S]`
 - All tensors contain `32-bit` floating point values
 - `gamma` and `lambda` are in the range `[0, 1]`
 - Performance is measured with `B = 64`, `S = 4,096`

## Reference
High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. (2015) https://arxiv.org/abs/1506.02438

## `solve.cu`

```
#include <cuda_runtime.h>

// rewards, values, advantages are device pointers
extern "C" void solve(const float* rewards, const float* values, float* advantages, float gamma,
                      float lam, int B, int S) {}

```
