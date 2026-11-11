## Kimi Delta Attention

<img width="813" height="659" alt="image" src="https://github.com/user-attachments/assets/e150508a-df3c-4032-b117-496830abcba8" />

Implement Kimi Delta Attention (KDA), the linear attention mechanism introduced in "Kimi Linear: An Expressive, Efficient Attention Architecture". KDA extends the gated delta rule with a channel-wise forget gate: the recurrent state 
 is a fast-weight associative memory that is decayed per feature channel, corrected by the delta rule, and then queried. For each batch, the state starts at 
 and is updated for t = 1, ..., seq_len:

<img width="822" height="83" alt="image" src="https://github.com/user-attachments/assets/9973c055-ba83-4044-b054-3f1c62dca55e" />
 
where 
 are the query, key, and value vectors at step t, 
 is the channel-wise decay gate, and 
 is the update strength. The update can be read as online gradient descent on the reconstruction loss 
 
 with per-channel weight decay: old associations fade channel by channel while the delta rule overwrites the value currently associated with 
.

All tensors are `float32`. `Q`, `K`, `V`, `alpha`, and `output` have shape `batch × seq_len × d`; `beta` has `shape batch × seq_len`. Key vectors are L2-normalized, so 
 (or 
). Each batch element is an independent sequence.

<img width="822" height="259" alt="image" src="https://github.com/user-attachments/assets/b1ae7545-5d2e-47c1-b727-a7e2eff5bef4" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The `solve` function signature must remain unchanged
 - The final result must be stored in `output`

## Example

<img width="828" height="326" alt="image" src="https://github.com/user-attachments/assets/005a4a8a-6c9d-4b86-95aa-92d0b7a102b0" />

 

## Constraints
 - 1 ≤ `batch` ≤ 8
 - 1 ≤ `seq_len` ≤ 16,384
 - 1 ≤ `d` ≤ 128
 - 0 < `alpha values` ≤ 1, 0 ≤ `beta values` ≤ 1
 - Rows of K are L2-normalized (unit norm or all-zero)
 - Performance is measured with `batch = 4`, `seq_len = 8,192`, `d = 128`

## References
 - Moonshot AI, "Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv:2510.26692) — introduces Kimi Delta Attention
 - Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv:2412.06464) — the head-wise gated delta rule KDA refines
 - "Linear Attention: Kimi Delta Attention" — walkthrough of the KDA recurrence and chunkwise algorithm


```
#include <cuda_runtime.h>

// Q, K, V, alpha, beta, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* alpha,
                      const float* beta, float* output, int batch, int seq_len, int d) {}

```

