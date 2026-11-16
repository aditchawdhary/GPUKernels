## MoE Token Dispatch

In Mixture-of-Experts (MoE) neural networks, a router assigns each input token to one of `E` expert networks. 
Before those experts can execute in parallel, tokens must be dispatched: reorganized from their original sequence order into contiguous per-expert buffers. 
Given `T` tokens of feature dimension `D` and their expert assignments, fill each expert's slice of the output buffer with the tokens routed to that expert, preserving the original token order within each expert's group.

<img width="1002" height="281" alt="image" src="https://github.com/user-attachments/assets/24b8a26f-f3c5-435d-8a13-5e3f30eb63c6" />


## Implementation Requirements
Your `solve` function signature must not be changed. No external libraries beyond those already imported. Write outputs to `dispatched_x` and `token_counts` in place.

For each expert `e`, `dispatched_x[e, :token_counts[e], :]` must contain exactly the tokens assigned to expert `e`, in the same relative order they appear in the original token sequence (i.e., token index ascending). 
Entries at positions `token_counts[e]` and beyond may be left as zero.

## Example
Given `T` = 4 tokens of dimension `D` = 3 routed to `E` = 2 experts (capacity = 4):

Input tokens `x`:
<img width="998" height="140" alt="image" src="https://github.com/user-attachments/assets/3f62f513-81e1-403f-bd11-d98a9f7ba9bf" />

 

Expert assignments:
```
expert_idx = [0, 1, 0, 1]
```
Output token counts:
```
token_counts = [2, 2]
```

<img width="1006" height="221" alt="image" src="https://github.com/user-attachments/assets/e500823f-5c8b-4c6c-8483-074639ff74d8" />

 

## Constraints
 - 1 ≤ `T` ≤ 65,536
 - 1 ≤ `D` ≤ 2,048
 - 2 ≤ `E` ≤ 64
 - `capacity` = `T` (always sufficient for any routing assignment)
 - All values in expert_idx are in the range [0, E − 1]
 - `x` values are `float32`; expert_idx and token_counts are `int32`
 - Performance is measured with `T = 16,384`, `D = 512`, `E = 8`
