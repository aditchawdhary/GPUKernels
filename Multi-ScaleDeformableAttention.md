## Multi-Scale Deformable Attention

Implement multi-scale deformable attention, the kernel at the heart of detectors such as Deformable DETR, Grounding DINO and BEVFormer. 
Instead of attending to every position of a feature map, each query predicts a handful of sampling locations per attention head and per feature-pyramid level, reads the feature map at those (fractional) locations with bilinear interpolation, and combines the samples with pre-computed attention weights. 
The feature maps of all `num_levels` levels are concatenated into one `value` tensor, so the kernel must turn a level index plus a normalized coordinate into a row of that flat tensor.

The levels are stored back to back in value, level l having shape `(H_l, W_l)` in row-major order, so its first row sits at the exclusive prefix sum `level_start[l]` of the level sizes `H_l × W_l`. 
For a query `q`, head `h`, level `l` and point `p`, the normalized location `(loc_x, loc_y)` is mapped to the pixel grid of its own level and interpolated:

<img width="1141" height="357" alt="image" src="https://github.com/user-attachments/assets/609b269b-0a5f-481d-b67a-8d72222abe1a" />
 

Sampling locations are not clamped: a location may fall outside `[0, 1]`, in which case the corners that land outside the level's grid contribute zero (zero padding, as in `grid_sample` with `align_corners=False`). 
The attention weights are already normalized by the caller and must be used exactly as given. All feature data is `float32`; `spatial_shapes` is `int32`.

<img width="1122" height="571" alt="image" src="https://github.com/user-attachments/assets/9f02eef5-f338-4b11-8ddb-6791f6badbb7" />


## Implementation Requirements
 - Implement the function `solve(value, spatial_shapes, sampling_loc, attn_weight, output, num_queries, num_heads, head_dim, num_levels, num_points)`.
 - Do not change the function signature or use external libraries beyond the standard GPU framework
 - `value` has shape `(total_spatial, num_heads, head_dim)`, where `total_spatial` is the sum of `H_l × W_l` over all levels.
 - `spatial_shapes` has shape `(num_levels, 2)` and holds `(H_l, W_l)` per level as `int32`; the level offsets are the exclusive prefix sum of `H_l × W_l`.
 - `sampling_loc` has shape `(num_queries, num_heads, num_levels, num_points, 2)`, the last dimension being `(loc_x, loc_y)` normalized by the width and height of that level.
 - `attn_weight` has shape (num_queries, num_heads, num_levels, num_points) and is applied as given; do not re-normalize it.
 - Use bilinear interpolation with the `half-pixel` convention above and zero padding for corners outside the level grid.
 - Write the result into the provided output buffer of shape `(num_queries, num_heads, head_dim)`.

## Example
With `num_queries = 2`, `num_heads = 1`, `head_dim = 1`, `num_levels = 2`, `num_points = 1` and `spatial_shapes = [(2, 2), (1, 1)]`, the five rows of value are the `2×2` map of level 0 followed by the `1×1` map of `level 1`:

  
  <img width="1127" height="95" alt="image" src="https://github.com/user-attachments/assets/f2bb8fb4-8bd3-4924-9653-97c9ec4840ed" />

  
 

Sampling locations (one point per level, per query) and attention weights:
  
 <img width="1117" height="77" alt="image" src="https://github.com/user-attachments/assets/a78ce26b-3b27-481b-8131-979b443dd722" />


Output `(2×1×1)`:
 <img width="1125" height="82" alt="image" src="https://github.com/user-attachments/assets/22fb0593-b3f3-4815-a1a8-4363e28cad4d" />


Query 0 samples the center of level 0 at `x = y = 0.5`, which averages all four pixels to 2.5, and the single pixel of level 1 at x = y = 0, giving 10; the weighted sum is `0.5 × 2.5 + 0.5 × 10 = 6.25`. Query 0's level-1 location is exactly on a pixel center, so `dx = dy = 0` and the three phantom corners get weight zero. 
Query 1 samples level 0 at `x = y = 1.5`, where only the `(1, 1)` corner is inside the grid, so the sample is `0.25 × 4 = 1.0`, and the total is `0.75 × 1.0 + 0.25 × 10 = 3.25`.

## Constraints
 - 1 ≤ `num_queries` ≤ 25,000
 - 1 ≤ `num_heads` ≤ 16
 - 1 ≤ `head_dim` ≤ 64
 - 1 ≤ `num_levels` ≤ 8
 - 1 ≤ `num_points` ≤ 8
 - 1 ≤ `H_l`, `W_l` ≤ 128 and `total_spatial` ≤ 25,000
 - `sampling_loc` values lie in `[-0.25, 1.25]`; `attn_weight` values lie in `[-5, 5]`
 - `value`, `sampling_loc`, `attn_weight` and `output` are `float32`; `spatial_shapes` is `int32`
 - Performance is measured with `num_queries` = `21,760`, `num_heads` = `8`, `head_dim` = `32`, `num_levels` = `4`, `num_points` = `4` and `spatial_shapes` = `[(128, 128), (64, 64), (32, 32), (16, 16)]`

## `solve.cu`
```
#include <cuda_runtime.h>

// value, spatial_shapes, sampling_loc, attn_weight, output are device pointers
extern "C" void solve(const float* value, const int* spatial_shapes, const float* sampling_loc,
                      const float* attn_weight, float* output, int num_queries, int num_heads,
                      int head_dim, int num_levels, int num_points) {}

```
