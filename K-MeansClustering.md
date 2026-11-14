## K-Means Clustering

Implement the k-means clustering algorithm for 2D points. Given arrays of x and y coordinates for data points, initial centroids, and other parameters, assign each point to the nearest centroid and update the centroids iteratively. The final centroids and labels should be stored in the output arrays.

## Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - The final result must be stored in labels, `final_centroid_x`, and `final_centroid_y`

## Example 1:
```
Input:
sample_size = 4, k = 2, max_iterations = 10
data_x = [1.0, 2.0, 8.0, 9.0]
data_y = [1.0, 2.0, 8.0, 9.0]
initial_centroid_x = [1.0, 8.0]
initial_centroid_y = [1.0, 8.0]
Output: (see reference implementation for expected output)
```

## Constraints
 - 1 ≤ `sample_size` ≤ 1000000
 - 1 ≤ `k` ≤ 1000
 - All arrays are `float32` except labels, which is `int32`
 - Performance is measured with `k = 5`, `max_iterations = 30`, `sample_size = 10,000`

## `solve.cu`
```
#include <cuda_runtime.h>

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {}
```
