## Segmented Exclusive Prefix Sum

Given an array of `N` `32-bit` floating point values and an integer array flags of the same length, where `flags[i] = 1` marks the start of a new segment and `flags[i] = 0` continues the current segment, 
compute the exclusive prefix sum within each segment and store the result in output. The first element is always a segment start `(flags[0] = 1)`. 
Within each segment, `output[i]` equals the sum of all values elements in the same segment that appear before index `i`, so the first element of every segment is always `0.0`.

## Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in output
Read from values and flags; write to output

## Example
```
Input values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
Input flags:  [  1,   0,   0,   1,   0,   1]

Segments:     [1.0, 2.0, 3.0] | [4.0, 5.0] | [6.0]

Output:       [0.0, 1.0, 3.0,   0.0, 4.0,   0.0]
Segment 1: exclusive prefix sums of [1, 2, 3] → [0, 1, 3]
Segment 2: exclusive prefix sums of [4, 5] → [0, 4]
Segment 3: exclusive prefix sums of [6] → [0]
```

## Constraints
 - 1 ≤ `N` ≤ 100,000,000
 - `flags[0]` = 1 always (the first element starts the first segment)
 - `flags[i] ∈ {0, 1}` for all `i`
 - Values are `32-bit` floats in the range `[-100, 100]`
 - Performance is measured with `N = 50,000,000`

## `solve.cu`
```
#include <cuda_runtime.h>

// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {}

```
