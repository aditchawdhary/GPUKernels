# N-body Gravitational Force

Given N bodies in 3D space, each with a position vector `(x, y, z)` and a scalar mass, compute the gravitational force acting on each body due to all other bodies. The softened gravitational force on body `i` is:
<img width="815" height="71" alt="image" src="https://github.com/user-attachments/assets/726ac611-3b91-41de-b164-9031f572d566" />

 
 

where <img width="80" height="20" alt="image" src="https://github.com/user-attachments/assets/455f51c2-70b8-4082-a4a5-e6a0b01e01c8" /> is a softening factor that prevents singularities when two bodies occupy the same position. 
Positions and forces are stored in row-major order: `array[i * 3 + k]` is the k-th coordinate of body `i`.

<img width="817" height="249" alt="image" src="https://github.com/user-attachments/assets/3ae74ab0-d912-4eb5-8028-eaa6b9285ecb" />

## Implementation Requirements
 - Use only native GPU features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The computed force for each body must be written to forces

## Example
<img width="833" height="370" alt="image" src="https://github.com/user-attachments/assets/5906ffb4-40b8-4370-a5c3-f710bace3220" />

 

## Constraints
 - 1 ≤ `N` ≤ 65,536
 - Positions and forces are `float32` arrays of shape `N × 3` in row-major order
 - -10.0 ≤ `positions[i]` ≤ 10.0
 - 0.0 < `masses[i]` ≤ 100.0
 - Performance is measured with `N = 8,192`

 ## `solve.cu`
 ```
 #include <cuda_runtime.h>

// positions, masses, forces are device pointers
extern "C" void solve(const float* positions, const float* masses, float* forces, int N) {}

 ```
