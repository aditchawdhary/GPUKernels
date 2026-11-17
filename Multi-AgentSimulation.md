## Multi-Agent Simulation
Implement a program for a multi-agent flocking simulation (boids). The input consists of:

- An array agents containing N agents, where N is the total number of agents
- Each agent occupies `4 consecutive 32-bit floating point numbers` in the array: `[x, y, v_x, v_y]`, where: `(x, y)`  represents the agent's position in 2D space `(v_x, v_y)` represents the agent's velocity vector
- The total array size is `4 * N` floats, with agent 's data stored at indices `[4i, 4i+1, 4i+2, 4i+3]`

## Simulation Rules
<img width="697" height="437" alt="image" src="https://github.com/user-attachments/assets/95d1c2c7-d3fe-45d8-8c72-07fe99a9db28" />


## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the agents_next array

## Example 1:
```
Input: N = 2
agents = [
  0.0, 0.0, 1.0, 0.0,    // Agent 0: [x, y, vx, vy]
  3.0, 4.0, 0.0, -1.0    // Agent 1: [x, y, vx, vy]
]

Output:
agents_next = [
  1.0, 0.0, 1.0, 0.0,    // Agent 0: [x, y, vx, vy]
  3.0, 3.0, 0.0, -1.0    // Agent 1: [x, y, vx, vy]
]
```

## Constraints
 - 1 ≤ `N` ≤ 100,000
 - Each agent's position and velocity components are `32-bit` floats
 - Performance is measured with `N = 10,000`

## `solve.cu`
```
#include <cuda_runtime.h>

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {}

```
