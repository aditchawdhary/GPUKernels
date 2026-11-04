/*
Reduction

Write a GPU program that performs parallel reduction on an array of 32-bit floating point numbers to compute their sum. 
The program should take an input array and produce a single output value containing the sum of all elements.

Implementation Requirements
 - Use only GPU native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in the output variable

Example 1:
Input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Output: 36.0

Example 2:
Input: [-2.5, 1.5, -1.0, 2.0]
Output: 0.0

Constraints
 - 1 ≤ N ≤ 100,000,000
 - -1000.0 ≤ input[i] ≤ 1000.0
 - The final sum will always fit within a 32-bit float
 - Performance is measured with N = 4,194,304
*/


#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) 
{
  
}
