/*
Rainbow Table

Implement a program that performs R rounds of parallel hashing on an array of 32-bit integers using the provided hash function. 
The hash should be applied R times iteratively (the output of one round becomes the input to the next).

Implementation Requirements
- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in array output

Example 1:
Input:  numbers = [123, 456, 789], R = 2
Output: hashes = [1636807824, 1273011621, 2193987222]

Example 2:
Input:  numbers = [0, 1, 2147483647], R = 3
Output: hashes = [96754810, 3571711400, 2006156166]

Constraints
- 1 ≤ N ≤ 10,000,000
- 1 ≤ R ≤ 100
- 0 ≤ input[i] ≤ 2147483647
- Performance is measured with N = 5,000,000
*/

#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        unsigned int hashed_val = (unsigned int)input[idx];
        for(int i=0; i<R; i++)
        {
            hashed_val = fnv1a_hash(hashed_val);
        }

        output[idx] = hashed_val;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
