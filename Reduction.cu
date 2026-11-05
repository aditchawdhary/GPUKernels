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


// The include brings in the CUDA runtime functions (cudaMemset, kernel launches and so on). 
// THREADS is the number of threads in each block
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// __global__ means this function runs on the GPU and is launched from the CPU.
// Thousands of copies of it run at once, one per thread.
__global__ void reduction(const float* input, float* output, int N)
{
    // array stored in shared memory, a small, fast memory that every thread in the same block can read and write.
    // each block gets its own copy with one slot per thread.
    __shared__ float sdata[THREADS_PER_BLOCK];

    // Each thread keeps its own running total in a register, which is private to that thread and very fast
    float sum = 0.0f;

    // This is a grid-stride loop, and its where the array actually gets read
    // - blockIdx.x * blockDim.x + threadIdx.x is the threads global index, 
    // - gridDim.x * blockDim.x is the total number of threads launched. With 1024 blocks and 256 threads, thats 256K threads
    // - each thread reads one element, then jumps ahead by the total thread count and reads again until it passes N. Thread 0
    //   reads 0, 256K, 512K, 718K, 1024K...
    // Together the threads cover every element exactly once. Neighboring threads read neighborigin addresses at each step,
    // which keeps memory access fast. 
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
    {
        sum += input[i];
    }

    // This is grid-stride 
    // Each thread sotres its toatl in its slot of shared memory. __syncthreads() is a barrier: no thread in the block
    // continues until all of them reach this line. without it, a thread could read a slot that another thread hasn't written yet.
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Tree reduction in shared memory, which turns the blocks 256 values into one.
    // - First pass, s=128: thread 0-127 each add the value 128 slots ahead, so sdata[0] += sdata[128] and so on. That leaves 128 partial sums.
    // - Next pass, s=64: threads 0-63 fold the 128 partial sums into 64
    // - the passes continue at 32, 16, 8, 4, 2 and 1. After 8 steps (2**8 = 256), sdata[0] holds the whole blocks sum
    // - s = s/2, halves s on each pass. The __syncthreads() inside the loop makes each pass finish before the next one starts.
    for (int s = blockDim.x / 2; s > 0; s = s/2)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // One atomic add per block
    // Thread 0 of each block adds its blocks sum to the final result. It must be an atomicAdd because up to 1024 blocks write to the same address.
    // With a plain +=, two blocks could read the old value at the same moment and one of their additions would be lost. atomicAdd performs
    // the read, add and write as one step that can't be interrupted.
    if (threadIdx.x == 0)
    {
        atomicAdd(output, sdata[0]);
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) 
{
    // The first line is a ceiling division: just enough blocks for one thread per element. 
    // The cap at 1024 limits the number of blocks, and the grid-stride loop lets each thread handle many elements when N is large. 
    // Fewer blocks also means fewer atomic adds at the end.
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;

    // This launches the kernel. <<<blocks, THREADS>>> sets the grid size: blocks blocks of 256 threads each.
    reduction<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input, output, N, THREADS_PER_BLOCK);
    
    //Kernel launches are asynchronous, so the CPU would otherwise carry on right away. This line waits for the GPU to finish, so output holds the final sum when solve returns.
    cudaDeviceSynchronize();
}
