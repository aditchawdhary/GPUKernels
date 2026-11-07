/*
Prefix Sum
Write a GPU program that computes the prefix sum (cumulative sum) of an array of 32-bit floating point numbers. For an input array [a, b, c, d, ...], the prefix sum is [a, a+b, a+b+c, a+b+c+d, ...].

Implementation Requirements
 - Use only GPU native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The result must be stored in the output array

Example 1:
  Input: [1.0, 2.0, 3.0, 4.0]
  Output: [1.0, 3.0, 6.0, 10.0]

Example 2:
  Input: [5.0, -2.0, 3.0, 1.0, -4.0]
  Output: [5.0, 3.0, 6.0, 7.0, 3.0]

Constraints
 - 1 ≤ N ≤ 100,000,000
 - -1000.0 ≤ input[i] ≤ 1000.0
 - The largest value in the output array will fit within a 32-bit float
 - Performance is measured with N = 250,000
*/

/*
Here’s a work-efficient, multi-level block scan.
 1. Each block scans a tile of 2048 elements (256 threads × 8 items). It loads the tile into shared memory with coalesced reads, does a per-thread serial scan, then a warp-shuffle scan, and writes the tile’s total to a blockSums array.
 2. The blockSums array is scanned recursively with the same kernel, so it handles any N up to 10⁸.
 3. Each tile then adds the scanned total of all tiles before it.

Performance and edge cases:
 - Benchmark size. For N = 250,000 there are 123 tiles, so the kernel sequence is scan → 1-block scan of the sums → add. That is three launches, and memory traffic is about 3N floats.
 - Bank conflicts. PAD(i) stops the strided per-thread reads (tid*8 + k) from conflicting on shared-memory banks.
 - Scratch buffer. The static buffer is allocated once and reused, so cudaMalloc isn’t paid on every call. The largest case, 10⁸ elements, needs about 48.8K floats of scratch.
 - In-place recursion. Scanning the sums array in place is safe. Each block fully loads its tile into shared memory before writing, and different blocks touch disjoint ranges.
 - Small inputs. Inputs with 2048 or fewer elements go through a single block with no extra passes.
*/


#include <cuda_runtime.h>

#define THREADS 256
#define ITEMS   8
#define TILE    (THREADS * ITEMS)        // 2048 elements per block
#define NWARPS  (THREADS / 32)
#define PAD(i)  ((i) + ((i) >> 5))       // avoid shared-memory bank conflicts

// Scan one tile per block. Optionally write the tile's total to blockSums.
__global__ void scan_tile(const float* __restrict__ in, float* out,
                          float* blockSums, int n)
{
    __shared__ float s[TILE + TILE / 32];
    __shared__ float warpSums[NWARPS];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int base = blockIdx.x * TILE;

    // Coalesced load into shared memory
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        int idx = k * THREADS + tid;
        int g = base + idx;
        s[PAD(idx)] = (g < n) ? in[g] : 0.0f;
    }
    __syncthreads();

    // Each thread scans its 8 contiguous elements serially
    float v[ITEMS];
    float sum = 0.0f;
    const int local = tid * ITEMS;
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        sum += s[PAD(local + k)];
        v[k] = sum;
    }

    // Inclusive warp scan of the thread totals
    float x = sum;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        float y = __shfl_up_sync(0xffffffff, x, off);
        if (lane >= off) x += y;
    }
    if (lane == 31) warpSums[warp] = x;
    __syncthreads();

    // Scan the warp totals using the first warp
    if (warp == 0) {
        float w = (lane < NWARPS) ? warpSums[lane] : 0.0f;
        #pragma unroll
        for (int off = 1; off < NWARPS; off <<= 1) {
            float y = __shfl_up_sync(0xffffffff, w, off);
            if (lane >= off) w += y;
        }
        if (lane < NWARPS) warpSums[lane] = w;
    }
    __syncthreads();

    // Exclusive prefix for this thread's chunk
    float prefix = (x - sum) + (warp > 0 ? warpSums[warp - 1] : 0.0f);

    #pragma unroll
    for (int k = 0; k < ITEMS; ++k)
        s[PAD(local + k)] = v[k] + prefix;
    __syncthreads();

    // Coalesced store
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        int idx = k * THREADS + tid;
        int g = base + idx;
        if (g < n) out[g] = s[PAD(idx)];
    }

    if (blockSums && tid == 0)
        blockSums[blockIdx.x] = warpSums[NWARPS - 1];
}

// Add the scanned total of all previous tiles to each tile (except the first)
__global__ void add_offsets(float* out, const float* scannedSums, int n)
{
    if (blockIdx.x == 0) return;
    const float add = scannedSums[blockIdx.x - 1];
    const int base = blockIdx.x * TILE;
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        int g = base + k * THREADS + threadIdx.x;
        if (g < n) out[g] += add;
    }
}

static void scan_rec(const float* in, float* out, int n, float* scratch)
{
    int numBlocks = (n + TILE - 1) / TILE;
    if (numBlocks == 1) {
        scan_tile<<<1, THREADS>>>(in, out, nullptr, n);
        return;
    }
    float* sums = scratch;                  // numBlocks floats for this level
    scan_tile<<<numBlocks, THREADS>>>(in, out, sums, n);
    scan_rec(sums, sums, numBlocks, scratch + numBlocks);   // in-place is safe
    add_offsets<<<numBlocks, THREADS>>>(out, sums, n);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N)
{
    if (N <= 0) return;

    // Scratch space needed for all recursion levels
    size_t need = 0;
    for (long long m = N; m > TILE; ) {
        m = (m + TILE - 1) / TILE;
        need += m;
    }

    static float* scratch = nullptr;
    static size_t capacity = 0;
    if (need > capacity) {
        if (scratch) cudaFree(scratch);
        cudaMalloc(&scratch, need * sizeof(float));
        capacity = need;
    }

    scan_rec(input, output, N, scratch);
    cudaDeviceSynchronize();
}
