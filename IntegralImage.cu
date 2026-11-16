/*
Integral Image
Given an H x W image of 32-bit floats (row-major), compute its summed area table:
output[i][j] = sum of input[0..i][0..j] (inclusive).

Constraints
 - 1 ≤ H, W ≤ 16,384
 - Input values are in [-100, 100]
 - Performance is measured with H = W = 8,192
*/

/*
Approach: separable two-pass scan (rows, then columns), both fully coalesced.

 1. row_scan     One block per row (1024 threads). The row is processed in chunks of
                 4096 = 1024 threads x 4 items: a coalesced load into padded shared
                 memory, a 4-element serial scan per thread, a warp-shuffle scan, a scan
                 over the 32 warp totals, and a coalesced store to output. A carry
                 links the chunks, so any W up to 16,384 works. input is only read.
 2. col_scan     In place on output. Each block covers 32 adjacent columns with a
                 32x32 thread grid. threadIdx.x picks the column, so every warp reads
                 128 contiguous bytes of a row. threadIdx.y owns a contiguous band of
                 ceil(H/32) rows. (a) Each thread sums its band. (b) A shared-memory
                 scan over the 32 bands of each column gives every band its starting
                 value. (c) Each thread walks its band again, writing the running sum.

Traffic: pass 1 reads N and writes N, pass 2 reads 2N and writes N (about 1.3 GB
at 8192^2). Both passes are bandwidth-bound with coalesced accesses. At W = 8192,
pass 2 launches 256 blocks x 1024 threads, enough to keep every SM busy.
*/

#include <cuda_runtime.h>

#define ROW_THREADS 1024
#define ROW_ITEMS   4
#define ROW_CHUNK   (ROW_THREADS * ROW_ITEMS)   // 4096
#define PAD(i)      ((i) + ((i) >> 5))          // avoid shared-memory bank conflicts

#define COL_X 32                                // columns per block (one warp wide)
#define COL_Y 32                                // row bands per block

// ---------------------------------------------------------------- pass 1
__global__ void row_scan(const float* __restrict__ input, float* __restrict__ output,
                         int W)
{
    __shared__ float s[ROW_CHUNK + ROW_CHUNK / 32];
    __shared__ float warpSums[ROW_THREADS / 32];
    __shared__ float carry;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const size_t rowBase = (size_t)blockIdx.x * W;
    const float* in  = input  + rowBase;
    float*       out = output + rowBase;

    if (tid == 0) carry = 0.f;

    for (int start = 0; start < W; start += ROW_CHUNK) {
        // coalesced load
        #pragma unroll
        for (int k = 0; k < ROW_ITEMS; ++k) {
            int i = k * ROW_THREADS + tid;
            int g = start + i;
            s[PAD(i)] = (g < W) ? __ldg(in + g) : 0.f;
        }
        __syncthreads();

        // serial inclusive scan of this thread's 4 consecutive elements
        float v[ROW_ITEMS];
        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < ROW_ITEMS; ++k) {
            acc += s[PAD(tid * ROW_ITEMS + k)];
            v[k] = acc;
        }

        // inclusive warp scan of per-thread totals
        float x = acc;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            float y = __shfl_up_sync(0xffffffff, x, o);
            if (lane >= o) x += y;
        }
        if (lane == 31) warpSums[warp] = x;
        __syncthreads();

        if (warp == 0) {
            float w = warpSums[lane];           // 32 warps
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                float y = __shfl_up_sync(0xffffffff, w, o);
                if (lane >= o) w += y;
            }
            warpSums[lane] = w;                 // inclusive warp totals
        }
        __syncthreads();

        float prefix = carry + (warp ? warpSums[warp - 1] : 0.f) + (x - acc);
        #pragma unroll
        for (int k = 0; k < ROW_ITEMS; ++k)
            s[PAD(tid * ROW_ITEMS + k)] = v[k] + prefix;
        __syncthreads();

        // coalesced store
        #pragma unroll
        for (int k = 0; k < ROW_ITEMS; ++k) {
            int i = k * ROW_THREADS + tid;
            int g = start + i;
            if (g < W) out[g] = s[PAD(i)];
        }
        if (tid == ROW_THREADS - 1) carry = s[PAD(ROW_CHUNK - 1)];
        __syncthreads();                        // carry and s are ready for the next chunk
    }
}

// ---------------------------------------------------------------- pass 2
__global__ void col_scan(float* __restrict__ output, int H, int W)
{
    __shared__ float bandSum[COL_Y][COL_X + 1];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int col = blockIdx.x * COL_X + tx;
    const int band = (H + COL_Y - 1) / COL_Y;
    const int r0 = ty * band;
    const int r1 = min(r0 + band, H);
    const bool active = col < W;

    // (a) sum of this thread's band
    float sum = 0.f;
    if (active) {
        #pragma unroll 8
        for (int r = r0; r < r1; ++r) sum += output[(size_t)r * W + col];
    }
    bandSum[ty][tx] = sum;
    __syncthreads();

    // (b) exclusive scan over the 32 bands of each column (warp ty scans column ty)
    {
        const int c = ty, b = tx;               // reuse the 32x32 grid transposed
        float v = bandSum[b][c];
        float x = v;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            float y = __shfl_up_sync(0xffffffff, x, o);
            if (b >= o) x += y;
        }
        __syncthreads();
        bandSum[b][c] = x - v;                  // exclusive prefix
    }
    __syncthreads();

    // (c) rewrite the band with the running column sum
    if (active) {
        float run = bandSum[ty][tx];
        #pragma unroll 8
        for (int r = r0; r < r1; ++r) {
            size_t idx = (size_t)r * W + col;
            run += output[idx];
            output[idx] = run;
        }
    }
}

// ---------------------------------------------------------------- host
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int H, int W)
{
    if (H <= 0 || W <= 0) return;

    row_scan<<<H, ROW_THREADS>>>(input, output, W);

    dim3 block(COL_X, COL_Y);
    dim3 grid((W + COL_X - 1) / COL_X);
    col_scan<<<grid, block>>>(output, H, W);

    cudaDeviceSynchronize();
}
