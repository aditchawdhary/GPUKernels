#include <cuda_runtime.h>

// LSD radix sort of 32-bit keys: 4 passes of 8 bits. Each pass is a stable counting sort:
//   1. histogram : per-tile digit counts, stored digit-major  counts[d * T + t]
//   2. scan      : per-digit exclusive scan across tiles + exclusive scan of digit totals
//                  -> global start of (digit d, tile t) = digit_base[d] + counts[d * T + t]
//   3. scatter   : each tile ranks its keys stably by digit (warp match + per-warp counters),
//                  reorders them in shared memory by digit, then writes each digit run
//                  contiguously to its global position (mostly coalesced writes).
// Ping-pong: input -> tmp -> output -> tmp -> output.

#define RADIX_BITS 8
#define RADIX 256
#define THREADS 256
#define WARPS (THREADS / 32)
#define ITEMS 16                    // keys per thread
#define TILE (THREADS * ITEMS)      // 4096 keys per tile

// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS)
histogram_kernel(const unsigned* __restrict__ in, unsigned* __restrict__ counts,
                 int N, int T, int shift) {
    __shared__ unsigned hist[RADIX];
    hist[threadIdx.x] = 0;
    __syncthreads();

    const int base = blockIdx.x * TILE;
#pragma unroll
    for (int r = 0; r < ITEMS; r++) {
        int i = base + r * THREADS + threadIdx.x;
        if (i < N) atomicAdd(&hist[(__ldg(&in[i]) >> shift) & (RADIX - 1)], 1u);
    }
    __syncthreads();
    counts[threadIdx.x * T + blockIdx.x] = hist[threadIdx.x];
}

// ---------------------------------------------------------------------------------------------
// Block-wide exclusive scan of one value per thread (blockDim.x = 1024 or 256). Returns the
// exclusive prefix; *total gets the block sum.
__device__ unsigned block_exclusive_scan(unsigned v, unsigned* warp_sums, unsigned* total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    unsigned x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        unsigned y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        unsigned s = lane < nwarps ? warp_sums[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            unsigned y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        if (lane < nwarps) warp_sums[lane] = s;   // inclusive warp prefix
    }
    __syncthreads();
    unsigned warp_prefix = warp ? warp_sums[warp - 1] : 0;
    *total = warp_sums[nwarps - 1];
    __syncthreads();                               // warp_sums reusable after return
    return warp_prefix + x - v;
}

// One block per digit: exclusive scan of counts[d*T .. d*T+T) in place; totals[d] = sum.
__global__ void scan_tiles_kernel(unsigned* __restrict__ counts, unsigned* __restrict__ totals, int T) {
    __shared__ unsigned warp_sums[32];
    unsigned* row = counts + (size_t)blockIdx.x * T;
    unsigned carry = 0;
    for (int base = 0; base < T; base += blockDim.x) {
        int i = base + threadIdx.x;
        unsigned v = i < T ? row[i] : 0;
        unsigned chunk_total;
        unsigned ex = block_exclusive_scan(v, warp_sums, &chunk_total);
        if (i < T) row[i] = carry + ex;
        carry += chunk_total;
    }
    if (threadIdx.x == 0) totals[blockIdx.x] = carry;
}

// Exclusive scan of the 256 digit totals -> digit_base.
__global__ void scan_digits_kernel(const unsigned* __restrict__ totals, unsigned* __restrict__ digit_base) {
    __shared__ unsigned warp_sums[32];
    unsigned tot;
    digit_base[threadIdx.x] = block_exclusive_scan(totals[threadIdx.x], warp_sums, &tot);
}

// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS)
scatter_kernel(const unsigned* __restrict__ in, unsigned* __restrict__ out,
               const unsigned* __restrict__ counts, const unsigned* __restrict__ digit_base,
               int N, int T, int shift) {
    __shared__ unsigned global_start[RADIX];        // global position of this tile's digit run
    __shared__ unsigned run_count[RADIX];           // running per-digit count within tile
    __shared__ unsigned local_start[RADIX];         // tile-local exclusive scan of digit counts
    __shared__ unsigned warp_cnt[WARPS][RADIX];
    __shared__ unsigned warp_sums[32];
    __shared__ unsigned sorted[TILE];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int base = blockIdx.x * TILE;
    const unsigned lt_mask = (1u << lane) - 1u;

    global_start[tid] = digit_base[tid] + counts[tid * T + blockIdx.x];
    run_count[tid] = 0;

    unsigned keys[ITEMS], ranks[ITEMS];
#pragma unroll
    for (int r = 0; r < ITEMS; r++) {
        int i = base + r * THREADS + tid;
        keys[r] = i < N ? __ldg(&in[i]) : 0u;
    }

    // Stable ranking. Key order inside the tile is (round r, warp, lane) == global index order.
#pragma unroll
    for (int r = 0; r < ITEMS; r++) {
        int i = base + r * THREADS + tid;
        bool valid = i < N;
        unsigned d = valid ? (keys[r] >> shift) & (RADIX - 1) : RADIX;   // invalid -> own group

#pragma unroll
        for (int w = 0; w < WARPS; w++) warp_cnt[w][tid] = 0;
        __syncthreads();

        unsigned peers = __match_any_sync(0xffffffffu, d);
        unsigned in_warp_rank = __popc(peers & lt_mask);
        if (valid && in_warp_rank == 0) warp_cnt[warp][d] = __popc(peers);
        __syncthreads();

        // thread = digit: prefix over warps, carried across rounds
        {
            unsigned running = run_count[tid];
#pragma unroll
            for (int w = 0; w < WARPS; w++) {
                unsigned c = warp_cnt[w][tid];
                warp_cnt[w][tid] = running;
                running += c;
            }
            run_count[tid] = running;
        }
        __syncthreads();

        ranks[r] = valid ? warp_cnt[warp][d] + in_warp_rank : 0u;
        __syncthreads();   // warp_cnt is reset at the top of the next round
    }

    // tile-local digit starts
    {
        unsigned tot;
        local_start[tid] = block_exclusive_scan(run_count[tid], warp_sums, &tot);
    }
    __syncthreads();

    // reorder in shared memory by digit (stable)
#pragma unroll
    for (int r = 0; r < ITEMS; r++) {
        int i = base + r * THREADS + tid;
        if (i < N) {
            unsigned d = (keys[r] >> shift) & (RADIX - 1);
            sorted[local_start[d] + ranks[r]] = keys[r];
        }
    }
    __syncthreads();

    // write out: consecutive threads write consecutive keys of the same digit run
    const int tile_n = min(TILE, N - base);
    for (int j = tid; j < tile_n; j += THREADS) {
        unsigned k = sorted[j];
        unsigned d = (k >> shift) & (RADIX - 1);
        out[global_start[d] + (j - local_start[d])] = k;
    }
}

// ---------------------------------------------------------------------------------------------
static void*  g_buf = nullptr;
static size_t g_buf_bytes = 0;

static void* scratch(size_t bytes) {
    if (bytes > g_buf_bytes) {
        if (g_buf) cudaFree(g_buf);
        cudaMalloc(&g_buf, bytes);
        g_buf_bytes = bytes;
    }
    return g_buf;
}

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {
    if (N <= 0) return;
    const int T = (N + TILE - 1) / TILE;

    size_t tmp_bytes    = ((size_t)N * sizeof(unsigned) + 255) & ~(size_t)255;
    size_t counts_bytes = ((size_t)RADIX * T * sizeof(unsigned) + 255) & ~(size_t)255;
    char* base = (char*)scratch(tmp_bytes + counts_bytes + 2 * RADIX * sizeof(unsigned));
    unsigned* tmp        = (unsigned*)base;
    unsigned* counts     = (unsigned*)(base + tmp_bytes);
    unsigned* totals     = (unsigned*)(base + tmp_bytes + counts_bytes);
    unsigned* digit_base = totals + RADIX;

    const unsigned* src = input;
    unsigned* dst = tmp;
    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * RADIX_BITS;
        histogram_kernel<<<T, THREADS>>>(src, counts, N, T, shift);
        scan_tiles_kernel<<<RADIX, 1024>>>(counts, totals, T);
        scan_digits_kernel<<<1, RADIX>>>(totals, digit_base);
        scatter_kernel<<<T, THREADS>>>(src, dst, counts, digit_base, N, T, shift);
        src = dst;
        dst = (dst == tmp) ? output : tmp;
    }
    // 4 passes: input -> tmp -> output -> tmp -> output
}
