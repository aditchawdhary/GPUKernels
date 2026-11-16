/*
Stream Compaction
Given a 1D array A of N 32-bit floats, compact all positive elements (A[i] > 0)
to the front of out, preserving their relative order, and fill the rest with 0.0.

Constraints
 - 1 ≤ N ≤ 100,000,000
 - -1000.0 ≤ A[i] ≤ 1000.0
 - out is pre-allocated with N elements, initialised to 0.0
 - Performance is measured with N = 50,000,000
*/

/*
Approach: three-pass "count → scan → scatter" over 4096-element tiles.

 1. count_tiles   Each block counts the positives in its tile
                  (256 threads × 4 float4 loads, warp-shuffle reduce) → counts[b].
 2. scan_counts   One 1024-thread block does an exclusive scan of counts[]
                  (≈12.2K tiles at N = 50M, looped with a carry so any N works).
                  offsets[nTiles] holds the total number of positives k.
 3. scatter       Each block re-reads its tile in the same striped order. For
                  each float4 "pass", every thread counts its positives, a warp
                  shuffle scan + a scan over warp totals gives each thread its
                  write position, and positives are written to
                  out[offsets[b] + running + local]. Order is preserved because
                  ranks follow element order: pass, then thread, then lane in the float4.
                  Threads also write 0.0 to their own indices i ≥ k, so the result
                  does not depend on out having been zeroed beforehand.

Memory traffic: about 2N reads + N writes. All loads are coalesced float4, and
writes are contiguous within each warp. The scratch buffer (≈24.4K ints at
N = 1e8) is allocated once and reused across calls.

Edge cases: when N is not a multiple of 4, the last float4 falls back to guarded
scalar loads. A[i] == 0.0 (including -0.0) and NaN fail `> 0` and are dropped.
*/

#include <cuda_runtime.h>

#define THREADS 256
#define VEC     4                          // floats per float4
#define PASSES  4                          // float4 loads per thread per tile
#define TILE    (THREADS * VEC * PASSES)   // 4096 elements per block
#define NWARPS  (THREADS / 32)
#define SCAN_THREADS 1024

// Load 4 consecutive floats starting at idx (idx is a multiple of 4).
__device__ __forceinline__ float4 load4(const float* __restrict__ A, int idx, int N)
{
    if (idx + 3 < N) {
        return __ldg(reinterpret_cast<const float4*>(A + idx));
    }
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);   // padding is "not positive"
    if (idx     < N) v.x = A[idx];
    if (idx + 1 < N) v.y = A[idx + 1];
    if (idx + 2 < N) v.z = A[idx + 2];
    return v;
}

__device__ __forceinline__ int count4(float4 v)
{
    return (v.x > 0.f) + (v.y > 0.f) + (v.z > 0.f) + (v.w > 0.f);
}

// ---------------------------------------------------------------- pass 1
__global__ void count_tiles(const float* __restrict__ A, int N, int* __restrict__ counts)
{
    __shared__ int warpSums[NWARPS];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const long long base = (long long)blockIdx.x * TILE;

    int c = 0;
    #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
        long long idx = base + (long long)p * THREADS * VEC + tid * VEC;
        if (idx < N) c += count4(load4(A, (int)idx, N));
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffff, c, o);
    if (lane == 0) warpSums[warp] = c;
    __syncthreads();

    if (warp == 0) {
        c = (lane < NWARPS) ? warpSums[lane] : 0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffff, c, o);
        if (lane == 0) counts[blockIdx.x] = c;
    }
}

// ---------------------------------------------------------------- pass 2
// Exclusive scan of counts[0..n) into offsets[0..n]; offsets[n] = total.
__global__ void scan_counts(const int* __restrict__ counts, int* __restrict__ offsets, int n)
{
    __shared__ int warpSums[SCAN_THREADS / 32];
    __shared__ int carry;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (tid == 0) carry = 0;
    __syncthreads();

    for (int start = 0; start < n; start += SCAN_THREADS) {
        int i = start + tid;
        int v = (i < n) ? counts[i] : 0;

        // inclusive warp scan
        int x = v;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffff, x, o);
            if (lane >= o) x += y;
        }
        if (lane == 31) warpSums[warp] = x;
        __syncthreads();

        if (warp == 0) {
            int w = warpSums[lane];             // SCAN_THREADS/32 == 32 warps
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                int y = __shfl_up_sync(0xffffffff, w, o);
                if (lane >= o) w += y;
            }
            warpSums[lane] = w;                 // inclusive warp totals
        }
        __syncthreads();

        int c = carry;
        int excl = c + (warp ? warpSums[warp - 1] : 0) + x - v;
        if (i < n) offsets[i] = excl;
        __syncthreads();                        // everyone has read carry
        if (tid == SCAN_THREADS - 1) carry = excl + v;
        __syncthreads();
    }
    if (tid == 0) offsets[n] = carry;
}

// ---------------------------------------------------------------- pass 3
__global__ void scatter(const float* __restrict__ A, int N,
                        const int* __restrict__ offsets, int nTiles,
                        float* __restrict__ out)
{
    __shared__ int warpSums[NWARPS];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const long long base = (long long)blockIdx.x * TILE;
    const int total = offsets[nTiles];
    int running = offsets[blockIdx.x];

    #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
        long long idx = base + (long long)p * THREADS * VEC + tid * VEC;
        float4 v = (idx < N) ? load4(A, (int)idx, N) : make_float4(0.f, 0.f, 0.f, 0.f);
        int c = count4(v);

        // inclusive warp scan of per-thread counts
        int x = c;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffff, x, o);
            if (lane >= o) x += y;
        }
        if (lane == 31) warpSums[warp] = x;
        __syncthreads();

        int warpPrefix = 0, passTotal = 0;
        #pragma unroll
        for (int w = 0; w < NWARPS; ++w) {      // NWARPS = 8: cheap broadcast reads
            int s = warpSums[w];
            if (w < warp) warpPrefix += s;
            passTotal += s;
        }

        int pos = running + warpPrefix + x - c;
        if (v.x > 0.f) out[pos++] = v.x;
        if (v.y > 0.f) out[pos++] = v.y;
        if (v.z > 0.f) out[pos++] = v.z;
        if (v.w > 0.f) out[pos]   = v.w;

        // zero-fill this thread's own slots that lie past the compacted prefix
        // (disjoint from every write above, since those are all < total)
        int i0 = (int)idx;
        if (idx < N && i0 + 3 >= total) {
            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                int j = i0 + k;
                if (j >= total && j < N) out[j] = 0.f;
            }
        }

        running += passTotal;
        __syncthreads();                        // warpSums reused next pass
    }
}

// ---------------------------------------------------------------- host
static int* g_scratch = nullptr;
static int  g_scratchCap = 0;

// A, out are device pointers
extern "C" void solve(const float* A, int N, float* out)
{
    if (N <= 0) return;
    const int nTiles = (N + TILE - 1) / TILE;
    const int need   = 2 * nTiles + 1;          // counts[nTiles] + offsets[nTiles+1]

    if (need > g_scratchCap) {
        if (g_scratch) cudaFree(g_scratch);
        cudaMalloc(&g_scratch, need * sizeof(int));
        g_scratchCap = need;
    }
    int* counts  = g_scratch;
    int* offsets = g_scratch + nTiles;

    count_tiles<<<nTiles, THREADS>>>(A, N, counts);
    scan_counts<<<1, SCAN_THREADS>>>(counts, offsets, nTiles);
    scatter<<<nTiles, THREADS>>>(A, N, offsets, nTiles, out);
    cudaDeviceSynchronize();
}
