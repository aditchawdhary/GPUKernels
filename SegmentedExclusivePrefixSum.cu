#include <cuda_runtime.h>

// Segmented exclusive prefix sum.
// Each element is a pair (flag, value); pairs combine with the associative segmented operator
//     (f1, s1) (+) (f2, s2) = (f1 | f2,  f2 ? s2 : s1 + s2)
// and output[i] = flag[i] ? 0 : S_{i-1}, where S is the segmented inclusive scan.
//
// Three deterministic phases over tiles of TILE = 256 threads x 16 items:
//   1. tile_aggregate : each tile reduces to one pair
//   2. scan_carries   : one block scans the tile pairs -> carry-in (segment sum at tile start)
//   3. tile_scan      : each tile loads through shared memory (coalesced), scans each thread's
//                       16 contiguous items, block-scans the thread pairs, applies the carry
//                       and writes the exclusive results back through shared memory.
// All accumulation is in double (the reference is computed in float64): with one long segment
// the running sum crosses zero, where only the 1e-3 absolute tolerance applies, and fp32 running
// sums near a base of ~1e4 would carry ~1e-3 of rounding error. Outputs are rounded to fp32 once.

#define THREADS 256
#define ITEMS 16
#define TILE (THREADS * ITEMS)
#define CARRY_T 1024
#define PIDX(j) ((j) + ((j) >> 5))     // one pad word per 32: conflict-free strided access
#define SMEM (TILE + TILE / 32)

struct SPair { int f; double s; };

__device__ __forceinline__ SPair seg_combine(SPair a, SPair b) {
    SPair r;
    r.f = a.f | b.f;
    r.s = b.f ? b.s : a.s + b.s;
    return r;
}

// block-wide exclusive segmented scan of one pair per thread; *total = block aggregate
__device__ SPair block_excl_scan(SPair v, SPair* total) {
    __shared__ int wf[32];
    __shared__ double ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;

    SPair inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        SPair up;
        up.f = __shfl_up_sync(0xffffffffu, inc.f, o);
        up.s = __shfl_up_sync(0xffffffffu, inc.s, o);
        if (lane >= o) inc = seg_combine(up, inc);
    }
    SPair lane_ex;
    lane_ex.f = __shfl_up_sync(0xffffffffu, inc.f, 1);
    lane_ex.s = __shfl_up_sync(0xffffffffu, inc.s, 1);
    if (lane == 0) { lane_ex.f = 0; lane_ex.s = 0.0; }

    if (lane == 31) { wf[warp] = inc.f; ws[warp] = inc.s; }
    __syncthreads();
    if (warp == 0) {
        SPair w;
        w.f = lane < nw ? wf[lane] : 0;
        w.s = lane < nw ? ws[lane] : 0.0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            SPair up;
            up.f = __shfl_up_sync(0xffffffffu, w.f, o);
            up.s = __shfl_up_sync(0xffffffffu, w.s, o);
            if (lane >= o) w = seg_combine(up, w);
        }
        if (lane < nw) { wf[lane] = w.f; ws[lane] = w.s; }   // inclusive over warps
    }
    __syncthreads();
    SPair warp_ex;
    if (warp == 0) { warp_ex.f = 0; warp_ex.s = 0.0; }
    else { warp_ex.f = wf[warp - 1]; warp_ex.s = ws[warp - 1]; }
    total->f = wf[nw - 1];
    total->s = ws[nw - 1];
    __syncthreads();                  // shared scratch reusable after return
    return seg_combine(warp_ex, lane_ex);
}

// coalesced tile load into shared memory; elements past N become (flag 0, value 0)
__device__ __forceinline__ void load_tile(const float* __restrict__ values, const int* __restrict__ flags,
                                          float* sv, int* sf, size_t base, int N) {
#pragma unroll
    for (int r = 0; r < ITEMS; r++) {
        int j = r * THREADS + threadIdx.x;
        size_t i = base + j;
        bool ok = i < (size_t)N;
        sv[PIDX(j)] = ok ? __ldg(&values[i]) : 0.0f;
        sf[PIDX(j)] = ok ? (__ldg(&flags[i]) != 0) : 0;
    }
}

// the pair summarising this thread's ITEMS contiguous elements
__device__ __forceinline__ SPair thread_pair(const float* sv, const int* sf) {
    const int b = threadIdx.x * ITEMS;
    int f = 0;
    double s = 0.0;
#pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        const int j = PIDX(b + k);
        if (sf[j]) { f = 1; s = (double)sv[j]; }
        else s += (double)sv[j];
    }
    SPair p; p.f = f; p.s = s;
    return p;
}

__global__ void __launch_bounds__(THREADS)
tile_aggregate(const float* __restrict__ values, const int* __restrict__ flags,
               int* __restrict__ agg_f, double* __restrict__ agg_s, int N) {
    __shared__ float sv[SMEM];
    __shared__ int sf[SMEM];
    load_tile(values, flags, sv, sf, (size_t)blockIdx.x * TILE, N);
    __syncthreads();
    SPair tot;
    block_excl_scan(thread_pair(sv, sf), &tot);
    if (threadIdx.x == 0) { agg_f[blockIdx.x] = tot.f; agg_s[blockIdx.x] = tot.s; }
}

// carry[t] = segmented sum running into tile t (exclusive over tiles)
__global__ void __launch_bounds__(CARRY_T)
scan_carries(const int* __restrict__ agg_f, const double* __restrict__ agg_s,
             double* __restrict__ carry, int T) {
    SPair run; run.f = 0; run.s = 0.0;
    for (int base = 0; base < T; base += CARRY_T) {
        int t = base + threadIdx.x;
        SPair v;
        v.f = t < T ? agg_f[t] : 0;
        v.s = t < T ? agg_s[t] : 0.0;
        SPair tot;
        SPair ex = block_excl_scan(v, &tot);
        if (t < T) carry[t] = seg_combine(run, ex).s;
        run = seg_combine(run, tot);
    }
}

__global__ void __launch_bounds__(THREADS)
tile_scan(const float* __restrict__ values, const int* __restrict__ flags,
          const double* __restrict__ carry, float* __restrict__ output, int N) {
    __shared__ float sv[SMEM];
    __shared__ int sf[SMEM];
    const size_t base = (size_t)blockIdx.x * TILE;
    load_tile(values, flags, sv, sf, base, N);
    __syncthreads();

    SPair tot;
    SPair ex = block_excl_scan(thread_pair(sv, sf), &tot);
    SPair cin; cin.f = 0; cin.s = carry[blockIdx.x];
    double run = seg_combine(cin, ex).s;              // running sum entering this thread

    // sequential exclusive scan of this thread's items, in place in shared memory
    const int b = threadIdx.x * ITEMS;
#pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        const int j = PIDX(b + k);
        double v = (double)sv[j];
        if (sf[j]) { sv[j] = 0.0f; run = v; }
        else       { sv[j] = (float)run; run += v; }
    }
    __syncthreads();

#pragma unroll
    for (int r = 0; r < ITEMS; r++) {
        int j = r * THREADS + threadIdx.x;
        size_t i = base + j;
        if (i < (size_t)N) output[i] = sv[PIDX(j)];
    }
}

static char* g_buf = nullptr;
static size_t g_bytes = 0;

// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {
    if (N <= 0) return;
    const int T = (N + TILE - 1) / TILE;                 // <= 24415 for N <= 1e8
    size_t need = (size_t)T * (2 * sizeof(double) + sizeof(int)) + 64;
    if (need > g_bytes) {
        if (g_buf) cudaFree(g_buf);
        cudaMalloc(&g_buf, need);
        g_bytes = need;
    }
    double* agg_s = (double*)g_buf;
    double* carry = agg_s + T;
    int* agg_f = (int*)(carry + T);

    tile_aggregate<<<T, THREADS>>>(values, flags, agg_f, agg_s, N);
    scan_carries<<<1, CARRY_T>>>(agg_f, agg_s, carry, T);
    tile_scan<<<T, THREADS>>>(values, flags, carry, output, N);
}
