/*
Linear Recurrence
Given a, x of shape [B, L] (float32, row-major), compute h of shape [B, L]:
  h[b, 0] = x[b, 0],   h[b, t] = a[b, t] * h[b, t-1] + x[b, t]   (t ≥ 1)

Constraints
 - 1 ≤ B ≤ 256,  1 ≤ L ≤ 65,536
 - Performance is measured with B = 64, L = 16,384
*/

/*
Approach: parallel scan over affine maps, split into segments so the whole GPU works.

Each step is the affine map f_t(h) = a_t * h + x_t, written (A, X) = (a_t, x_t).
Composition (apply l first, then r) is associative:
    r ∘ l = (l.A * r.A,  r.A * l.X + r.X)
and h[t] = (f_t ∘ ... ∘ f_0)(0). a[b, 0] is forced to 0, so h[b, 0] = x[b, 0]
holds even if a[b, 0] is inf or NaN.

Each row is split into segments of 4096 = 512 threads x 8 items. The grid is
(segments, B): 4 x 64 = 256 blocks at the benchmark size.

 1. seg_aggregate  Each block loads its segment through shared memory (coalesced),
                   folds 8 contiguous items per thread, then runs a block scan
                   (warp shuffles, then a scan over the 16 warp totals) to get the
                   segment's composed map, written to agg[b][s].
 2. seg_scan       Thread 0 folds agg[b][0..s-1] (at most 16 segments) to get h
                   entering the segment. The block scan gives each thread the
                   composed map of everything before it in the segment, so the
                   thread's starting h is excl.A * h_in + excl.X. It then runs the
                   plain recurrence over its 8 items, and the results are stored
                   coalesced through shared memory.

Traffic: 2 reads of a and x plus 1 write of h (about 20 MB at the benchmark),
two launches, no allocation (agg is a static device array: 256 x 16 float2).
Padding past L uses the identity map (1, 0).
Note: the scan re-associates float ops, so results can differ from a strict
serial loop in the last bits.
*/

#include <cuda_runtime.h>

#define THREADS 512
#define ITEMS   8
#define SEG     (THREADS * ITEMS)        // 4096 elements per block
#define NWARPS  (THREADS / 32)           // 16
#define PAD(i)  ((i) + ((i) >> 5))       // avoid bank conflicts on strided reads
#define MAX_B   256
#define MAX_SEG ((65536 + SEG - 1) / SEG) // 16

__device__ float2 g_agg[MAX_B * MAX_SEG];

// r ∘ l : apply l first, then r.   float2 = (A, X)
__device__ __forceinline__ float2 compose(float2 l, float2 r)
{
    return make_float2(l.x * r.x, fmaf(r.x, l.y, r.y));
}

__device__ __forceinline__ float2 shfl_up2(float2 v, int o)
{
    return make_float2(__shfl_up_sync(0xffffffff, v.x, o),
                       __shfl_up_sync(0xffffffff, v.y, o));
}

// Block-wide exclusive scan of per-thread maps (in thread order).
// Returns the exclusive prefix and writes the block total to *total.
__device__ __forceinline__ float2 block_scan(float2 v, float2* total)
{
    __shared__ float2 warpAgg[NWARPS];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const float2 I = make_float2(1.f, 0.f);

    float2 inc = v;                                  // inclusive warp scan
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        float2 y = shfl_up2(inc, o);
        if (lane >= o) inc = compose(y, inc);
    }
    float2 exc = shfl_up2(inc, 1);
    if (lane == 0) exc = I;
    if (lane == 31) warpAgg[warp] = inc;
    __syncthreads();

    if (warp == 0) {                                 // scan the 16 warp totals
        float2 w = (lane < NWARPS) ? warpAgg[lane] : I;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            float2 y = shfl_up2(w, o);
            if (lane >= o) w = compose(y, w);
        }
        if (lane < NWARPS) warpAgg[lane] = w;        // inclusive warp prefixes
    }
    __syncthreads();

    const float2 wp = warp ? warpAgg[warp - 1] : I;
    *total = warpAgg[NWARPS - 1];
    return compose(wp, exc);
}

// Coalesced load of one segment into smem (identity padding past L, a[0] := 0).
__device__ __forceinline__ void load_segment(const float* __restrict__ a,
                                             const float* __restrict__ x,
                                             float* sa, float* sx,
                                             size_t rowBase, int segStart, int L)
{
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int i = k * THREADS + threadIdx.x;
        const int t = segStart + i;
        float av = 1.f, xv = 0.f;
        if (t < L) {
            av = (t == 0) ? 0.f : __ldg(a + rowBase + t);
            xv = __ldg(x + rowBase + t);
        }
        sa[PAD(i)] = av;
        sx[PAD(i)] = xv;
    }
    __syncthreads();
}

// ---------------------------------------------------------------- pass 1
__global__ void __launch_bounds__(THREADS)
seg_aggregate(const float* __restrict__ a, const float* __restrict__ x, int L, int nSeg)
{
    __shared__ float sa[SEG + SEG / 32];
    __shared__ float sx[SEG + SEG / 32];

    const int s = blockIdx.x, b = blockIdx.y;
    const size_t rowBase = (size_t)b * L;
    load_segment(a, x, sa, sx, rowBase, s * SEG, L);

    float2 m = make_float2(1.f, 0.f);
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int i = threadIdx.x * ITEMS + k;
        m = compose(m, make_float2(sa[PAD(i)], sx[PAD(i)]));
    }

    float2 total;
    block_scan(m, &total);
    if (threadIdx.x == 0) g_agg[b * nSeg + s] = total;
}

// ---------------------------------------------------------------- pass 2
__global__ void __launch_bounds__(THREADS)
seg_scan(const float* __restrict__ a, const float* __restrict__ x, float* __restrict__ h,
         int L, int nSeg)
{
    __shared__ float sa[SEG + SEG / 32];
    __shared__ float sx[SEG + SEG / 32];
    __shared__ float hin;

    const int s = blockIdx.x, b = blockIdx.y;
    const size_t rowBase = (size_t)b * L;
    const int segStart = s * SEG;

    if (threadIdx.x == 0) {                          // h entering this segment
        float hv = 0.f;
        for (int j = 0; j < s; ++j) {
            const float2 g = g_agg[b * nSeg + j];
            hv = fmaf(g.x, hv, g.y);
        }
        hin = hv;
    }
    load_segment(a, x, sa, sx, rowBase, segStart, L);   // contains __syncthreads

    float2 m = make_float2(1.f, 0.f);
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int i = threadIdx.x * ITEMS + k;
        m = compose(m, make_float2(sa[PAD(i)], sx[PAD(i)]));
    }

    float2 total;
    const float2 excl = block_scan(m, &total);       // contains __syncthreads

    float hv = fmaf(excl.x, hin, excl.y);
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {                // overwrite sx with h (own slots only)
        const int i = threadIdx.x * ITEMS + k;
        hv = fmaf(sa[PAD(i)], hv, sx[PAD(i)]);
        sx[PAD(i)] = hv;
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {                // coalesced store
        const int i = k * THREADS + threadIdx.x;
        const int t = segStart + i;
        if (t < L) h[rowBase + t] = sx[PAD(i)];
    }
}

// ---------------------------------------------------------------- host
// a, x, h are device pointers
extern "C" void solve(const float* a, const float* x, float* h, int B, int L)
{
    if (B <= 0 || L <= 0) return;
    const int nSeg = (L + SEG - 1) / SEG;
    dim3 grid(nSeg, B);

    if (nSeg > 1) seg_aggregate<<<grid, THREADS>>>(a, x, L, nSeg);
    seg_scan<<<grid, THREADS>>>(a, x, h, L, nSeg);
    cudaDeviceSynchronize();
}
