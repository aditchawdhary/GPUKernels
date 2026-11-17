/*
Parallel Reverse Scan (GAE)
rewards, values, advantages: [B, S] float32.
  delta_t = r_t + gamma * v_{t+1} - v_t        (v_S = 0)
  A_t     = delta_t + (gamma * lam) * A_{t+1}  (A_S = 0)
(Spec: leetgpu-challenges #110, tolerance 1e-3. B ≤ 256, S ≤ 65,536;
performance at B = 64, S = 4096.)
*/

/*
Approach: a segmented parallel scan over affine maps, run right to left.

Read each row backwards, i = S-1-t. Then A_i = c * A_{i-1} + delta_i with c = gamma*lam,
a forward linear recurrence. Each step is the affine map (A, X) = (c, delta).
Composition (apply l first, then r): r ∘ l = (l.A * r.A, r.A * l.X + r.X).

Rows are cut into segments of 1024 = 256 threads x 4 items (grid = segments x B;
at the benchmark 4 x 64 = 256 blocks).
 1. seg_aggregate  Each block loads its segment through shared memory (coalesced;
                   the reversed order just walks addresses downward), builds delta
                   from r, v and the next v on the fly, folds 4 items per thread,
                   and block-scans to get the segment's composed map -> agg[b][s].
                   Skipped when a row fits in one segment.
 2. seg_scan       Thread 0 folds agg[b][0..s-1] (≤ 64 segments) to get the A
                   entering the segment. The block scan gives each thread the map
                   of everything before it, so its starting A is excl.A * A_in + excl.X.
                   It runs the plain recurrence over its 4 items, and results are
                   stored coalesced to advantages[b][t].
The block scan is a warp-shuffle scan followed by a scan over the 8 warp totals.
Padding past the row end uses the identity map (1, 0). agg is a static device
array (256 x 64 float2), so there's no allocation.
*/

#include <cuda_runtime.h>

#define THREADS 256
#define ITEMS   4
#define SEG     (THREADS * ITEMS)           // 1024
#define NWARPS  (THREADS / 32)
#define PAD(i)  ((i) + ((i) >> 5))
#define MAX_B   256
#define MAX_SEG (65536 / SEG)               // 64

__device__ float2 g_agg[MAX_B * MAX_SEG];

__device__ __forceinline__ float2 compose(float2 l, float2 r)   // apply l, then r
{
    return make_float2(l.x * r.x, fmaf(r.x, l.y, r.y));
}
__device__ __forceinline__ float2 shfl_up2(float2 v, int o)
{
    return make_float2(__shfl_up_sync(0xffffffff, v.x, o), __shfl_up_sync(0xffffffff, v.y, o));
}

// exclusive block scan of per-thread maps; *total = whole block's map
__device__ __forceinline__ float2 block_scan(float2 v, float2* total)
{
    __shared__ float2 wa[NWARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float2 I = make_float2(1.f, 0.f);
    float2 inc = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float2 y = shfl_up2(inc, o);
        if (lane >= o) inc = compose(y, inc);
    }
    float2 exc = shfl_up2(inc, 1);
    if (lane == 0) exc = I;
    if (lane == 31) wa[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        float2 w = (lane < NWARPS) ? wa[lane] : I;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float2 y = shfl_up2(w, o);
            if (lane >= o) w = compose(y, w);
        }
        if (lane < NWARPS) wa[lane] = w;
    }
    __syncthreads();
    *total = wa[NWARPS - 1];
    return compose(warp ? wa[warp - 1] : I, exc);
}

// Load segment (reversed-time positions i in [seg0, seg0+SEG)) as maps into smem.
__device__ __forceinline__ void load_maps(const float* __restrict__ r, const float* __restrict__ v,
                                          float* sa, float* sx, size_t row, int seg0, int S,
                                          float gamma, float c)
{
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int j = k * THREADS + threadIdx.x;
        const int i = seg0 + j;                    // reversed position
        float a = 1.f, x = 0.f;
        if (i < S) {
            const int t = S - 1 - i;
            const float vn = (t + 1 < S) ? __ldg(v + row + t + 1) : 0.f;
            x = __ldg(r + row + t) + gamma * vn - __ldg(v + row + t);
            a = c;
        }
        sa[PAD(j)] = a;
        sx[PAD(j)] = x;
    }
    __syncthreads();
}

__global__ void __launch_bounds__(THREADS)
seg_aggregate(const float* __restrict__ r, const float* __restrict__ v, int S, int nSeg,
              float gamma, float c)
{
    __shared__ float sa[SEG + SEG / 32], sx[SEG + SEG / 32];
    const int s = blockIdx.x, b = blockIdx.y;
    load_maps(r, v, sa, sx, (size_t)b * S, s * SEG, S, gamma, c);
    float2 m = make_float2(1.f, 0.f);
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int j = threadIdx.x * ITEMS + k;
        m = compose(m, make_float2(sa[PAD(j)], sx[PAD(j)]));
    }
    float2 tot;
    block_scan(m, &tot);
    if (threadIdx.x == 0) g_agg[b * nSeg + s] = tot;
}

__global__ void __launch_bounds__(THREADS)
seg_scan(const float* __restrict__ r, const float* __restrict__ v, float* __restrict__ adv,
         int S, int nSeg, float gamma, float c)
{
    __shared__ float sa[SEG + SEG / 32], sx[SEG + SEG / 32];
    __shared__ float ain;
    const int s = blockIdx.x, b = blockIdx.y;
    const size_t row = (size_t)b * S;
    const int seg0 = s * SEG;

    if (threadIdx.x == 0) {                        // A entering this segment
        float h = 0.f;
        for (int q = 0; q < s; ++q) { const float2 g = g_agg[b * nSeg + q]; h = fmaf(g.x, h, g.y); }
        ain = h;
    }
    load_maps(r, v, sa, sx, row, seg0, S, gamma, c);   // has __syncthreads

    float2 m = make_float2(1.f, 0.f);
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int j = threadIdx.x * ITEMS + k;
        m = compose(m, make_float2(sa[PAD(j)], sx[PAD(j)]));
    }
    float2 tot;
    const float2 ex = block_scan(m, &tot);         // has __syncthreads

    float h = fmaf(ex.x, ain, ex.y);
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {              // own slots only
        const int j = threadIdx.x * ITEMS + k;
        h = fmaf(sa[PAD(j)], h, sx[PAD(j)]);
        sx[PAD(j)] = h;
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {              // coalesced store (descending t)
        const int j = k * THREADS + threadIdx.x;
        const int i = seg0 + j;
        if (i < S) adv[row + (S - 1 - i)] = sx[PAD(j)];
    }
}

// rewards, values, advantages are device pointers
extern "C" void solve(const float* rewards, const float* values, float* advantages, float gamma,
                      float lam, int B, int S)
{
    if (B <= 0 || S <= 0) return;
    const float c = gamma * lam;
    const int nSeg = (S + SEG - 1) / SEG;
    dim3 grid(nSeg, B);
    if (nSeg > 1) seg_aggregate<<<grid, THREADS>>>(rewards, values, S, nSeg, gamma, c);
    seg_scan<<<grid, THREADS>>>(rewards, values, advantages, S, nSeg, gamma, c);
    cudaDeviceSynchronize();
}
