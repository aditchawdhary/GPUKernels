#include <cuda_runtime.h>

// Multi-scale deformable attention (Deformable DETR sampling core).
//
// One warp per (query, head):
//   Phase 1: lanes split the num_levels * num_points samples. For each sample
//            they compute the 4 bilinear corners: the row offset into `value`
//            (or -1 if the corner is outside the level grid) and a weight that
//            already includes the attention weight. Results go to shared memory.
//   Phase 2: lanes split head_dim channels and accumulate the weighted corner
//            values. All lanes read the same corner row together, so each
//            gather is one coalesced 128-byte load at head_dim = 32.

#define WARPS_PER_BLOCK 8
#define MAX_LEVELS 8
#define MAX_SAMPLES 64   // num_levels * num_points <= 8 * 8

__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
msda_kernel(const float* __restrict__ value, const int* __restrict__ spatial_shapes,
            const float* __restrict__ sampling_loc, const float* __restrict__ attn_weight,
            float* __restrict__ output, int num_queries, int num_heads, int head_dim,
            int num_levels, int num_points)
{
    __shared__ int   sH[MAX_LEVELS], sW[MAX_LEVELS], sStart[MAX_LEVELS];
    __shared__ int   sIdx[WARPS_PER_BLOCK][MAX_SAMPLES * 4];
    __shared__ float sWt [WARPS_PER_BLOCK][MAX_SAMPLES * 4];

    // Level shapes and exclusive prefix-sum offsets (tiny: <= 8 levels).
    if (threadIdx.x == 0) {
        int start = 0;
        for (int l = 0; l < num_levels; ++l) {
            int h = spatial_shapes[2 * l], w = spatial_shapes[2 * l + 1];
            sH[l] = h; sW[l] = w; sStart[l] = start;
            start += h * w;
        }
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qh   = blockIdx.x * WARPS_PER_BLOCK + warp;     // (query, head) pair
    if (qh >= num_queries * num_heads) return;
    const int h = qh % num_heads;

    const int LP = num_levels * num_points;
    const int rowStride = num_heads * head_dim;                // floats per spatial row of value
    const float2* loc = reinterpret_cast<const float2*>(sampling_loc) + (size_t)qh * LP;
    const float*  aw  = attn_weight + (size_t)qh * LP;
    int*   idx = sIdx[warp];
    float* wt  = sWt[warp];

    // ---- Phase 1: per-sample corner offsets and weights ----
    for (int s = lane; s < LP; s += 32) {
        const int l  = s / num_points;
        const int Hl = sH[l], Wl = sW[l];
        const float2 p = __ldg(loc + s);
        const float  a = __ldg(aw + s);

        // Half-pixel convention: pixel centers at integer coords.
        const float x = p.x * (float)Wl - 0.5f;
        const float y = p.y * (float)Hl - 0.5f;
        const float xf = floorf(x), yf = floorf(y);
        const int   x0 = (int)xf,   y0 = (int)yf;
        const float lx = x - xf,    ly = y - yf;
        const float hx = 1.f - lx,  hy = 1.f - ly;

        const int base = (sStart[l] * num_heads + h) * head_dim;
        const bool vx0 = (x0 >= 0) && (x0 < Wl);
        const bool vx1 = (x0 + 1 >= 0) && (x0 + 1 < Wl);
        const bool vy0 = (y0 >= 0) && (y0 < Hl);
        const bool vy1 = (y0 + 1 >= 0) && (y0 + 1 < Hl);

        const int r00 = y0 * Wl + x0;   // row of the top-left corner within the level
        idx[4 * s + 0] = (vy0 && vx0) ? base + (r00)          * rowStride : -1;
        idx[4 * s + 1] = (vy0 && vx1) ? base + (r00 + 1)      * rowStride : -1;
        idx[4 * s + 2] = (vy1 && vx0) ? base + (r00 + Wl)     * rowStride : -1;
        idx[4 * s + 3] = (vy1 && vx1) ? base + (r00 + Wl + 1) * rowStride : -1;
        wt[4 * s + 0] = a * hy * hx;
        wt[4 * s + 1] = a * hy * lx;
        wt[4 * s + 2] = a * ly * hx;
        wt[4 * s + 3] = a * ly * lx;
    }
    __syncwarp();

    // ---- Phase 2: weighted gather over channels ----
    float* out = output + (size_t)qh * head_dim;
    for (int c = lane; c < head_dim; c += 32) {
        float acc = 0.f;
        for (int k = 0; k < LP * 4; k += 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int o = idx[k + j];          // warp-uniform branch
                if (o >= 0) acc = fmaf(wt[k + j], __ldg(value + o + c), acc);
            }
        }
        out[c] = acc;
    }
}

// value, spatial_shapes, sampling_loc, attn_weight, output are device pointers
extern "C" void solve(const float* value, const int* spatial_shapes, const float* sampling_loc,
                      const float* attn_weight, float* output, int num_queries, int num_heads,
                      int head_dim, int num_levels, int num_points) {
    const int pairs  = num_queries * num_heads;
    const int blocks = (pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    msda_kernel<<<blocks, WARPS_PER_BLOCK * 32>>>(value, spatial_shapes, sampling_loc, attn_weight,
                                                  output, num_queries, num_heads, head_dim,
                                                  num_levels, num_points);
    cudaDeviceSynchronize();
}
