/*
RMSNorm Backward
x, dy, dx: (N, C); weight, dweight: (C). float32.
Spec from the reference implementation (leetgpu-challenges #142), tolerance 1e-3:
  rstd = 1/sqrt(mean(x^2) + eps),  xhat = x * rstd,  g = dy * weight
  dx      = rstd * (g - xhat * mean(g * xhat))          (per row)
  dweight = sum over rows of dy * xhat                   (per column)
Constraints: N ≤ 65,536, C ≤ 16,384, |x| ≤ 250, |dy| ≤ 10, |w| ≤ 2, eps in [1e-6, 1].
Performance at N = 16,384, C = 4,096.
*/

/*
Approach: dx needs two row reductions; dweight needs a column reduction across all
rows. One kernel does both by giving each block a contiguous chunk of rows, plus a
small deterministic column-sum kernel.

 1. rms_bwd_rows   grid = G row-chunks (about 3 per SM, so G is roughly 300-400), 512
                   threads. Thread t owns columns t, t+512, ... (KMAX of them,
                   templated on 1/2/4/8/16/32 for C up to 16,384), so its weight
                   values and its dweight accumulators stay in registers for the
                   whole chunk. For each row:
                     - load x and dy for its columns (coalesced) and keep them in
                       registers
                     - one fused block reduction (double) of S1 = sum x^2 and
                       S2 = sum dy*w*x
                     - rstd = 1/sqrt(S1/C + eps); since mean(g*xhat) = rstd*S2/C,
                       proj = rstd*S2/C
                     - dx = rstd*(dy*w - x*rstd*proj); dweight_acc += dy*x*rstd
                   At the end of the chunk it writes its partial dweight row to scratch.
 2. col_sum        dweight[c] = sum over the G partials (double, fixed order), so the
                   result is identical run to run. No atomics.
 Each row is read once and dx written once (bandwidth-bound: about 0.8 GB at the
 benchmark). The dweight partials are only G x C floats.
 Edge cases from the tests: all-zero x gives rstd = 1/sqrt(eps) (eps is inside the
 sqrt); dy = 0 gives exact zeros; C = 1 cancels to ~0 through the projection term.
*/

#include <cuda_runtime.h>
#include <math.h>

#define THREADS 512
#define NWARPS  (THREADS / 32)

__device__ __forceinline__ double2 block_sum2(double a, double b)
{
    __shared__ double2 ws[NWARPS];
    __shared__ double2 res;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a += __shfl_xor_sync(0xffffffff, a, o);
        b += __shfl_xor_sync(0xffffffff, b, o);
    }
    if (lane == 0) ws[warp] = make_double2(a, b);
    __syncthreads();
    if (warp == 0) {
        double2 v = (lane < NWARPS) ? ws[lane] : make_double2(0.0, 0.0);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v.x += __shfl_xor_sync(0xffffffff, v.x, o);
            v.y += __shfl_xor_sync(0xffffffff, v.y, o);
        }
        if (lane == 0) res = v;
    }
    __syncthreads();
    const double2 r = res;
    __syncthreads();                                   // ws / res reusable next row
    return r;
}

template <int KMAX>
__global__ void __launch_bounds__(THREADS)
rms_bwd_rows(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ dy,
             float* __restrict__ dx, float* __restrict__ part, int N, int C, float eps, int rowsPer)
{
    const int r0 = blockIdx.x * rowsPer;
    const int r1 = min(r0 + rowsPer, N);

    float wv[KMAX], acc[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        const int c = threadIdx.x + k * THREADS;
        wv[k] = (c < C) ? __ldg(w + c) : 0.f;
        acc[k] = 0.f;
    }

    for (int r = r0; r < r1; ++r) {
        const size_t row = (size_t)r * C;
        float xv[KMAX], gv[KMAX];
        float s1 = 0.f, s2 = 0.f;
        #pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            const int c = threadIdx.x + k * THREADS;
            xv[k] = 0.f; gv[k] = 0.f;
            if (c < C) {
                xv[k] = __ldg(x + row + c);
                gv[k] = __ldg(dy + row + c);               // dy for now
                s1 = fmaf(xv[k], xv[k], s1);
                s2 = fmaf(gv[k] * wv[k], xv[k], s2);
            }
        }
        const double2 S = block_sum2((double)s1, (double)s2);
        const float rstd = (float)(1.0 / sqrt(S.x / C + (double)eps));
        const float proj = (float)((double)rstd * S.y / C);      // mean(g * xhat)
        const float rp = rstd * proj;
        #pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            const int c = threadIdx.x + k * THREADS;
            if (c < C) {
                const float d = gv[k];
                dx[row + c] = rstd * fmaf(d, wv[k], -xv[k] * rp);
                acc[k] = fmaf(d * xv[k], rstd, acc[k]);
            }
        }
    }

    float* p = part + (size_t)blockIdx.x * C;
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        const int c = threadIdx.x + k * THREADS;
        if (c < C) p[c] = acc[k];
    }
}

__global__ void col_sum(const float* __restrict__ part, float* __restrict__ dweight, int G, int C)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    double s = 0.0;
    for (int g = 0; g < G; ++g) s += part[(size_t)g * C + c];
    dweight[c] = (float)s;
}

static float* g_part = nullptr;
static size_t g_cap = 0;

template <int KMAX>
static void run_rows(const float* x, const float* w, const float* dy, float* dx, float* part,
                     int N, int C, float eps, int G, int rowsPer)
{
    rms_bwd_rows<KMAX><<<G, THREADS>>>(x, w, dy, dx, part, N, C, eps, rowsPer);
}

// x, weight, dy, dx, dweight are device pointers
extern "C" void solve(const float* x, const float* weight, const float* dy, float* dx,
                      float* dweight, int N, int C, float eps)
{
    if (N <= 0 || C <= 0) return;
    int dev = 0, sms = 80;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    int G = 3 * sms;
    if (G > N) G = N;
    const int rowsPer = (N + G - 1) / G;
    G = (N + rowsPer - 1) / rowsPer;

    const size_t need = (size_t)G * C;
    if (need > g_cap) {
        if (g_part) cudaFree(g_part);
        cudaMalloc(&g_part, need * sizeof(float));
        g_cap = need;
    }

    const int km = (C + THREADS - 1) / THREADS;       // columns per thread
    if      (km <= 1)  run_rows<1> (x, weight, dy, dx, g_part, N, C, eps, G, rowsPer);
    else if (km <= 2)  run_rows<2> (x, weight, dy, dx, g_part, N, C, eps, G, rowsPer);
    else if (km <= 4)  run_rows<4> (x, weight, dy, dx, g_part, N, C, eps, G, rowsPer);
    else if (km <= 8)  run_rows<8> (x, weight, dy, dx, g_part, N, C, eps, G, rowsPer);
    else if (km <= 16) run_rows<16>(x, weight, dy, dx, g_part, N, C, eps, G, rowsPer);
    else               run_rows<32>(x, weight, dy, dx, g_part, N, C, eps, G, rowsPer);

    col_sum<<<(C + 255) / 256, 256>>>(g_part, dweight, G, C);
    cudaDeviceSynchronize();
}
