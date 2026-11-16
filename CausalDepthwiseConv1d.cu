/*
Causal Depthwise Conv1d
x, output: (B, L, D) channels-last; weight: (D, K); bias: (D,)
  output[b, l, d] = bias[d] + sum_{k=0}^{K-1} weight[d, k] * x[b, l - k, d]
with x[b, l - k, d] = 0 when l - k < 0.  (weight[d, 0] multiplies the current input.)

Constraints
 - 1 ≤ B ≤ 16, 1 ≤ L ≤ 8192, 1 ≤ D ≤ 8192, 1 ≤ K ≤ 8
 - Performance is measured with B = 8, L = 2048, D = 4096, K = 4
*/

/*
Approach: a register sliding window, one thread per (channel group, run of positions).

 - Each thread owns 4 adjacent channels (one float4, when D % 4 == 0; otherwise 1
   channel) and walks a run of LT = 32 consecutive positions. It keeps the last K
   inputs in a register window, so each step does one coalesced float4 load, K FMAs
   per channel, and one float4 store. Only the K-1 halo rows before the run are
   loaded twice, about 9% extra at K = 4.
 - K is a template parameter (1..8), so the window shift and the FMA loop unroll
   completely and the window lives in registers. Weights and bias are loaded into
   registers once per thread.
 - Grid: x = channel groups (128 threads per block), y = position runs, z = batch.
   At the benchmark that's 8 x 64 x 8 = 4096 blocks. A warp touches 512 contiguous
   bytes per load and per store.

Traffic is about 1.1 reads of x plus 1 write of the output (about 600 MB at the
benchmark), so the kernel is bandwidth-bound.
*/

#include <cuda_runtime.h>

#define THREADS 128
#define LT 32                   // positions per thread

template <int K>
__global__ void __launch_bounds__(THREADS)
dwconv_vec4(const float4* __restrict__ x, const float* __restrict__ w,
            const float4* __restrict__ bias, float4* __restrict__ out,
            int L, int D4)
{
    const int c = blockIdx.x * THREADS + threadIdx.x;   // float4 channel group
    if (c >= D4) return;
    const int l0 = blockIdx.y * LT;
    const int b  = blockIdx.z;
    const int l1 = min(l0 + LT, L);

    // weights for channels 4c..4c+3: wk[k] = (w[4c][k], w[4c+1][k], w[4c+2][k], w[4c+3][k])
    float4 wk[K];
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const int d = c * 4;
        wk[k] = make_float4(__ldg(w + (size_t)(d    ) * K + k), __ldg(w + (size_t)(d + 1) * K + k),
                            __ldg(w + (size_t)(d + 2) * K + k), __ldg(w + (size_t)(d + 3) * K + k));
    }
    const float4 bv = __ldg(bias + c);

    const size_t rowBase = (size_t)b * L;
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);

    // In the loop win[j] = x[l - j]. Preload the halo so that after the first shift
    // win[j] = x[l0 - j]: before the shift, win[j-1] = x[l0 - j].
    float4 win[K];
    win[K - 1] = z;
    #pragma unroll
    for (int j = 1; j < K; ++j) {
        const int l = l0 - j;
        win[j - 1] = (l >= 0) ? __ldg(x + (rowBase + l) * D4 + c) : z;
    }

    for (int l = l0; l < l1; ++l) {
        #pragma unroll
        for (int j = K - 1; j > 0; --j) win[j] = win[j - 1];
        win[0] = __ldg(x + (rowBase + l) * D4 + c);

        float4 acc = bv;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            acc.x = fmaf(wk[k].x, win[k].x, acc.x);
            acc.y = fmaf(wk[k].y, win[k].y, acc.y);
            acc.z = fmaf(wk[k].z, win[k].z, acc.z);
            acc.w = fmaf(wk[k].w, win[k].w, acc.w);
        }
        out[(rowBase + l) * D4 + c] = acc;
    }
}

template <int K>
__global__ void __launch_bounds__(THREADS)
dwconv_scalar(const float* __restrict__ x, const float* __restrict__ w,
              const float* __restrict__ bias, float* __restrict__ out,
              int L, int D)
{
    const int d = blockIdx.x * THREADS + threadIdx.x;
    if (d >= D) return;
    const int l0 = blockIdx.y * LT;
    const int b  = blockIdx.z;
    const int l1 = min(l0 + LT, L);

    float wk[K];
    #pragma unroll
    for (int k = 0; k < K; ++k) wk[k] = __ldg(w + (size_t)d * K + k);
    const float bv = __ldg(bias + d);
    const size_t rowBase = (size_t)b * L;

    float win[K];                       // same halo layout as dwconv_vec4
    win[K - 1] = 0.f;
    #pragma unroll
    for (int j = 1; j < K; ++j) {
        const int l = l0 - j;
        win[j - 1] = (l >= 0) ? __ldg(x + (rowBase + l) * D + d) : 0.f;
    }

    for (int l = l0; l < l1; ++l) {
        #pragma unroll
        for (int j = K - 1; j > 0; --j) win[j] = win[j - 1];
        win[0] = __ldg(x + (rowBase + l) * D + d);

        float acc = bv;
        #pragma unroll
        for (int k = 0; k < K; ++k) acc = fmaf(wk[k], win[k], acc);
        out[(rowBase + l) * D + d] = acc;
    }
}

template <int K>
static void launch(const float* x, const float* w, const float* bias, float* out,
                   int B, int L, int D)
{
    const int runs = (L + LT - 1) / LT;
    if (D % 4 == 0) {
        const int D4 = D / 4;
        dim3 grid((D4 + THREADS - 1) / THREADS, runs, B);
        dwconv_vec4<K><<<grid, THREADS>>>(reinterpret_cast<const float4*>(x), w,
                                          reinterpret_cast<const float4*>(bias),
                                          reinterpret_cast<float4*>(out), L, D4);
    } else {
        dim3 grid((D + THREADS - 1) / THREADS, runs, B);
        dwconv_scalar<K><<<grid, THREADS>>>(x, w, bias, out, L, D);
    }
}

// x, weight, bias, output are device pointers
extern "C" void solve(const float* x, const float* weight, const float* bias, float* output,
                      int B, int L, int D, int K)
{
    if (B <= 0 || L <= 0 || D <= 0 || K <= 0) return;
    switch (K) {
        case 1: launch<1>(x, weight, bias, output, B, L, D); break;
        case 2: launch<2>(x, weight, bias, output, B, L, D); break;
        case 3: launch<3>(x, weight, bias, output, B, L, D); break;
        case 4: launch<4>(x, weight, bias, output, B, L, D); break;
        case 5: launch<5>(x, weight, bias, output, B, L, D); break;
        case 6: launch<6>(x, weight, bias, output, B, L, D); break;
        case 7: launch<7>(x, weight, bias, output, B, L, D); break;
        default: launch<8>(x, weight, bias, output, B, L, D); break;
    }
    cudaDeviceSynchronize();
}
