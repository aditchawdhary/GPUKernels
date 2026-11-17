/*
Fused Residual Add + LayerNorm with Saved Statistics
x, residual, y, residual_out: (N, C); gamma, beta: (C); mean, rstd: (N). float32.
Spec from the reference implementation (leetgpu-challenges #143), tolerance 1e-3:
  h            = fl32(x + residual)            -> residual_out
  mu           = mean(h),  var = mean((h - mu)^2)   (biased),  rstd = 1/sqrt(var + eps)
  y            = (h - mu) * rstd * gamma + beta
  mean, rstd   saved per row
Constraints: N ≤ 65,536, C ≤ 16,384, |x|, |residual| ≤ 1000, eps in [1e-6, 1].
Performance at N = 16,384, C = 4,096.
*/

/*
Approach: one 512-thread block per row, three sweeps over the row. Only the first
touches DRAM for the inputs; the second and third re-read h from residual_out, which
this block has just written and is still in L1/L2 (a row is ≤ 64 KB).

 1. h = x + residual, rounded to float exactly as stored, and written to residual_out.
    Accumulate sum(h - h0), where h0 is the row's first element. This shifted sum
    keeps the per-thread fp32 partials small even for rows with a large DC offset
    (the mean ~ 500 test). Block-reduce in double: mu = h0 + S / C.
 2. Re-read h; accumulate sum((h - mu)^2) (two-pass, centered, so there's no
    E[h^2] - E[h]^2 cancellation); block-reduce in double; rstd = 1/sqrt(var + eps)
    in double. For C = 1 or constant rows, var = 0 and rstd = 1/sqrt(eps), so y = beta.
 3. Re-read h; y = (h - mu) * (rstd * gamma) + beta.
 Every thread re-reads exactly the elements it wrote (same striding), so no barrier
 is needed between writing and re-reading residual_out. float4 accesses are used when
 C % 4 == 0 (row starts are then 16 B aligned), scalar otherwise.
Traffic: read x, residual, write residual_out and y (1 GB at the benchmark), so it's
bandwidth-bound.
*/

#include <cuda_runtime.h>
#include <math.h>

#define THREADS 512
#define NWARPS  (THREADS / 32)

__device__ __forceinline__ double block_sum(double v)
{
    __shared__ double ws[NWARPS];
    __shared__ double res;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < NWARPS) ? ws[lane] : 0.0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) res = v;
    }
    __syncthreads();
    const double r = res;
    __syncthreads();                                  // ws / res reusable
    return r;
}

template <bool VEC>
__global__ void __launch_bounds__(THREADS)
add_layernorm(const float* __restrict__ x, const float* __restrict__ res,
              const float* __restrict__ gamma, const float* __restrict__ beta,
              float* __restrict__ y, float* __restrict__ hout,
              float* __restrict__ mean, float* __restrict__ rstd, int C, float eps)
{
    const size_t row = (size_t)blockIdx.x * C;
    const float h0 = __fadd_rn(__ldg(x + row), __ldg(res + row));

    // ---- pass 1: h, residual_out, shifted sum
    float s = 0.f;
    if (VEC) {
        const int C4 = C >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x + row);
        const float4* r4 = reinterpret_cast<const float4*>(res + row);
        float4* h4 = reinterpret_cast<float4*>(hout + row);
        for (int i = threadIdx.x; i < C4; i += THREADS) {
            const float4 a = __ldg(x4 + i), b = __ldg(r4 + i);
            const float4 h = make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y),
                                         __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
            h4[i] = h;
            s += (h.x - h0) + (h.y - h0) + (h.z - h0) + (h.w - h0);
        }
    } else {
        for (int i = threadIdx.x; i < C; i += THREADS) {
            const float h = __fadd_rn(__ldg(x + row + i), __ldg(res + row + i));
            hout[row + i] = h;
            s += h - h0;
        }
    }
    const double mu_d = (double)h0 + block_sum((double)s) / (double)C;
    const float mu = (float)mu_d;

    // ---- pass 2: centered variance
    float q = 0.f;
    if (VEC) {
        const float4* h4 = reinterpret_cast<const float4*>(hout + row);
        for (int i = threadIdx.x; i < (C >> 2); i += THREADS) {
            const float4 h = h4[i];
            const float a = h.x - mu, b = h.y - mu, c = h.z - mu, d = h.w - mu;
            q += a * a + b * b + c * c + d * d;
        }
    } else {
        for (int i = threadIdx.x; i < C; i += THREADS) {
            const float a = hout[row + i] - mu;
            q += a * a;
        }
    }
    const double var = block_sum((double)q) / (double)C;
    const double r_d = 1.0 / sqrt(var + (double)eps);
    const float r = (float)r_d;

    if (threadIdx.x == 0) { mean[blockIdx.x] = mu; rstd[blockIdx.x] = r; }

    // ---- pass 3: normalize, scale, shift
    if (VEC) {
        const float4* h4 = reinterpret_cast<const float4*>(hout + row);
        const float4* g4 = reinterpret_cast<const float4*>(gamma);
        const float4* b4 = reinterpret_cast<const float4*>(beta);
        float4* y4 = reinterpret_cast<float4*>(y + row);
        for (int i = threadIdx.x; i < (C >> 2); i += THREADS) {
            const float4 h = h4[i], g = __ldg(g4 + i), b = __ldg(b4 + i);
            y4[i] = make_float4(fmaf(h.x - mu, r * g.x, b.x), fmaf(h.y - mu, r * g.y, b.y),
                                fmaf(h.z - mu, r * g.z, b.z), fmaf(h.w - mu, r * g.w, b.w));
        }
    } else {
        for (int i = threadIdx.x; i < C; i += THREADS)
            y[row + i] = fmaf(hout[row + i] - mu, r * __ldg(gamma + i), __ldg(beta + i));
    }
}

// x, residual, gamma, beta, y, residual_out, mean, rstd are device pointers
extern "C" void solve(const float* x, const float* residual, const float* gamma, const float* beta,
                      float* y, float* residual_out, float* mean, float* rstd, int N, int C,
                      float eps)
{
    if (N <= 0 || C <= 0) return;
    if (C % 4 == 0)
        add_layernorm<true ><<<N, THREADS>>>(x, residual, gamma, beta, y, residual_out, mean, rstd, C, eps);
    else
        add_layernorm<false><<<N, THREADS>>>(x, residual, gamma, beta, y, residual_out, mean, rstd, C, eps);
    cudaDeviceSynchronize();
}
