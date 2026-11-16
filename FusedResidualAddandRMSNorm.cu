/*
Fused Residual Add and RMS Norm
For each row i of x, residual (N x C, float32):
  z      = x[i] + residual[i]
  rms_i  = sqrt(mean_j(z_j^2) + eps)
  out[i, j] = z_j / rms_i * weight[j]

Constraints
 - 1 ≤ N, C ≤ 65,536;  eps = 1e-5
 - Performance is measured with N = C = 4,096
*/

/*
Approach: one block (256 threads) per row; z never touches global memory.

 - Fast path (C % 4 == 0 and C ≤ 8192): templated on V = float4s per thread
   (V = 4 at C = 4096). Each thread loads V float4s of x and residual
   (coalesced, strided by blockDim), adds them in registers, and accumulates
   sum(z^2). A warp-shuffle + shared-memory block reduce gives the row's
   1/rms, then each thread scales its z values (still in registers) by
   weight and stores float4. Traffic is the minimum: read x, residual (and
   weight, which stays cached in L1/L2) once, write out once.
 - General path (C % 4 != 0, or C > 8192): two loops over the row. The first
   accumulates sum(z^2); the second recomputes z = x + residual, which is
   mostly an L2 hit since a block's row is ≤ 512 KB, and writes the output.
   It uses float4 when C % 4 == 0 and scalars otherwise.

Every row's reduction is done in fp32 with rsqrtf.
*/

#include <cuda_runtime.h>

#define THREADS 256
#define NWARPS  (THREADS / 32)
#define MAX_V   8                      // float4s per thread in registers -> C ≤ 8192

// Block-wide sum, result broadcast to all threads.
__device__ __forceinline__ float block_sum(float v)
{
    __shared__ float ws[NWARPS];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < NWARPS) ? ws[lane] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) total = v;
    }
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------- fast path
template <int V>
__global__ void __launch_bounds__(THREADS)
rmsnorm_regs(const float4* __restrict__ x, const float4* __restrict__ res,
             const float4* __restrict__ w, float4* __restrict__ out,
             int C4, float invC, float eps)
{
    const size_t base = (size_t)blockIdx.x * C4;
    float4 z[V];
    float ss = 0.f;

    #pragma unroll
    for (int j = 0; j < V; ++j) {
        const int c = threadIdx.x + j * THREADS;
        if (c < C4) {
            const float4 a = __ldg(x + base + c);
            const float4 b = __ldg(res + base + c);
            z[j] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            ss += z[j].x * z[j].x + z[j].y * z[j].y + z[j].z * z[j].z + z[j].w * z[j].w;
        }
    }

    const float r = rsqrtf(block_sum(ss) * invC + eps);

    #pragma unroll
    for (int j = 0; j < V; ++j) {
        const int c = threadIdx.x + j * THREADS;
        if (c < C4) {
            const float4 g = __ldg(w + c);
            out[base + c] = make_float4(z[j].x * r * g.x, z[j].y * r * g.y,
                                        z[j].z * r * g.z, z[j].w * r * g.w);
        }
    }
}

// ---------------------------------------------------------------- general path
__global__ void __launch_bounds__(THREADS)
rmsnorm_general(const float* __restrict__ x, const float* __restrict__ res,
                const float* __restrict__ w, float* __restrict__ out,
                int C, float invC, float eps, bool vec)
{
    const size_t base = (size_t)blockIdx.x * C;
    float ss = 0.f;

    if (vec) {
        const int C4 = C >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x + base);
        const float4* r4 = reinterpret_cast<const float4*>(res + base);
        for (int c = threadIdx.x; c < C4; c += THREADS) {
            const float4 a = __ldg(x4 + c), b = __ldg(r4 + c);
            const float zx = a.x + b.x, zy = a.y + b.y, zz = a.z + b.z, zw = a.w + b.w;
            ss += zx * zx + zy * zy + zz * zz + zw * zw;
        }
        const float r = rsqrtf(block_sum(ss) * invC + eps);
        const float4* w4 = reinterpret_cast<const float4*>(w);
        float4* o4 = reinterpret_cast<float4*>(out + base);
        for (int c = threadIdx.x; c < C4; c += THREADS) {
            const float4 a = __ldg(x4 + c), b = __ldg(r4 + c), g = __ldg(w4 + c);
            o4[c] = make_float4((a.x + b.x) * r * g.x, (a.y + b.y) * r * g.y,
                                (a.z + b.z) * r * g.z, (a.w + b.w) * r * g.w);
        }
    } else {
        for (int c = threadIdx.x; c < C; c += THREADS) {
            const float z = __ldg(x + base + c) + __ldg(res + base + c);
            ss += z * z;
        }
        const float r = rsqrtf(block_sum(ss) * invC + eps);
        for (int c = threadIdx.x; c < C; c += THREADS) {
            const float z = __ldg(x + base + c) + __ldg(res + base + c);
            out[base + c] = z * r * __ldg(w + c);
        }
    }
}

// ---------------------------------------------------------------- host
// x, residual, weight, out are device pointers
extern "C" void solve(const float* x, const float* residual, const float* weight, float* out, int N,
                      int C, float eps)
{
    if (N <= 0 || C <= 0) return;
    const float invC = 1.0f / (float)C;
    const bool vec = (C % 4) == 0;

    if (vec && C <= MAX_V * THREADS * 4) {
        const int C4 = C / 4;
        const int V = (C4 + THREADS - 1) / THREADS;          // 1..8
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* r4 = reinterpret_cast<const float4*>(residual);
        const float4* w4 = reinterpret_cast<const float4*>(weight);
        float4* o4 = reinterpret_cast<float4*>(out);
        switch (V) {
            case 1: rmsnorm_regs<1><<<N, THREADS>>>(x4, r4, w4, o4, C4, invC, eps); break;
            case 2: rmsnorm_regs<2><<<N, THREADS>>>(x4, r4, w4, o4, C4, invC, eps); break;
            case 3: rmsnorm_regs<3><<<N, THREADS>>>(x4, r4, w4, o4, C4, invC, eps); break;
            case 4: rmsnorm_regs<4><<<N, THREADS>>>(x4, r4, w4, o4, C4, invC, eps); break;
            case 5: rmsnorm_regs<5><<<N, THREADS>>>(x4, r4, w4, o4, C4, invC, eps); break;
            case 6: rmsnorm_regs<6><<<N, THREADS>>>(x4, r4, w4, o4, C4, invC, eps); break;
            case 7: rmsnorm_regs<7><<<N, THREADS>>>(x4, r4, w4, o4, C4, invC, eps); break;
            default: rmsnorm_regs<8><<<N, THREADS>>>(x4, r4, w4, o4, C4, invC, eps); break;
        }
    } else {
        rmsnorm_general<<<N, THREADS>>>(x, residual, weight, out, C, invC, eps, vec);
    }
    cudaDeviceSynchronize();
}
