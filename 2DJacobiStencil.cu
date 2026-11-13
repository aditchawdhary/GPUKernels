#include <cuda_runtime.h>

// One 5-point Jacobi step: interior out[i][j] = 0.25 * (top + bottom + left + right),
// boundary cells copied. Summation order matches the reference: ((top + bottom) + left) + right.
//
// Fast path (cols % 4 == 0, 16-byte aligned): each thread owns 4 adjacent columns (one float4)
// and marches down RCHUNK rows, keeping the rows above / at / below in registers, so each input
// value is fetched from DRAM ~once (plus one halo row per chunk). Horizontal neighbours across
// the float4 boundary come from adjacent lanes via warp shuffles; only the warp's edge lanes
// issue an extra scalar load. Stores are float4.
// Fallback: one thread per element.

#define VT 128        // threads per block in the fast path (4 warps)
#define RCHUNK 32     // rows per thread

__device__ __forceinline__ float jac(float t, float b, float l, float r) {
    return 0.25f * (((t + b) + l) + r);
}

__global__ void __launch_bounds__(VT)
jacobi_vec4(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    const int ncol4 = cols >> 2;
    const int c4 = blockIdx.x * VT + threadIdx.x;
    const bool active = c4 < ncol4;
    const int col0 = c4 * 4;
    const int lane = threadIdx.x & 31;
    const int r0 = blockIdx.y * RCHUNK;
    const int r1 = min(rows, r0 + RCHUNK);

    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);

    auto ld = [&](int r) -> float4 {
        return (active && r >= 0 && r < rows) ? __ldg(&in4[(size_t)r * ncol4 + c4]) : zero;
    };

    float4 up = ld(r0 - 1);
    float4 mid = ld(r0);

    for (int r = r0; r < r1; r++) {
        float4 dn = ld(r + 1);

        // horizontal neighbours across float4 boundaries (all lanes take part in the shuffles)
        float left = __shfl_up_sync(0xffffffffu, mid.w, 1);
        float right = __shfl_down_sync(0xffffffffu, mid.x, 1);
        if (active) {
            const size_t rb = (size_t)r * cols;
            if (lane == 0)  left  = (col0 > 0) ? __ldg(&in[rb + col0 - 1]) : 0.0f;
            if (lane == 31) right = (col0 + 4 < cols) ? __ldg(&in[rb + col0 + 4]) : 0.0f;

            float4 o = mid;
            if (r > 0 && r < rows - 1) {
                // element 0 (col0) / 3 (col0+3) may be boundary columns
                if (col0 > 0)            o.x = jac(up.x, dn.x, left,  mid.y);
                                         o.y = jac(up.y, dn.y, mid.x, mid.z);
                                         o.z = jac(up.z, dn.z, mid.y, mid.w);
                if (col0 + 3 < cols - 1) o.w = jac(up.w, dn.w, mid.z, right);
                // cols % 4 == 0, so only element 0 or 3 can be a boundary column
            }
            out4[(size_t)r * ncol4 + c4] = o;
        }
        up = mid;
        mid = dn;
    }
}

__global__ void jacobi_scalar(const float* __restrict__ in, float* __restrict__ out, int rows,
                              int cols) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= cols) return;
    const size_t idx = (size_t)i * cols + j;
    if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
        out[idx] = in[idx];
    } else {
        out[idx] = jac(__ldg(&in[idx - cols]), __ldg(&in[idx + cols]), __ldg(&in[idx - 1]),
                       __ldg(&in[idx + 1]));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    bool aligned = ((reinterpret_cast<size_t>(input) | reinterpret_cast<size_t>(output)) & 15) == 0;
    if (cols % 4 == 0 && aligned) {
        dim3 grid((cols / 4 + VT - 1) / VT, (rows + RCHUNK - 1) / RCHUNK);
        jacobi_vec4<<<grid, VT>>>(input, output, rows, cols);
    } else {
        dim3 block(32, 8);
        dim3 grid((cols + 31) / 32, (rows + 7) / 8);
        jacobi_scalar<<<grid, block>>>(input, output, rows, cols);
    }
}
