#include <cuda_fp16.h>
#include <cuda_runtime.h>

// FP16 dot product with FP32 accumulation.
// Memory-bound: stream A and B with 16-byte loads (8 halves each), convert pairs with
// __half22float2, accumulate in fp32 registers, reduce per block (warp shuffles), then a
// single block sums the per-block partials in a fixed order (deterministic) and writes half.

#define THREADS 256
#define BLOCKS 1024

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float ws[THREADS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    v = (threadIdx.x < THREADS / 32) ? ws[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    return v;   // valid in thread 0
}

__device__ __forceinline__ float dot8(const uint4& a, const uint4& b) {
    const __half2* ha = reinterpret_cast<const __half2*>(&a);
    const __half2* hb = reinterpret_cast<const __half2*>(&b);
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 fa = __half22float2(ha[i]);
        float2 fb = __half22float2(hb[i]);
        s = fmaf(fa.x, fb.x, s);
        s = fmaf(fa.y, fb.y, s);
    }
    return s;
}

__global__ void __launch_bounds__(THREADS)
dot_partial(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ partial,
            int N, bool vec) {
    float acc0 = 0.0f, acc1 = 0.0f;
    const int tid = blockIdx.x * THREADS + threadIdx.x;
    const int stride = gridDim.x * THREADS;
    int tail_start = 0;

    if (vec) {
        const uint4* A8 = reinterpret_cast<const uint4*>(A);
        const uint4* B8 = reinterpret_cast<const uint4*>(B);
        const int n8 = N / 8;
        int i = tid;
        // two independent 16-byte loads in flight per operand per iteration
        for (; i + stride < n8; i += 2 * stride) {
            uint4 a0 = __ldg(&A8[i]), b0 = __ldg(&B8[i]);
            uint4 a1 = __ldg(&A8[i + stride]), b1 = __ldg(&B8[i + stride]);
            acc0 += dot8(a0, b0);
            acc1 += dot8(a1, b1);
        }
        if (i < n8) acc0 += dot8(__ldg(&A8[i]), __ldg(&B8[i]));
        tail_start = n8 * 8;
    }
    for (int i = tail_start + tid; i < N; i += stride)
        acc0 = fmaf(__half2float(A[i]), __half2float(B[i]), acc0);

    float s = block_sum(acc0 + acc1);
    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

__global__ void __launch_bounds__(THREADS)
dot_final(const float* __restrict__ partial, int n, half* __restrict__ result) {
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += THREADS) v += partial[i];
    v = block_sum(v);
    if (threadIdx.x == 0) *result = __float2half_rn(v);
}

static float* g_partial = nullptr;

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    if (!g_partial) cudaMalloc(&g_partial, BLOCKS * sizeof(float));

    int blocks = (N + THREADS * 8 - 1) / (THREADS * 8);
    if (blocks > BLOCKS) blocks = BLOCKS;
    if (blocks < 1) blocks = 1;

    bool vec = ((reinterpret_cast<size_t>(A) | reinterpret_cast<size_t>(B)) & 15) == 0;
    dot_partial<<<blocks, THREADS>>>(A, B, g_partial, N, vec);
    dot_final<<<1, THREADS>>>(g_partial, blocks, result);
}
