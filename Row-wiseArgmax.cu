#include <cuda_runtime.h>

#define THREADS 256
#define WARPS   (THREADS / 32)

typedef unsigned long long u64;

// Monotone float -> uint mapping (finite floats, no NaN)
__device__ __forceinline__ unsigned int orderable(float f)
{
    f += 0.0f;                                   // canonicalize -0.0 to +0.0
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ u64 makeKey(float v, int idx)
{
    return ((u64)orderable(v) << 32) | (u64)(~(unsigned int)idx);
}

__device__ __forceinline__ u64 warpMax(u64 k)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        u64 o = __shfl_down_sync(0xffffffff, k, off);
        k = (o > k) ? o : k;
    }
    return k;
}

// grid = (splits, B): block (s, b) handles one chunk of row b
__global__ void argmax_partial(const float* __restrict__ logits,
                               u64* __restrict__ keys,
                               int V, int chunk)
{
    __shared__ u64 warpKeys[WARPS];

    const int b     = blockIdx.y;
    const int start = blockIdx.x * chunk;
    const int end   = min(V, start + chunk);
    const float* row = logits + (size_t)b * V;
    const int tid = threadIdx.x;

    float best = -__int_as_float(0x7f800000);    // -inf
    int   bestIdx = 0x7fffffff;

    if ((V & 3) == 0) {
        // Vectorized path: row and chunk starts are 16-byte aligned
        const float4* r4 = reinterpret_cast<const float4*>(row);
        for (int i = (start >> 2) + tid; i < (end >> 2); i += THREADS) {
            float4 v = __ldg(&r4[i]);
            int base = i << 2;                   // indices increase, so strict '>' keeps the first max
            if (v.x > best) { best = v.x; bestIdx = base;     }
            if (v.y > best) { best = v.y; bestIdx = base + 1; }
            if (v.z > best) { best = v.z; bestIdx = base + 2; }
            if (v.w > best) { best = v.w; bestIdx = base + 3; }
        }
    } else {
        for (int i = start + tid; i < end; i += THREADS) {
            float v = __ldg(&row[i]);
            if (v > best) { best = v; bestIdx = i; }
        }
    }

    u64 key = (bestIdx == 0x7fffffff) ? 0ull : makeKey(best, bestIdx);

    key = warpMax(key);
    if ((tid & 31) == 0) warpKeys[tid >> 5] = key;
    __syncthreads();

    if (tid < 32) {
        key = (tid < WARPS) ? warpKeys[tid] : 0ull;
        key = warpMax(key);
        if (tid == 0) atomicMax(&keys[b], key);
    }
}

__global__ void argmax_finalize(const u64* __restrict__ keys, int* __restrict__ output, int B)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) output[b] = (int)(~(unsigned int)(keys[b] & 0xffffffffull));
}

// logits, output are device pointers
extern "C" void solve(const float* logits, int* output, int B, int V)
{
    static u64* keys = nullptr;
    static int numSMs = 0;
    if (!keys) {
        cudaMalloc(&keys, 1024 * sizeof(u64));
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
    }
    cudaMemset(keys, 0, B * sizeof(u64));        // 0 is below every real key

    // Split rows so the grid has ~8 blocks per SM, each chunk >= 4096 elements
    int splits = (numSMs * 8 + B - 1) / B;
    int maxSplits = (V + 4095) / 4096;
    if (splits > maxSplits) splits = maxSplits;
    if (splits < 1) splits = 1;
    int chunk = (V + splits - 1) / splits;
    chunk = (chunk + 3) & ~3;                    // keep chunk starts float4-aligned
    splits = (V + chunk - 1) / chunk;

    argmax_partial<<<dim3(splits, B), THREADS>>>(logits, keys, V, chunk);
    argmax_finalize<<<(B + 255) / 256, 256>>>(keys, output, B);
    cudaDeviceSynchronize();
}
