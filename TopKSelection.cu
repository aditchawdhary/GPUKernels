#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256
#define SMALL_SORT_MAX 8192

struct State {
    unsigned prefix, mask;
    unsigned newPrefix, newMask;
    int kRem;
    int srcIsBuf, srcCount, curBuf;
    int doScatter, writeEq;
    int gtCount, candCount;
};

__device__ __forceinline__ unsigned f2k(float f) {
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float k2f(unsigned k) {
    unsigned u = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
    return __uint_as_float(u);
}

// Warp-aggregated append to a global array.
__device__ __forceinline__ void append(bool p, float v, int* ctr, float* dst) {
    unsigned m = __activemask();
    unsigned b = __ballot_sync(m, p);
    if (!b) return;
    int lane = threadIdx.x & 31;
    int leader = __ffs(b) - 1;
    int base = 0;
    if (lane == leader) base = atomicAdd(ctr, __popc(b));
    base = __shfl_sync(m, base, leader);
    if (p) dst[base + __popc(b & ((1u << lane) - 1u))] = v;
}

// Calls proc(f) on every element of src[0..n), using float4 loads where aligned.
template <typename F>
__device__ __forceinline__ void for_each_elem(const float* src, int n, F& proc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int head = (int)(((16 - ((size_t)src & 15)) & 15) >> 2);   // 0..3 scalars until aligned
    if (head > n) head = n;
    for (int i = tid; i < head; i += stride) proc(__ldg(src + i));
    const float* a = src + head;
    int m = n - head;
    int m4 = m >> 2;
    const float4* s4 = reinterpret_cast<const float4*>(a);
    for (int i = tid; i < m4; i += stride) {
        float4 v = __ldg(s4 + i);
        proc(v.x); proc(v.y); proc(v.z); proc(v.w);
    }
    for (int i = (m4 << 2) + tid; i < m; i += stride) proc(__ldg(a + i));
}

__global__ void init_kernel(State* st, unsigned* ghist, int k) {
    int t = threadIdx.x;
    if (t < 256) ghist[t] = 0;
    if (t == 0) {
        st->prefix = st->mask = st->newPrefix = st->newMask = 0;
        st->kRem = k;
        st->srcIsBuf = st->srcCount = st->curBuf = 0;
        st->doScatter = st->writeEq = 0;
        st->gtCount = st->candCount = 0;
    }
}

__global__ void hist_kernel(const float* __restrict__ in, int N,
                            const float* bufA, const float* bufB,
                            const State* __restrict__ st, unsigned* ghist, int shift) {
    __shared__ unsigned sh[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) sh[i] = 0;
    __syncthreads();

    const float* src; int n;
    if (st->srcIsBuf) { src = st->curBuf ? bufB : bufA; n = st->srcCount; }
    else              { src = in; n = N; }
    const unsigned prefix = st->prefix, mask = st->mask;

    auto proc = [&](float f) {
        unsigned key = f2k(f);
        if ((key & mask) == prefix) atomicAdd(&sh[(key >> shift) & 255u], 1u);
    };
    for_each_elem(src, n, proc);

    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        if (sh[i]) atomicAdd(&ghist[i], sh[i]);
}

__global__ void select_kernel(State* st, unsigned* ghist, int shift, int cap, int isLast) {
    if (threadIdx.x != 0) return;
    unsigned kRem = (unsigned)st->kRem;
    unsigned cum = 0;
    int d = 255;
    for (; d > 0; --d) {
        unsigned h = ghist[d];
        if (cum + h >= kRem) break;
        cum += h;
    }
    unsigned cntEq = ghist[d];
    st->kRem = (int)(kRem - cum);
    st->newMask = st->mask | (0xFFu << shift);
    st->newPrefix = st->prefix | ((unsigned)d << shift);
    st->doScatter = (isLast || st->srcIsBuf || cntEq <= (unsigned)cap) ? 1 : 0;
    st->writeEq = isLast ? 0 : 1;
    st->candCount = 0;
    for (int i = 0; i < 256; ++i) ghist[i] = 0;
}

__global__ void scatter_kernel(const float* __restrict__ in, int N,
                               float* bufA, float* bufB, State* st, float* out) {
    if (!st->doScatter) return;
    const float* src; int n;
    if (st->srcIsBuf) { src = st->curBuf ? bufB : bufA; n = st->srcCount; }
    else              { src = in; n = N; }
    float* dst = st->curBuf ? bufA : bufB;
    const unsigned np = st->newPrefix, nm = st->newMask;
    const bool writeEq = st->writeEq != 0;
    int* gtCtr = &st->gtCount;
    int* eqCtr = &st->candCount;

    auto proc = [&](float f) {
        unsigned t = f2k(f) & nm;
        append(t > np, f, gtCtr, out);
        if (writeEq) append(t == np, f, eqCtr, dst);
    };
    for_each_elem(src, n, proc);
}

__global__ void finalize_kernel(State* st) {
    if (threadIdx.x != 0) return;
    if (st->doScatter && st->writeEq) {
        st->srcIsBuf = 1;
        st->srcCount = st->candCount;
        st->curBuf ^= 1;
    }
    st->prefix = st->newPrefix;
    st->mask = st->newMask;
}

__global__ void fill_kernel(const State* __restrict__ st, float* out, int k) {
    int kRem = st->kRem;
    float T = k2f(st->prefix);
    int start = k - kRem;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kRem; i += gridDim.x * blockDim.x)
        out[start + i] = T;
}

// Bitonic sort (descending) of out[0..k) in shared memory, P = next pow2 >= k.
__global__ void small_sort_kernel(float* out, int k, int P) {
    extern __shared__ float s[];
    for (int i = threadIdx.x; i < P; i += blockDim.x) s[i] = (i < k) ? out[i] : -INFINITY;
    __syncthreads();
    for (int sz = 2; sz <= P; sz <<= 1) {
        for (int j = sz >> 1; j > 0; j >>= 1) {
            for (int t = threadIdx.x; t < (P >> 1); t += blockDim.x) {
                int i = ((t & ~(j - 1)) << 1) | (t & (j - 1));
                int l = i + j;
                float a = s[i], b = s[l];
                bool desc = (i & sz) == 0;
                if (desc ? (a < b) : (a > b)) { s[i] = b; s[l] = a; }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) out[i] = s[i];
}

// Global-memory bitonic sort pieces for large k.
__global__ void pad_copy_kernel(const float* src, float* dst, int k, int P) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < P; i += gridDim.x * blockDim.x)
        dst[i] = (i < k) ? src[i] : -INFINITY;
}
__global__ void bitonic_step_kernel(float* a, int P, int sz, int j) {
    int half = P >> 1;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < half; t += gridDim.x * blockDim.x) {
        int i = ((t & ~(j - 1)) << 1) | (t & (j - 1));
        int l = i + j;
        float x = a[i], y = a[l];
        bool desc = (i & sz) == 0;
        if (desc ? (x < y) : (x > y)) { a[i] = y; a[l] = x; }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    if (N <= 0 || k <= 0) return;
    if (k > N) k = N;

    static int numSMs = 0;
    if (!numSMs) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
        if (numSMs <= 0) numSMs = 80;
    }
    const int blocks = numSMs * 8;

    // Candidate buffers (256-byte aligned); compaction kicks in once a bucket fits in cap.
    const int cap = N < (1 << 24) ? N : (1 << 24);
    size_t bufBytes = (((size_t)cap * sizeof(float)) + 255) & ~(size_t)255;
    size_t total = 2 * bufBytes + 256 * sizeof(unsigned) + 256 + sizeof(State);
    char* mem = nullptr;
    cudaMalloc(&mem, total);
    float* bufA = reinterpret_cast<float*>(mem);
    float* bufB = reinterpret_cast<float*>(mem + bufBytes);
    unsigned* ghist = reinterpret_cast<unsigned*>(mem + 2 * bufBytes);
    State* st = reinterpret_cast<State*>(mem + 2 * bufBytes + 256 * sizeof(unsigned) + 256);

    init_kernel<<<1, 256>>>(st, ghist, k);
    for (int p = 0; p < 4; ++p) {
        int shift = 24 - 8 * p;
        int isLast = (p == 3);
        hist_kernel<<<blocks, THREADS>>>(input, N, bufA, bufB, st, ghist, shift);
        select_kernel<<<1, 32>>>(st, ghist, shift, cap, isLast);
        scatter_kernel<<<blocks, THREADS>>>(input, N, bufA, bufB, st, output);
        finalize_kernel<<<1, 32>>>(st);
    }
    int fillBlocks = (k + THREADS - 1) / THREADS;
    if (fillBlocks > blocks) fillBlocks = blocks;
    if (fillBlocks < 1) fillBlocks = 1;
    fill_kernel<<<fillBlocks, THREADS>>>(st, output, k);

    int P = 1;
    while (P < k) P <<= 1;
    if (P <= SMALL_SORT_MAX) {
        int th = P / 2 < 1024 ? (P / 2 > 32 ? P / 2 : 32) : 1024;
        small_sort_kernel<<<1, th, P * sizeof(float)>>>(output, k, P);
    } else {
        float* tmp = nullptr;
        cudaMalloc(&tmp, (size_t)P * sizeof(float));
        pad_copy_kernel<<<blocks, THREADS>>>(output, tmp, k, P);
        for (int sz = 2; sz <= P; sz <<= 1)
            for (int j = sz >> 1; j > 0; j >>= 1)
                bitonic_step_kernel<<<blocks, THREADS>>>(tmp, P, sz, j);
        cudaMemcpy(output, tmp, (size_t)k * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(tmp);
    }

    cudaDeviceSynchronize();
    cudaFree(mem);
}
