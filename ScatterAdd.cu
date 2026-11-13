#include <cuda_runtime.h>

// Row-wise scatter-add: output[index[i], :] += src[i, :], output starts at zero.
//
// Instead of N*D float atomics into the output, invert the index into CSR form and gather:
//   1. count   : cnt[m] = #{i : index[i] == m}                   (int atomics on M counters)
//   2. scan    : off = exclusive_scan(cnt), off[M] = N           (two-level block scan)
//   3. bucket  : perm[off[m] + k] = i for the k-th i with index[i] == m
//   4. gather  : one warp per output row m sums src rows perm[off[m] .. off[m+1]) with
//                coalesced (float4 when possible) loads and writes the row exactly once
//                (zeros for rows nothing maps to), so no pre-zeroing and no float atomics.
// Sources are summed 4 at a time for memory-level parallelism.

#define SCAN_T 1024
#define GATHER_T 256                  // 8 warps = 8 output rows per block

__global__ void count_kernel(const int* __restrict__ index, int* __restrict__ cnt, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        atomicAdd(&cnt[__ldg(&index[i])], 1);
}

__device__ int block_exclusive_scan(int v, int* total) {
    __shared__ int ws[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) ws[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int s = lane < nw ? ws[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        if (lane < nw) ws[lane] = s;
    }
    __syncthreads();
    int pre = warp ? ws[warp - 1] : 0;
    *total = ws[nw - 1];
    __syncthreads();
    return pre + x - v;
}

// per-chunk exclusive scan; chunk sums to bsum
__global__ void scan_chunks(const int* __restrict__ cnt, int* __restrict__ off,
                            int* __restrict__ bsum, int M) {
    int i = blockIdx.x * SCAN_T + threadIdx.x;
    int v = i < M ? cnt[i] : 0, tot;
    int ex = block_exclusive_scan(v, &tot);
    if (i < M) off[i] = ex;
    if (threadIdx.x == 0) bsum[blockIdx.x] = tot;
}

// scan of chunk sums (nb <= SCAN_T), in place
__global__ void scan_bsums(int* __restrict__ bsum, int nb) {
    int v = threadIdx.x < nb ? bsum[threadIdx.x] : 0, tot;
    int ex = block_exclusive_scan(v, &tot);
    if (threadIdx.x < nb) bsum[threadIdx.x] = ex;
}

__global__ void add_bsums(int* __restrict__ off, const int* __restrict__ bsum, int M, int N) {
    int i = blockIdx.x * SCAN_T + threadIdx.x;
    if (i < M) off[i] += bsum[blockIdx.x];
    if (i == 0) off[M] = N;
}

__global__ void bucket_kernel(const int* __restrict__ index, const int* __restrict__ off,
                              int* __restrict__ cursor, int* __restrict__ perm, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        int m = __ldg(&index[i]);
        perm[off[m] + atomicAdd(&cursor[m], 1)] = i;
    }
}

template <bool VEC>
__global__ void __launch_bounds__(GATHER_T)
gather_kernel(const float* __restrict__ src, const int* __restrict__ off,
              const int* __restrict__ perm, float* __restrict__ out, int M, int D) {
    const int lane = threadIdx.x & 31;
    const int m = blockIdx.x * (GATHER_T / 32) + (threadIdx.x >> 5);
    if (m >= M) return;
    const int beg = off[m], end = off[m + 1];
    float* orow = out + (size_t)m * D;

    if (VEC) {
        const int D4 = D >> 2;
        for (int c = lane; c < D4; c += 32) {
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
            int p = beg;
            for (; p + 4 <= end; p += 4) {
                int i0 = perm[p], i1 = perm[p + 1], i2 = perm[p + 2], i3 = perm[p + 3];
                float4 a = __ldg(reinterpret_cast<const float4*>(src + (size_t)i0 * D) + c);
                float4 b = __ldg(reinterpret_cast<const float4*>(src + (size_t)i1 * D) + c);
                float4 e = __ldg(reinterpret_cast<const float4*>(src + (size_t)i2 * D) + c);
                float4 f = __ldg(reinterpret_cast<const float4*>(src + (size_t)i3 * D) + c);
                acc.x += a.x + b.x + e.x + f.x;
                acc.y += a.y + b.y + e.y + f.y;
                acc.z += a.z + b.z + e.z + f.z;
                acc.w += a.w + b.w + e.w + f.w;
            }
            for (; p < end; p++) {
                float4 a = __ldg(reinterpret_cast<const float4*>(src + (size_t)perm[p] * D) + c);
                acc.x += a.x; acc.y += a.y; acc.z += a.z; acc.w += a.w;
            }
            reinterpret_cast<float4*>(orow)[c] = acc;
        }
    } else {
        for (int c = lane; c < D; c += 32) {
            float acc = 0.0f;
            int p = beg;
            for (; p + 4 <= end; p += 4) {
                float a = __ldg(&src[(size_t)perm[p] * D + c]);
                float b = __ldg(&src[(size_t)perm[p + 1] * D + c]);
                float e = __ldg(&src[(size_t)perm[p + 2] * D + c]);
                float f = __ldg(&src[(size_t)perm[p + 3] * D + c]);
                acc += a + b + e + f;
            }
            for (; p < end; p++) acc += __ldg(&src[(size_t)perm[p] * D + c]);
            orow[c] = acc;
        }
    }
}

static char* g_buf = nullptr;
static size_t g_bytes = 0;

// src, index, output are device pointers
extern "C" void solve(const float* src, const int* index, float* output, int N, int D, int M) {
    const int nb = (M + SCAN_T - 1) / SCAN_T;           // <= 977 for M <= 1e6
    size_t need = ((size_t)2 * M + (M + 1) + N + nb) * sizeof(int);
    if (need > g_bytes) {
        if (g_buf) cudaFree(g_buf);
        cudaMalloc(&g_buf, need);
        g_bytes = need;
    }
    int* cnt = (int*)g_buf;           // [M]  per-row counts
    int* cursor = cnt + M;            // [M]
    int* off = cursor + M;            // [M+1]
    int* perm = off + (M + 1);        // [N]
    int* bsum = perm + N;             // [nb]

    cudaMemsetAsync(cnt, 0, (size_t)2 * M * sizeof(int));   // cnt and cursor
    int gb = (N + 255) / 256; if (gb > 4096) gb = 4096;
    count_kernel<<<gb, 256>>>(index, cnt, N);
    scan_chunks<<<nb, SCAN_T>>>(cnt, off, bsum, M);
    scan_bsums<<<1, SCAN_T>>>(bsum, nb);
    add_bsums<<<nb, SCAN_T>>>(off, bsum, M, N);
    bucket_kernel<<<gb, 256>>>(index, off, cursor, perm, N);

    bool vec = (D % 4 == 0) &&
               (((reinterpret_cast<size_t>(src) | reinterpret_cast<size_t>(output)) & 15) == 0);
    int blocks = (M + GATHER_T / 32 - 1) / (GATHER_T / 32);
    if (vec) gather_kernel<true><<<blocks, GATHER_T>>>(src, off, perm, output, M, D);
    else     gather_kernel<false><<<blocks, GATHER_T>>>(src, off, perm, output, M, D);
}
