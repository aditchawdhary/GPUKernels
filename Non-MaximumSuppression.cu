/*
Non-Maximum Suppression (greedy)
boxes (N, 4) = (x1, y1, x2, y2), scores (N), keep (N) int32 (1 = kept). Spec from the
reference implementation (leetgpu-challenges #117), exact match required:
  visit boxes by decreasing score (ties: smaller index first); a box not yet suppressed is
  kept and suppresses every later box with IoU > iou_threshold (strict).
Constraints: N ≤ 20,000, integer-valued coordinates in [-10000, 10000].
Performance at N = 16,384, threshold 0.5.
*/

/*
Approach: the classic bitmask GPU NMS (as in torchvision / Detectron), with the IoU
evaluated in the reference's exact float32 operation order. Areas reach 4e8, beyond
float's exact-integer range, so the rounding has to match: every op uses an _rn
intrinsic so nothing is fused.

 1. rank_scores  rank_i = #{j : s_j > s_i or (s_j == s_i and j < i)}, with the scores
                 tiled through shared memory (N^2 compares, about 0.27 G at N = 16K).
                 order[rank_i] = i gives exactly the stable descending argsort.
 2. gather       sorted boxes (float4) and areas = (x2 - x1) * (y2 - y1).
 3. nms_mask     grid (NW x NW) of 64-thread blocks, NW = ceil(N/64). Block (rb, cb) with
                 cb ≥ rb: thread t computes the 64-bit word for rank i = 64rb + t over ranks
                 64cb .. 64cb+63. Bit k is set iff rank j > i and IoU(i, j) > thr, where
                   inter = max(min(x2) - max(x1), 0) * max(min(y2) - max(y1), 0)
                   IoU   = inter / ((area_i + area_j) - inter)
                 Blocks below the diagonal exit at once (those words are never read).
 4. nms_sweep    one 1024-thread block, 64 ranks at a time. The removed-bitset lives in
                 shared memory. For block b:
                   - 64 threads prefetch the diagonal words mask[64b + t][b] into shared
                     memory
                   - thread 0 resolves the block serially in shared memory: rank r is
                     kept if its bit is not removed, then cur |= diag[r]
                   - every thread ORs the kept rows' words into the removed words after
                     b (one word per thread, coalesced; the loads are independent, so
                     they pipeline)
                   - keep[order[r]] is written for the block
                 That's NW (256) rounds of 3 barriers, about 1 ms or less.
Scratch: order, sorted boxes and areas, plus the N x NW 64-bit mask (32 MB at N = 16K).
*/

#include <cuda_runtime.h>
#include <stdint.h>

#define RT 256
#define ST 1024
#define MAXNW ((20000 + 63) / 64)        // 313

// ---------------------------------------------------------------- 1. stable descending rank
__global__ void __launch_bounds__(RT)
rank_scores(const float* __restrict__ scores, int* __restrict__ order, int N)
{
    __shared__ float sc[RT];
    const int i = blockIdx.x * RT + threadIdx.x;
    const float mine = (i < N) ? scores[i] : 0.f;
    int rank = 0;
    for (int base = 0; base < N; base += RT) {
        const int j = base + threadIdx.x;
        sc[threadIdx.x] = (j < N) ? scores[j] : 0.f;
        __syncthreads();
        const int n = min(RT, N - base);
        for (int t = 0; t < n; ++t) {
            const float o = sc[t];
            rank += (o > mine) || (o == mine && base + t < i);
        }
        __syncthreads();
    }
    if (i < N) order[rank] = i;
}

// ---------------------------------------------------------------- 2. gather sorted boxes
__global__ void gather(const float4* __restrict__ boxes, const int* __restrict__ order,
                       float4* __restrict__ sb, float* __restrict__ area, int N)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    const float4 b = boxes[order[r]];
    sb[r] = b;
    area[r] = __fmul_rn(__fsub_rn(b.z, b.x), __fsub_rn(b.w, b.y));
}

// ---------------------------------------------------------------- 3. suppression bitmask
__global__ void __launch_bounds__(64)
nms_mask(const float4* __restrict__ sb, const float* __restrict__ area,
         unsigned long long* __restrict__ mask, int N, int NW, float thr)
{
    const int rb = blockIdx.y, cb = blockIdx.x;
    if (cb < rb) return;
    __shared__ float4 cbox[64];
    __shared__ float carea[64];
    const int t = threadIdx.x;
    const int jc = cb * 64 + t;
    if (jc < N) { cbox[t] = sb[jc]; carea[t] = area[jc]; }
    __syncthreads();

    const int i = rb * 64 + t;
    if (i >= N) return;
    const float4 a = sb[i];
    const float ai = area[i];
    const int ncol = min(64, N - cb * 64);
    unsigned long long bits = 0ull;
    for (int k = (cb == rb ? t + 1 : 0); k < ncol; ++k) {
        const float4 b = cbox[k];
        const float w = fmaxf(__fsub_rn(fminf(a.z, b.z), fmaxf(a.x, b.x)), 0.f);
        const float h = fmaxf(__fsub_rn(fminf(a.w, b.w), fmaxf(a.y, b.y)), 0.f);
        const float inter = __fmul_rn(w, h);
        const float den = __fsub_rn(__fadd_rn(ai, carea[k]), inter);
        const float iou = __fdiv_rn(inter, den);
        if (iou > thr) bits |= 1ull << k;
    }
    mask[(size_t)i * NW + cb] = bits;
}

// ---------------------------------------------------------------- 4. greedy sweep
__global__ void __launch_bounds__(ST)
nms_sweep(const unsigned long long* __restrict__ mask, const int* __restrict__ order,
          int* __restrict__ keep, int N, int NW)
{
    __shared__ unsigned long long removed[MAXNW];
    __shared__ unsigned long long diag[64];
    __shared__ unsigned long long kept;
    const int tid = threadIdx.x;
    for (int w = tid; w < NW; w += ST) removed[w] = 0ull;
    __syncthreads();

    for (int b = 0; b < NW; ++b) {
        const int r0 = b * 64;
        const int n = min(64, N - r0);
        if (tid < n) diag[tid] = mask[(size_t)(r0 + tid) * NW + b];
        __syncthreads();
        if (tid == 0) {
            unsigned long long cur = removed[b], kb = 0ull;
            for (int t = 0; t < n; ++t) {
                if (!((cur >> t) & 1ull)) { kb |= 1ull << t; cur |= diag[t]; }
            }
            removed[b] = cur;
            kept = kb;
        }
        __syncthreads();
        const unsigned long long kb = kept;
        for (int w = b + 1 + tid; w < NW; w += ST) {
            unsigned long long v = removed[w];
            unsigned long long m = kb;
            while (m) {
                const int t = __ffsll((long long)m) - 1;
                m &= m - 1;
                v |= mask[(size_t)(r0 + t) * NW + w];
            }
            removed[w] = v;
        }
        if (tid < n) keep[order[r0 + tid]] = (int)((kb >> tid) & 1ull);
        __syncthreads();
    }
}

// ---------------------------------------------------------------- host
static void*  g_ws = nullptr;
static size_t g_bytes = 0;

// boxes, scores, keep are device pointers
extern "C" void solve(const float* boxes, const float* scores, int* keep, int N,
                      float iou_threshold)
{
    if (N <= 0) return;
    const int NW = (N + 63) / 64;
    const size_t maskBytes = (((size_t)N * NW * sizeof(unsigned long long)) + 15) & ~(size_t)15;
    const size_t need = maskBytes + (size_t)N * (sizeof(float4) + sizeof(float) + sizeof(int)) + 64;
    if (need > g_bytes) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, need);
        g_bytes = need;
    }
    unsigned long long* mask = reinterpret_cast<unsigned long long*>(g_ws);
    float4* sb  = reinterpret_cast<float4*>(reinterpret_cast<char*>(g_ws) + maskBytes);   // 16 B aligned
    float* area = reinterpret_cast<float*>(sb + N);
    int* order  = reinterpret_cast<int*>(area + N);

    rank_scores<<<(N + RT - 1) / RT, RT>>>(scores, order, N);
    gather<<<(N + 255) / 256, 256>>>(reinterpret_cast<const float4*>(boxes), order, sb, area, N);
    nms_mask<<<dim3(NW, NW), 64>>>(sb, area, mask, N, NW, iou_threshold);
    nms_sweep<<<1, ST>>>(mask, order, keep, N, NW);
    cudaDeviceSynchronize();
}
