/*
MoE Token Dispatch
x: (T, D) float32, expert_idx: (T,) int32 in [0, E-1].
dispatched_x: (E, capacity = T, D). Expert e's slice gets its tokens in ascending
token order; token_counts[e] = number of tokens routed to e. Slots past the count
may stay zero.

Constraints
 - 1 ≤ T ≤ 65,536, 1 ≤ D ≤ 2,048, 2 ≤ E ≤ 64
 - Performance is measured with T = 16,384, D = 512, E = 8
*/

/*
Approach: a stable counting sort of token ids by expert, then a row scatter.

 1. compute_slots  One 1024-thread block walks the tokens in rounds of 1024.
                   For each token, __match_any_sync finds the lanes in the same
                   warp with the same expert, and popc(peers & lanes_below) is
                   the token's rank inside its warp. One leader per (warp, expert)
                   writes the group size to smem. E threads then turn the
                   per-warp counts into exclusive offsets, in warp order, on top
                   of each expert's running total from earlier rounds. So
                     slot(t) = running[e] + warp offset + rank,
                   which preserves token order. The kernel writes
                   dest[t] = e * T + slot(t) and, at the end, token_counts.
                   This only reads expert_idx (64 KB at most), so one block is
                   plenty.
 2. scatter_rows   One warp per token copies its D floats from x[t] to
                   dispatched_x[dest[t]], using float4 when D % 4 == 0. This is
                   the bandwidth-bound part: T*D*4 bytes read and written once
                   each (32 MB + 32 MB at the benchmark).

The scratch dest[] is a static device array (65,536 ints), so there's no allocation.
The unused tail of each expert's slice is not touched, since the problem
allows it to stay zero. Requires sm_70+ for __match_any_sync.
*/

#include <cuda_runtime.h>

#define SLOT_THREADS 1024
#define SLOT_WARPS   (SLOT_THREADS / 32)
#define MAX_E        64
#define MAX_T        65536
#define COPY_THREADS 256

__device__ int g_dest[MAX_T];

// ---------------------------------------------------------------- pass 1
__global__ void __launch_bounds__(SLOT_THREADS)
compute_slots(const int* __restrict__ expert_idx, int* __restrict__ token_counts, int T, int E)
{
    __shared__ int wc[SLOT_WARPS][MAX_E];   // per-warp counts -> per-warp offsets
    __shared__ int running[MAX_E];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned lt = (1u << lane) - 1u;
    if (tid < MAX_E) running[tid] = 0;

    for (int base = 0; base < T; base += SLOT_THREADS) {
        for (int i = tid; i < SLOT_WARPS * MAX_E; i += SLOT_THREADS) (&wc[0][0])[i] = 0;
        __syncthreads();

        const int t = base + tid;
        const int e = (t < T) ? __ldg(expert_idx + t) : -1;
        const unsigned peers = __match_any_sync(0xffffffff, e);
        const int rank = __popc(peers & lt);
        if (e >= 0 && rank == 0) wc[warp][e] = __popc(peers);   // group leader
        __syncthreads();

        if (tid < E) {                                          // scan over warps, per expert
            int run = running[tid];
            for (int w = 0; w < SLOT_WARPS; ++w) {
                const int c = wc[w][tid];
                wc[w][tid] = run;
                run += c;
            }
            running[tid] = run;
        }
        __syncthreads();

        if (e >= 0) g_dest[t] = e * T + wc[warp][e] + rank;
        __syncthreads();                                        // wc reused next round
    }

    if (tid < E) token_counts[tid] = running[tid];
}

// ---------------------------------------------------------------- pass 2
__global__ void __launch_bounds__(COPY_THREADS)
scatter_rows(const float* __restrict__ x, float* __restrict__ out, int T, int D, bool vec)
{
    const int t = blockIdx.x * (COPY_THREADS / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (t >= T) return;

    const size_t src = (size_t)t * D;
    const size_t dst = (size_t)g_dest[t] * D;

    if (vec) {
        const float4* s4 = reinterpret_cast<const float4*>(x + src);
        float4* d4 = reinterpret_cast<float4*>(out + dst);
        const int D4 = D >> 2;
        for (int i = lane; i < D4; i += 32) d4[i] = __ldg(s4 + i);
    } else {
        for (int i = lane; i < D; i += 32) out[dst + i] = __ldg(x + src + i);
    }
}

// ---------------------------------------------------------------- host
// x, expert_idx, dispatched_x, token_counts are device pointers
extern "C" void solve(const float* x, const int* expert_idx, float* dispatched_x,
                      int* token_counts, int T, int D, int E)
{
    if (T <= 0 || D <= 0 || E <= 0) return;

    compute_slots<<<1, SLOT_THREADS>>>(expert_idx, token_counts, T, E);

    const int rowsPerBlock = COPY_THREADS / 32;
    const int blocks = (T + rowsPerBlock - 1) / rowsPerBlock;
    scatter_rows<<<blocks, COPY_THREADS>>>(x, dispatched_x, T, D, (D % 4) == 0);
    cudaDeviceSynchronize();
}
