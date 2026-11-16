/*
Beam Search Step
cand[b, k, v] = beam_scores[b, k] + token_logprobs[b, k, v]
For each batch b, pick the top K of the K x V candidates, sorted by descending score:
  new_beam_scores[b, i], parent_beam_indices[b, i] = k, next_tokens[b, i] = v

Constraints
 - 1 ≤ B ≤ 64, 1 ≤ K ≤ 32, 1 ≤ V ≤ 100,000
 - All K x V candidate scores within a batch are distinct
 - Performance is measured with B = 16, K = 8, V = 50,000
*/

/*
Approach: two-stage top-K selection.

The candidates of batch b are the flat range f = k*V + v of token_logprobs[b],
which is contiguous in memory.

 1. chunk_topk   grid (chunks, B). Each block takes 4096 candidates
                 (256 threads x 16, coalesced loads), adds beam_scores[k], and
                 keeps them in registers along with each thread's best. It then
                 runs K selection rounds: a warp-shuffle + shared-memory argmax over
                 the per-thread bests picks the block maximum, and only the thread
                 that owns it marks it removed and rescans its 16 values. Result:
                 the chunk's top K (score, flat index), already in descending order.
                 At the benchmark: 98 chunks x 16 batches = 1568 blocks, and the
                 400K candidates per batch are read exactly once.
 2. merge_topk   One block per batch runs the same K-round selection over the
                 chunks*K partial results (784 at the benchmark). Winners are
                 removed in place, and each round writes output slot i directly, so
                 the outputs come out sorted. parent = f / V, token = f % V.

Missing entries (a chunk with fewer than K candidates) use index -1 and are never
selected. Ties, which the test data doesn't have, go to the lower flat index.
The scratch buffer is allocated once and reused across calls.
*/

#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256
#define ITEMS   16
#define CHUNK   (THREADS * ITEMS)        // 4096 candidates per block
#define NWARPS  (THREADS / 32)

struct Cand { float s; int i; };         // i < 0 -> invalid

__device__ __forceinline__ bool better(Cand a, Cand b)
{
    if (a.i < 0) return false;
    if (b.i < 0) return true;
    return a.s > b.s || (a.s == b.s && a.i < b.i);
}

__device__ __forceinline__ Cand shfl_cand(Cand c, int off)
{
    Cand o;
    o.s = __shfl_xor_sync(0xffffffff, c.s, off);
    o.i = __shfl_xor_sync(0xffffffff, c.i, off);
    return o;
}

// Block-wide argmax; every thread gets the winner.
__device__ __forceinline__ Cand block_argmax(Cand c)
{
    __shared__ Cand ws[NWARPS];
    __shared__ Cand win;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        Cand o = shfl_cand(c, off);
        if (better(o, c)) c = o;
    }
    if (lane == 0) ws[warp] = c;
    __syncthreads();
    if (warp == 0) {
        c = (lane < NWARPS) ? ws[lane] : Cand{-INFINITY, -1};
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            Cand o = shfl_cand(c, off);
            if (better(o, c)) c = o;
        }
        if (lane == 0) win = c;
    }
    __syncthreads();
    Cand w = win;
    __syncthreads();                     // ws / win reused next round
    return w;
}

// ---------------------------------------------------------------- stage 1
__global__ void __launch_bounds__(THREADS)
chunk_topk(const float* __restrict__ beam_scores, const float* __restrict__ logp,
           Cand* __restrict__ partial, int K, int V, int nChunks)
{
    const int c = blockIdx.x, b = blockIdx.y;
    const int total = K * V;
    const int start = c * CHUNK;
    const float* lp = logp + (size_t)b * total;
    const float* bs = beam_scores + (size_t)b * K;

    float s[ITEMS];
    unsigned live = 0;                   // bit j set -> item j present and not yet taken
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int f = start + j * THREADS + threadIdx.x;
        s[j] = -INFINITY;
        if (f < total) {
            s[j] = __ldg(bs + f / V) + __ldg(lp + f);
            live |= 1u << j;
        }
    }

    auto local_best = [&]() {
        Cand best{-INFINITY, -1};
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            if (live & (1u << j)) {
                Cand x{s[j], start + j * THREADS + (int)threadIdx.x};
                if (better(x, best)) best = x;
            }
        }
        return best;
    };

    Cand mine = local_best();
    Cand* out = partial + ((size_t)b * nChunks + c) * K;

    for (int r = 0; r < K; ++r) {
        const Cand w = block_argmax(mine);
        if (threadIdx.x == 0) out[r] = w;
        if (w.i >= 0 && w.i == mine.i) {             // I own the winner
            const int j = (w.i - start) / THREADS;
            live &= ~(1u << j);
            mine = local_best();
        }
    }
}

// ---------------------------------------------------------------- stage 2
__global__ void __launch_bounds__(THREADS)
merge_topk(Cand* __restrict__ partial, float* __restrict__ new_scores,
           int* __restrict__ parents, int* __restrict__ tokens,
           int K, int V, int nChunks)
{
    const int b = blockIdx.x;
    const int n = nChunks * K;
    Cand* p = partial + (size_t)b * n;

    auto local_best = [&]() {
        Cand best{-INFINITY, -1};
        for (int i = threadIdx.x; i < n; i += THREADS) {
            const Cand x = p[i];
            if (better(x, best)) best = x;
        }
        return best;
    };

    Cand mine = local_best();
    for (int r = 0; r < K; ++r) {
        const Cand w = block_argmax(mine);
        if (threadIdx.x == 0) {
            const size_t o = (size_t)b * K + r;
            new_scores[o] = w.s;
            parents[o] = (w.i >= 0) ? w.i / V : 0;
            tokens[o]  = (w.i >= 0) ? w.i % V : 0;
        }
        if (w.i >= 0 && w.i == mine.i) {             // remove winner from my slots
            for (int i = threadIdx.x; i < n; i += THREADS)
                if (p[i].i == w.i) { p[i].i = -1; break; }
            mine = local_best();
        }
    }
}

// ---------------------------------------------------------------- host
static Cand*  g_partial = nullptr;
static size_t g_cap = 0;

// beam_scores, token_logprobs, new_beam_scores, parent_beam_indices, next_tokens are device pointers
extern "C" void solve(const float* beam_scores, const float* token_logprobs,
                      float* new_beam_scores, int* parent_beam_indices, int* next_tokens,
                      int B, int K, int V)
{
    if (B <= 0 || K <= 0 || V <= 0) return;
    const int total   = K * V;
    const int nChunks = (total + CHUNK - 1) / CHUNK;
    const size_t need = (size_t)B * nChunks * K;

    if (need > g_cap) {
        if (g_partial) cudaFree(g_partial);
        cudaMalloc(&g_partial, need * sizeof(Cand));
        g_cap = need;
    }

    chunk_topk<<<dim3(nChunks, B), THREADS>>>(beam_scores, token_logprobs, g_partial, K, V, nChunks);
    merge_topk<<<B, THREADS>>>(g_partial, new_beam_scores, parent_beam_indices, next_tokens,
                               K, V, nChunks);
    cudaDeviceSynchronize();
}
