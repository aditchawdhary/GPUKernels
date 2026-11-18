#include <cuda_runtime.h>
#include <limits.h>

// Fused logit penalties (vLLM semantics):
//   seen[v]  = v appears in prompt_tokens[b] or output_tokens[b]
//   count[v] = number of times v appears in output_tokens[b]
//   x = logits[b][v]
//   if seen[v]:     x = (x > 0) ? x / repetition : x * repetition   (applied first)
//   x -= frequency * count[v]
//   if count[v]>0:  x -= presence
//
// The work is dominated by streaming B*V logits (~131 MB at the benchmark size),
// while at most P+G <= 1536 entries per row change. So:
//   Kernel 1: bandwidth-bound vectorized copy logits -> output.
//   Kernel 2: one block per row sorts that row's (token, source) keys in shared
//             memory, and the first slot of each distinct token applies the
//             penalties to that single output element. Each token is written by
//             exactly one thread, so no atomics or scratch arrays are needed.

#define SORT_N   2048          // next pow2 >= P + G (1024 + 512)
#define FIX_THREADS 1024

__global__ void copy_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t n, bool vec)
{
    size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    if (vec) {
        const size_t n4 = n >> 2;
        const float4* s4 = reinterpret_cast<const float4*>(src);
        float4*       d4 = reinterpret_cast<float4*>(dst);
        for (size_t i = tid; i < n4; i += stride) d4[i] = __ldcs(s4 + i);   // streaming: read once
        for (size_t i = (n4 << 2) + tid; i < n; i += stride) dst[i] = src[i];
    } else {
        for (size_t i = tid; i < n; i += stride) dst[i] = src[i];
    }
}

__global__ void __launch_bounds__(FIX_THREADS)
penalty_kernel(const float* __restrict__ logits, const int* __restrict__ prompt_tokens,
               const int* __restrict__ output_tokens, const float* __restrict__ presence_penalty,
               const float* __restrict__ frequency_penalty, const float* __restrict__ repetition_penalty,
               float* __restrict__ output, int V, int P, int G)
{
    __shared__ int keys[SORT_N];
    const int b = blockIdx.x;
    const int total = P + G;

    // key = (token << 1) | fromOutput ; padding / out-of-range -> INT_MAX (sorts last)
    for (int i = threadIdx.x; i < SORT_N; i += blockDim.x) {
        int k = INT_MAX;
        if (i < P) {
            int t = prompt_tokens[(size_t)b * P + i];
            if (t >= 0 && t < V) k = t << 1;
        } else if (i < total) {
            int t = output_tokens[(size_t)b * G + (i - P)];
            if (t >= 0 && t < V) k = (t << 1) | 1;
        }
        keys[i] = k;
    }
    __syncthreads();

    // Bitonic sort (ascending) over SORT_N keys in shared memory.
    for (int k = 2; k <= SORT_N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < SORT_N; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    int a = keys[i], c = keys[ixj];
                    bool up = ((i & k) == 0);
                    if ((a > c) == up) { keys[i] = c; keys[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }

    const float pres = presence_penalty[b];
    const float freq = frequency_penalty[b];
    const float rep  = repetition_penalty[b];

    // Segment heads: first slot of each distinct token.
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int k = keys[i];
        if (k == INT_MAX) continue;
        const int tok = k >> 1;
        if (i > 0 && (keys[i - 1] >> 1) == tok) continue;

        int cnt = 0;   // output occurrences (prompt keys sort before output keys)
        for (int j = i; j < total && (keys[j] >> 1) == tok && keys[j] != INT_MAX; ++j)
            cnt += keys[j] & 1;

        const size_t idx = (size_t)b * V + tok;
        float x = logits[idx];
        x = (x > 0.f) ? x / rep : x * rep;            // repetition (seen in prompt or output)
        x -= freq * (float)cnt;                        // frequency
        if (cnt > 0) x -= pres;                        // presence
        output[idx] = x;
    }
}

// logits, prompt_tokens, output_tokens, presence_penalty, frequency_penalty, repetition_penalty,
// output are device pointers
extern "C" void solve(const float* logits, const int* prompt_tokens, const int* output_tokens,
                      const float* presence_penalty, const float* frequency_penalty,
                      const float* repetition_penalty, float* output, int B, int V, int P, int G) {
    const size_t n = (size_t)B * V;
    const bool vec = ((reinterpret_cast<size_t>(logits) | reinterpret_cast<size_t>(output)) & 15) == 0;

    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    size_t work = vec ? (n >> 2) : n;
    int blocks = (int)((work + 255) / 256);
    int maxBlocks = sms * 8;
    if (blocks > maxBlocks) blocks = maxBlocks;
    if (blocks < 1) blocks = 1;
    copy_kernel<<<blocks, 256>>>(logits, output, n, vec);

    penalty_kernel<<<B, FIX_THREADS>>>(logits, prompt_tokens, output_tokens, presence_penalty,
                                       frequency_penalty, repetition_penalty, output, V, P, G);
    cudaDeviceSynchronize();
}
