/*
Top-p Sampling

Write a GPU program that implements top-p (nucleus) sampling for LLM inference.

Top-p sampling is a text generation technique where you sample from the smallest set of tokens whose cumulative probability exceeds threshold p. This balances randomness and quality better than pure top-k or greedy sampling.

Given logits (unnormalized scores) from a language model:

 1. Convert logits to probabilities using softmax
 2. Sort tokens by probability (descending); tokens with equal probability are ordered by increasing index
 3. Find the smallest set where cumulative probability ≥ p (the "nucleus")
 4. Renormalize the nucleus probabilities to sum to 1
 5. Sample a token from the nucleus using the provided random seed, as defined below

The sample is deterministic, so every correct implementation returns the same token. Each token with index i in logits gets a uniform number u_i in (0, 1)
 from the seed and its index, using the PCG hash h on unsigned 32-bit integers. All arithmetic wraps modulo 2^32, and seed is reinterpreted as an unsigned 32-bit integer:

```
uint32 h(uint32 x) {
    uint32 state = x * 747796405u + 2891336453u;
    uint32 word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
```
u_i = (h(h(seed) + i) + 0.5) / 2^32
The sampled token is the nucleus token with the largest q_i/(-ln(u_i)), where q_i is its renormalized probability. 
This "exponential race" picks token i with probability. Renormalizing scales every score by the same factor, so it does not change which token wins.

Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - Ensure numerical stability when computing softmax

```
Example 1:
Input:
  logits = [1.0, 2.0, 3.0, 0.5]
  p = 0.9
  seed = 42

Output:
  sampled_token = 0

  The nucleus is tokens 2, 1, 0 (cumulative probability 0.948 ≥ 0.9), and h(seed) = 1223963391:
  token 2: q = 0.6652, u = 0.0049, q / -ln(u) = 0.1249
  token 1: q = 0.2447, u = 0.2550, q / -ln(u) = 0.1791
  token 0: q = 0.0900, u = 0.6485, q / -ln(u) = 0.2079  (largest)
```

```
Example 2:
Input:
  logits = [10.0, 1.0, 1.0]
  p = 0.5
  seed = 123
Output:
  sampled_token = 0
  (single token dominates the probability mass)
```

Constraints
 - 3 ≤ vocab_size ≤ 50,000
 - -100.0 ≤ logits[i] ≤ 100.0
 - 0.0 < p ≤ 1.0
 - 0 ≤ sampled_token < vocab_size
 - Performance is measured with vocab_size = 50,000
*/



#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>

#define THREADS   512
#define MAX_VOCAB 50000

typedef unsigned int uint32;

// Scratch buffer for probabilities (lives in global memory, cached in L2).
__device__ float g_probs[MAX_VOCAB];

__device__ uint32 pcg_hash(uint32 x) {
    uint32 state = x * 747796405u + 2891336453u;
    uint32 word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// ---------- block-wide reductions (every thread must call them) ----------
__device__ double block_sum(double v) {
    __shared__ double sh[THREADS / 32];
    for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffff, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    double t = 0.0;
    for (int w = 0; w < THREADS / 32; w++) t += sh[w];
    return t;
}

__device__ float block_max(float v) {
    __shared__ float sh[THREADS / 32];
    for (int o = 16; o > 0; o /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int w = 0; w < THREADS / 32; w++) t = fmaxf(t, sh[w]);
    return t;
}

// ---------- helpers that scan all probabilities ----------
// Sum of probabilities >= v
__device__ double mass_ge(float v, int V) {
    double acc = 0.0;
    for (int i = threadIdx.x; i < V; i += THREADS) {
        float q = g_probs[i];
        if (q >= v) acc += q;
    }
    return block_sum(acc);
}

// Number of tokens with probability == v and index < limit
__device__ int count_ties(float v, int limit) {
    double c = 0.0;
    for (int i = threadIdx.x; i < limit; i += THREADS)
        if (g_probs[i] == v) c += 1.0;
    return (int)block_sum(c);
}

__global__ void __launch_bounds__(THREADS)
top_p_kernel(const float* logits, const float* p_ptr, const int* seed_ptr,
             int* sampled_token, int V) {
    int tid = threadIdx.x;
    double p    = (double)(*p_ptr);
    uint32 seed = (uint32)(*seed_ptr);

    // ---- 1. Stable softmax into g_probs ----
    float mx = -INFINITY;
    for (int i = tid; i < V; i += THREADS) mx = fmaxf(mx, logits[i]);
    mx = block_max(mx);

    double s = 0.0;
    for (int i = tid; i < V; i += THREADS) s += (double)expf(logits[i] - mx);
    s = block_sum(s);

    for (int i = tid; i < V; i += THREADS)
        g_probs[i] = (float)((double)expf(logits[i] - mx) / s);
    __syncthreads();                       // probs visible to the whole block

    // ---- 2. Find cutoff v*: largest v with mass(prob >= v) >= p ----
    float vstar;
    int   idx_cut;                         // ties at v* included iff index < idx_cut

    if (mass_ge(0.0f, V) < p) {
        vstar = 0.0f;                      // rounding: total < p, keep every token
        idx_cut = V;
    } else {
        // Non-negative floats sort the same as their bit patterns.
        uint32 lo = 0;                             // mass(lo) >= p
        uint32 hi = __float_as_uint(1.0f) + 1u;    // mass(hi) <  p
        while (hi - lo > 1u) {
            uint32 mid = lo + (hi - lo) / 2u;
            if (mass_ge(__uint_as_float(mid), V) >= p) lo = mid; else hi = mid;
        }
        vstar = __uint_as_float(lo);
        double mass_above = mass_ge(__uint_as_float(hi), V);   // strictly > v*

        // ---- 3. Ties at v*: take the lowest-index ones until mass reaches p ----
        int ties = count_ties(vstar, V);
        int need = 0;
        double acc = mass_above;
        while (need < ties && acc < p) { acc += vstar; need++; }

        if (need >= ties) {
            idx_cut = V;                   // all ties included (the usual case)
        } else {
            int a = 0, b = V;              // smallest cut with need ties below it
            while (a < b) {
                int mid = (a + b) / 2;
                if (count_ties(vstar, mid) >= need) b = mid; else a = mid + 1;
            }
            idx_cut = a;
        }
    }

    // ---- 4. Exponential race over the nucleus ----
    uint32 hs = pcg_hash(seed);
    double best = -1.0;
    int    best_i = 0x7fffffff;
    for (int i = tid; i < V; i += THREADS) {
        float q = g_probs[i];
        bool in_nucleus = (q > vstar) || (q == vstar && i < idx_cut);
        if (!in_nucleus) continue;

        double u = ((double)pcg_hash(hs + (uint32)i) + 0.5) / 4294967296.0;
        double score = (double)q / -log(u);
        if (score > best || (score == best && i < best_i)) { best = score; best_i = i; }
    }

    // Block argmax (ties -> smaller index)
    for (int o = 16; o > 0; o /= 2) {
        double s2 = __shfl_xor_sync(0xffffffff, best, o);
        int    i2 = __shfl_xor_sync(0xffffffff, best_i, o);
        if (s2 > best || (s2 == best && i2 < best_i)) { best = s2; best_i = i2; }
    }
    __shared__ double sh_s[THREADS / 32];
    __shared__ int    sh_i[THREADS / 32];
    if ((tid & 31) == 0) { sh_s[tid / 32] = best; sh_i[tid / 32] = best_i; }
    __syncthreads();

    if (tid == 0) {
        double bs = -1.0; int bi = 0x7fffffff;
        for (int w = 0; w < THREADS / 32; w++)
            if (sh_s[w] > bs || (sh_s[w] == bs && sh_i[w] < bi)) { bs = sh_s[w]; bi = sh_i[w]; }
        *sampled_token = bi;
    }
}

// logits, p, seed, sampled_token are device pointers
extern "C" void solve(const float* logits, const float* p, const int* seed, int* sampled_token,
                      int vocab_size) {
    top_p_kernel<<<1, THREADS>>>(logits, p, seed, sampled_token, vocab_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("top_p launch failed: %s\n", cudaGetErrorString(err));
}
