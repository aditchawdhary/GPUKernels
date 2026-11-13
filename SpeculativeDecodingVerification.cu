#include <cuda_runtime.h>

// Speculative decoding verification (one block per sequence).
//
// For sequence b: walk i = 0..T-1, accept draft token tok if u[b,i] < min(1, q/p) with
// p = draft[b,i,tok], q = target[b,i,tok]. At the first rejection i, resample from
//   adj = max(target[b,i,:] - draft[b,i,:], 0) / sum(adj)    (uniform if sum(adj) == 0)
// using r = u[b,T], write it at position i and stop (later positions stay 0). If everything
// is accepted, sample the bonus token from target[b,T-1,:] with r = u[b,T] at position T.
// Sampling = searchsorted(cdf, r): the first k with cdf[k] >= r, clamped to V-1.
//
// Sampling kernel structure (one row of V values, 1024 threads = 32 warps):
//   pass 1: warp w sums its contiguous segment of the row (coalesced) -> 32 segment sums;
//           thread 0 scans them, gets the total, and finds the segment where the CDF crosses
//           the threshold (r * total for adj, r for the bonus row).
//   pass 2: that one warp rescans its segment 32 values at a time with a shuffle scan and
//           a ballot to find the exact first index.
// Sums are in double so the crossing point is not disturbed by fp32 summation-order noise.

#define THREADS 1024
#define NWARPS (THREADS / 32)

__device__ __forceinline__ float row_val(const float* __restrict__ t, const float* __restrict__ d,
                                         bool adjusted, int v) {
    float tv = __ldg(&t[v]);
    return adjusted ? fmaxf(tv - __ldg(&d[v]), 0.0f) : tv;
}

__global__ void __launch_bounds__(THREADS)
spec_verify(const int* __restrict__ draft_tokens, const float* __restrict__ draft_probs,
            const float* __restrict__ target_probs, const float* __restrict__ uniform_samples,
            int* __restrict__ output_tokens, int T, int V) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    __shared__ int s_reject;          // first rejected position, or -1
    __shared__ double s_wsum[NWARPS];
    __shared__ double s_carry;        // CDF mass before the crossing segment
    __shared__ double s_thr;
    __shared__ int s_target_warp;     // -1: answer already decided by thread 0
    __shared__ int s_answer;

    const size_t row_base = (size_t)b * T * V;
    const float* us = uniform_samples + (size_t)b * (T + 1);

    if (tid == 0) {
        int rej = -1;
        for (int i = 0; i < T; i++) {
            int tok = draft_tokens[b * T + i];
            double p = draft_probs[row_base + (size_t)i * V + tok];
            double q = target_probs[row_base + (size_t)i * V + tok];
            double alpha = fmin(1.0, q / p);
            if (!((double)us[i] < alpha)) { rej = i; break; }
        }
        s_reject = rej;
    }
    __syncthreads();

    const int rej = s_reject;
    const bool adjusted = rej >= 0;
    const int pos = adjusted ? rej : T - 1;
    const float* trow = target_probs + row_base + (size_t)pos * V;
    const float* drow = draft_probs + row_base + (size_t)pos * V;
    const double r = (double)us[T];

    // ---- pass 1: per-warp segment sums ----
    const int seg = (V + NWARPS - 1) / NWARPS;
    const int seg_start = min(V, warp * seg), seg_end = min(V, seg_start + seg);
    double s = 0.0;
    for (int v = seg_start + lane; v < seg_end; v += 32) s += (double)row_val(trow, drow, adjusted, v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) s_wsum[warp] = s;
    __syncthreads();

    if (tid == 0) {
        double total = 0.0;
        for (int w = 0; w < NWARPS; w++) total += s_wsum[w];
        s_target_warp = -1;
        if (adjusted && total <= 0.0) {
            // uniform fallback: cdf[k] = (k+1)/V, first k with cdf[k] >= r
            long long k = (long long)ceil(r * (double)V) - 1;
            s_answer = (int)min((long long)V - 1, max(0LL, k));
        } else {
            double thr = adjusted ? r * total : r;
            s_thr = thr;
            if (thr <= 0.0) {
                s_answer = 0;                       // cdf[0] >= 0 always
            } else {
                double excl = 0.0;
                int w = 0;
                for (; w < NWARPS; w++) {
                    if (excl + s_wsum[w] >= thr) break;
                    excl += s_wsum[w];
                }
                if (w == NWARPS) {
                    s_answer = V - 1;               // r beyond the final cdf value -> clamp
                } else {
                    s_target_warp = w;
                    s_carry = excl;
                }
            }
        }
    }
    __syncthreads();

    // ---- pass 2: exact index inside the crossing segment ----
    if (warp == s_target_warp) {
        const double thr = s_thr;
        double carry = s_carry;
        int ans = seg_end - 1;
        for (int base = seg_start; base < seg_end; base += 32) {
            int v = base + lane;
            double x = v < seg_end ? (double)row_val(trow, drow, adjusted, v) : 0.0;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                double y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            unsigned hit = __ballot_sync(0xffffffffu, v < seg_end && carry + x >= thr);
            if (hit) { ans = base + __ffs(hit) - 1; break; }
            carry += __shfl_sync(0xffffffffu, x, 31);
        }
        if (lane == 0) s_answer = ans;
    }
    __syncthreads();

    if (tid == 0) {
        int* out = output_tokens + (size_t)b * (T + 1);
        for (int i = 0; i <= T; i++) out[i] = 0;
        int last = adjusted ? rej : T;
        for (int i = 0; i < last; i++) out[i] = draft_tokens[b * T + i];
        out[last] = s_answer;
    }
}

// draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) {
    spec_verify<<<B, THREADS>>>(draft_tokens, draft_probs, target_probs, uniform_samples,
                                output_tokens, T, V);
}
