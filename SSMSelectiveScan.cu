/*
SSM Selective Scan (Mamba forward)
u, delta, y : (batch, L, D)    A : (D, N)    B, C : (batch, L, N)    skip : (D)
For each (b, d), with h in R^N starting at 0:
  h[n] = exp(delta[b,t,d] * A[d,n]) * h[n] + delta[b,t,d] * B[b,t,n] * u[b,t,d]
  y[b,t,d] = sum_n C[b,t,n] * h[n] + skip[d] * u[b,t,d]
(This matches the worked example: B_bar = delta * B, A_bar = exp(delta * A).)

Constraints
 - batch ≤ 16, L ≤ 8192, D ≤ 2048, N ≤ 64; delta > 0, A < 0
 - Performance is measured with batch = 4, L = 4096, D = 512, N = 16
*/

/*
Approach: Mamba-style. One thread owns one channel (b, d) and keeps all N state
values in registers. The sequence is split into S chunks so there's enough
parallelism, and the chunks are linked by a small sequential fix-up.

Within a chunk the recurrence is affine in the incoming state:
  h_end = P (.) h_in + h_loc,   P[n] = prod_t exp(delta_t A[n]) = exp(A[n] * sum_t delta_t)

 1. chunk_summary  grid (D/128, S, batch). Each thread runs its chunk from h = 0 and
                   writes h_loc[b][c][n][d] and sum(delta)[b][c][d].
 2. chunk_fold     One thread per (b, n, d) walks the S chunks in order and replaces
                   h_loc with h_in, the state entering each chunk:
                   h_in(c+1) = exp(A * sumdelta(c)) * h_in(c) + h_loc(c).
 3. chunk_output   Same grid as pass 1. Each thread restarts its chunk from h_in and
                   produces y = C . h + skip * u.

 - Loads of u, delta and y are coalesced across d. B[b,t,:] and C[b,t,:] are the
   same for every thread in the block, so they become broadcast loads served from
   L1. Each step has N independent FMA chains plus N expf, which is plenty of ILP.
 - N is a template bound (4, 8, 16, 32 or 64) with a runtime guard, so h and A stay
   in registers.
 - S is chosen so batch*D*S ≈ 128K threads (S = 64 chunks of 64 steps at the
   benchmark), capped so each chunk has at least 32 steps. When S = 1, passes 1
   and 2 are skipped.
 - Scratch (h_loc/h_in plus sum(delta)) is allocated once, grown when needed and
   reused; it stays under about 45 MB for any allowed shape.
*/

#include <cuda_runtime.h>
#include <math.h>

#define THREADS 128

// ---------------------------------------------------------------- pass 1
template <int NMAX>
__global__ void __launch_bounds__(THREADS)
chunk_summary(const float* __restrict__ u, const float* __restrict__ delta,
              const float* __restrict__ A, const float* __restrict__ Bm,
              float* __restrict__ hbuf, float* __restrict__ sdel,
              int L, int D, int N, int S, int TC)
{
    const int d = blockIdx.x * THREADS + threadIdx.x;
    if (d >= D) return;
    const int c = blockIdx.y, b = blockIdx.z;
    const int t0 = c * TC, t1 = min(t0 + TC, L);

    float a[NMAX], h[NMAX];
    #pragma unroll
    for (int n = 0; n < NMAX; ++n) { a[n] = (n < N) ? __ldg(A + (size_t)d * N + n) : 0.f; h[n] = 0.f; }

    float sd = 0.f;
    for (int t = t0; t < t1; ++t) {
        const size_t i = ((size_t)b * L + t) * D + d;
        const float dt = __ldg(delta + i);
        const float du = dt * __ldg(u + i);
        const float* Bt = Bm + ((size_t)b * L + t) * N;
        sd += dt;
        #pragma unroll
        for (int n = 0; n < NMAX; ++n)
            if (n < N) h[n] = fmaf(expf(dt * a[n]), h[n], du * __ldg(Bt + n));
    }

    #pragma unroll
    for (int n = 0; n < NMAX; ++n)
        if (n < N) hbuf[(((size_t)b * S + c) * N + n) * D + d] = h[n];
    sdel[((size_t)b * S + c) * D + d] = sd;
}

// ---------------------------------------------------------------- pass 2
__global__ void chunk_fold(const float* __restrict__ A, float* __restrict__ hbuf,
                           const float* __restrict__ sdel, int batch, int D, int N, int S)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)batch * N * D;
    if (idx >= total) return;
    const int d = (int)(idx % D);
    const int n = (int)((idx / D) % N);
    const int b = (int)(idx / ((size_t)D * N));
    const float a = __ldg(A + (size_t)d * N + n);

    float hin = 0.f;
    for (int c = 0; c < S; ++c) {
        float* p = hbuf + (((size_t)b * S + c) * N + n) * D + d;
        const float hloc = *p;
        const float sd = sdel[((size_t)b * S + c) * D + d];
        *p = hin;                                       // state entering chunk c
        hin = fmaf(expf(a * sd), hin, hloc);
    }
}

// ---------------------------------------------------------------- pass 3
template <int NMAX>
__global__ void __launch_bounds__(THREADS)
chunk_output(const float* __restrict__ u, const float* __restrict__ delta,
             const float* __restrict__ A, const float* __restrict__ Bm,
             const float* __restrict__ Cm, const float* __restrict__ skip,
             const float* __restrict__ hbuf, float* __restrict__ y,
             int L, int D, int N, int S, int TC)
{
    const int d = blockIdx.x * THREADS + threadIdx.x;
    if (d >= D) return;
    const int c = blockIdx.y, b = blockIdx.z;
    const int t0 = c * TC, t1 = min(t0 + TC, L);

    float a[NMAX], h[NMAX];
    #pragma unroll
    for (int n = 0; n < NMAX; ++n) {
        a[n] = (n < N) ? __ldg(A + (size_t)d * N + n) : 0.f;
        h[n] = (S > 1 && n < N) ? hbuf[(((size_t)b * S + c) * N + n) * D + d] : 0.f;
    }
    const float sk = __ldg(skip + d);

    for (int t = t0; t < t1; ++t) {
        const size_t i = ((size_t)b * L + t) * D + d;
        const float dt = __ldg(delta + i);
        const float ut = __ldg(u + i);
        const float du = dt * ut;
        const float* Bt = Bm + ((size_t)b * L + t) * N;
        const float* Ct = Cm + ((size_t)b * L + t) * N;
        float acc = sk * ut;
        #pragma unroll
        for (int n = 0; n < NMAX; ++n) {
            if (n < N) {
                h[n] = fmaf(expf(dt * a[n]), h[n], du * __ldg(Bt + n));
                acc = fmaf(__ldg(Ct + n), h[n], acc);
            }
        }
        y[i] = acc;
    }
}

// ---------------------------------------------------------------- host
static float* g_scratch = nullptr;
static size_t g_cap = 0;

template <int NMAX>
static void run(const float* u, const float* delta, const float* A, const float* B,
                const float* C, const float* skip, float* y,
                int batch, int L, int D, int N, int S, int TC, float* hbuf, float* sdel)
{
    dim3 grid((D + THREADS - 1) / THREADS, S, batch);
    if (S > 1) {
        chunk_summary<NMAX><<<grid, THREADS>>>(u, delta, A, B, hbuf, sdel, L, D, N, S, TC);
        const size_t total = (size_t)batch * N * D;
        chunk_fold<<<(unsigned)((total + 255) / 256), 256>>>(A, hbuf, sdel, batch, D, N, S);
    }
    chunk_output<NMAX><<<grid, THREADS>>>(u, delta, A, B, C, skip, hbuf, y, L, D, N, S, TC);
}

// u, delta, A, B, C, skip, y are device pointers
extern "C" void solve(const float* u, const float* delta, const float* A, const float* B,
                      const float* C, const float* skip, float* y, int batch, int seq_len,
                      int d_model, int d_state)
{
    const int L = seq_len, D = d_model, N = d_state;
    if (batch <= 0 || L <= 0 || D <= 0 || N <= 0) return;

    // chunk count: aim for ~128K threads, but keep chunks ≥ 32 steps
    const long long chans = (long long)batch * D;
    int S = (int)((131072 + chans - 1) / chans);
    S = max(1, min(S, (L + 31) / 32));
    const int TC = (L + S - 1) / S;
    S = (L + TC - 1) / TC;

    float* hbuf = nullptr;
    float* sdel = nullptr;
    if (S > 1) {
        const size_t hN = (size_t)batch * S * N * D;
        const size_t sN = (size_t)batch * S * D;
        const size_t need = hN + sN;
        if (need > g_cap) {
            if (g_scratch) cudaFree(g_scratch);
            cudaMalloc(&g_scratch, need * sizeof(float));
            g_cap = need;
        }
        hbuf = g_scratch;
        sdel = g_scratch + hN;
    }

    if      (N <= 4)  run<4> (u, delta, A, B, C, skip, y, batch, L, D, N, S, TC, hbuf, sdel);
    else if (N <= 8)  run<8> (u, delta, A, B, C, skip, y, batch, L, D, N, S, TC, hbuf, sdel);
    else if (N <= 16) run<16>(u, delta, A, B, C, skip, y, batch, L, D, N, S, TC, hbuf, sdel);
    else if (N <= 32) run<32>(u, delta, A, B, C, skip, y, batch, L, D, N, S, TC, hbuf, sdel);
    else              run<64>(u, delta, A, B, C, skip, y, batch, L, D, N, S, TC, hbuf, sdel);
    cudaDeviceSynchronize();
}
