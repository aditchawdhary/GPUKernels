#include <cuda_runtime.h>

// Kimi Delta Attention
//   S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T
//   o_t = S_t^T q_t
//
// Key observation: column j of S (S[:, j], length d) evolves independently of
// the other columns. With P = Diag(a) S:
//   x_j = k . P[:,j]     y_j = q . P[:,j]     z = q . k
//   e_j = b (v_j - x_j)
//   S[:,j] = P[:,j] + k e_j
//   o_j = q . S[:,j] = y_j + z e_j
// So each column needs ONE round of (3 independent) reductions per step.
//
// Mapping: 8 lanes own one column (16 state rows per lane, in registers).
// A warp holds 4 columns, a block of 4 warps holds 16 columns.
// Grid = (ceil(d/16), batch). k, alpha, q, v, beta are staged in shared memory
// CH timesteps at a time, with the next chunk prefetched into registers.

#define DMAX 128
#define LPC 8                              // lanes per column
#define R (DMAX / LPC)                     // 16 state rows per lane
#define COLS_PER_WARP (32 / LPC)           // 4
#define WARPS 4
#define THREADS (WARPS * 32)               // 128
#define COLS (WARPS * COLS_PER_WARP)       // 16 columns per block
#define CH 16                              // timesteps per chunk
#define LOADS (CH * DMAX / THREADS)        // 16 per thread per array
#define VLOADS (CH * COLS / THREADS)       // 2

__global__ void __launch_bounds__(THREADS)
kda_kernel(const float* __restrict__ Q, const float* __restrict__ K,
           const float* __restrict__ V, const float* __restrict__ A,
           const float* __restrict__ Bt, float* __restrict__ O,
           int seq_len, int d) {
    __shared__ float sk[CH][DMAX];
    __shared__ float sa[CH][DMAX];
    __shared__ float sq[CH][DMAX];
    __shared__ float sv[CH][COLS];
    __shared__ float so[CH][COLS];
    __shared__ float sb[CH];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int sub  = lane & (LPC - 1);                     // row-slot within column
    const int cl   = warp * COLS_PER_WARP + lane / LPC;    // local column 0..15
    const int col0 = blockIdx.x * COLS;
    const int b    = blockIdx.y;

    const size_t base = (size_t)b * seq_len * d;
    const float* Qb = Q + base;
    const float* Kb = K + base;
    const float* Vb = V + base;
    const float* Ab = A + base;
    const float* Bb = Bt + (size_t)b * seq_len;
    float*       Ob = O + base;

    // Prefetch registers for the next chunk.
    float rk[LOADS], ra[LOADS], rq[LOADS], rv[VLOADS], rb = 0.f;

    auto prefetch = [&](int t0) {
        #pragma unroll
        for (int j = 0; j < LOADS; ++j) {
            int idx = tid + j * THREADS;
            int tt = idx / DMAX, i = idx % DMAX;
            int t = t0 + tt;
            bool ok = (t < seq_len) && (i < d);
            size_t off = (size_t)t * d + i;
            // Pad rows i >= d with zeros: k = q = 0 means they contribute nothing.
            rk[j] = ok ? __ldg(Kb + off) : 0.f;
            ra[j] = ok ? __ldg(Ab + off) : 0.f;
            rq[j] = ok ? __ldg(Qb + off) : 0.f;
        }
        #pragma unroll
        for (int j = 0; j < VLOADS; ++j) {
            int idx = tid + j * THREADS;
            int tt = idx / COLS, c = idx % COLS;
            int t = t0 + tt, cc = col0 + c;
            rv[j] = (t < seq_len && cc < d) ? __ldg(Vb + (size_t)t * d + cc) : 0.f;
        }
        rb = (tid < CH && t0 + tid < seq_len) ? __ldg(Bb + t0 + tid) : 0.f;
    };

    auto store = [&]() {
        #pragma unroll
        for (int j = 0; j < LOADS; ++j) {
            int idx = tid + j * THREADS;
            (&sk[0][0])[idx] = rk[j];
            (&sa[0][0])[idx] = ra[j];
            (&sq[0][0])[idx] = rq[j];
        }
        #pragma unroll
        for (int j = 0; j < VLOADS; ++j)
            (&sv[0][0])[tid + j * THREADS] = rv[j];
        if (tid < CH) sb[tid] = rb;
    };

    float s[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) s[r] = 0.f;

    prefetch(0);

    for (int t0 = 0; t0 < seq_len; t0 += CH) {
        store();
        __syncthreads();
        if (t0 + CH < seq_len) prefetch(t0 + CH);   // overlaps with compute below

        const int n = min(CH, seq_len - t0);
        for (int tt = 0; tt < n; ++tt) {
            float kr[R];
            float x = 0.f, y = 0.f, z = 0.f;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const int i = sub + LPC * r;
                const float kk = sk[tt][i];
                const float aa = sa[tt][i];
                const float qq = sq[tt][i];
                const float p  = aa * s[r];          // channel-wise decay
                s[r]  = p;
                kr[r] = kk;
                x = fmaf(kk, p, x);                 // k^T (Diag(a) S)
                y = fmaf(qq, p, y);                 // q^T (Diag(a) S)
                z = fmaf(qq, kk, z);                // q^T k
            }
            #pragma unroll
            for (int off = LPC / 2; off > 0; off >>= 1) {
                x += __shfl_xor_sync(0xffffffffu, x, off);
                y += __shfl_xor_sync(0xffffffffu, y, off);
                z += __shfl_xor_sync(0xffffffffu, z, off);
            }
            const float e = sb[tt] * (sv[tt][cl] - x);   // b (v - k^T P)
            #pragma unroll
            for (int r = 0; r < R; ++r) s[r] = fmaf(kr[r], e, s[r]);
            if (sub == 0) so[tt][cl] = fmaf(z, e, y);    // o = q^T S_t
        }
        __syncthreads();

        // Coalesced-ish write of this chunk's outputs.
        for (int idx = tid; idx < CH * COLS; idx += THREADS) {
            int tt = idx / COLS, c = idx % COLS;
            int t = t0 + tt, cc = col0 + c;
            if (t < seq_len && cc < d) Ob[(size_t)t * d + cc] = so[tt][c];
        }
    }
}

// Q, K, V, alpha, beta, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* alpha,
                      const float* beta, float* output, int batch, int seq_len, int d) {
    dim3 grid((d + COLS - 1) / COLS, batch);
    kda_kernel<<<grid, THREADS>>>(Q, K, V, alpha, beta, output, seq_len, d);
    cudaDeviceSynchronize();
}
