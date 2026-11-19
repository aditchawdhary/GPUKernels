/*
Fused Mamba-2 SSD Scan
x, z, y: (b, L, H, P); dt: (b, L, H); dt_bias, A, D: (H); B, C: (b, L, G, N); float32.
Head h reads group g = h / (H/G). Spec from the reference implementation
(leetgpu-challenges #154), tolerance 1e-4:
  delta_t = softplus(dt_t + dt_bias)                (torch: linear above 20)
  S_t     = exp(delta_t * A) * S_{t-1} + (delta_t * x_t) B_t^T      S: P x N, S_{-1} = 0
  y_t     = (S_t C_t + D * x_t) * silu(z_t)
Constraints: b ≤ 4, L ≤ 4096, H ≤ 32, P ≤ 128, N ≤ 128.
Performance at b = 2, L = 4096, H = 32, P = 64, G = 4, N = 128.
*/

/*
Approach: the scalar-decay recurrence computed directly as a chunked parallel scan
(the same three-pass structure as SSMSelectiveScan). It's exact in FP32 and avoids
the O(T^2) intra-chunk matrices of the SSD form.

Within a chunk the map is affine in the incoming state:
  S_end = exp(Lambda) * S_in + S_loc,   Lambda = sum_t delta_t * A   (one scalar per head)
 1. ssd_chunk<false>  grid (chunks, H, b). Runs the chunk from S = 0 and writes S_loc
                      and Lambda.
 2. fold              one thread per state element (b, h, p, n): walks the chunks in order
                      and turns S_loc into S_in, the state entering each chunk.
 3. ssd_chunk<true>   restarts each chunk from S_in and writes y.
Thread layout of ssd_chunk: thread = (n-group ng, row p), with tid = ng * Pp + p, where
Pp = P rounded up to 32. Every warp therefore shares one n-group, so the B_t / C_t reads
from shared memory are broadcasts. Each thread keeps 32 state values S[p, 32ng .. 32ng+31]
in registers. Per step:
  u = delta_t * x_t[p];  s_j = a_t * s_j + u * B_t[j];  y += s_j * C_t[j]
B, C, x, delta and a = exp(delta * A) are staged through shared memory 8 steps at a time.
The y partials of the n-groups are summed in shared memory, and the n-group-0 threads
apply (+ D x) * silu(z) and store. Shared memory is 28 KB at most.
Chunk count: about 512 blocks total (8 chunks of 512 steps at the benchmark). The
scratch S_loc / S_in is b*H*chunks*P*N floats (33 MB at the benchmark).
*/

#include <cuda_runtime.h>
#include <math.h>

#define NPT 32                     // state columns per thread
#define SUB 8                      // steps staged per sub-tile
#define MAXP 128
#define MAXN 128
#define MAXNG (MAXN / NPT)

__device__ __forceinline__ float softplus_t(float v)        // torch default: beta=1, threshold=20
{
    return v > 20.f ? v : log1pf(expf(v));
}

template <bool OUT>
__global__ void ssd_chunk(const float* __restrict__ x, const float* __restrict__ dt,
                          const float* __restrict__ dt_bias, const float* __restrict__ A,
                          const float* __restrict__ B, const float* __restrict__ C,
                          const float* __restrict__ Dsk, const float* __restrict__ z,
                          float* __restrict__ y, float* __restrict__ Sbuf, float* __restrict__ lam,
                          int L, int H, int P, int G, int N, int Pp, int NG, int TC, int nc)
{
    __shared__ __align__(16) float Bs[SUB][MAXN];
    __shared__ __align__(16) float Cs[SUB][MAXN];
    __shared__ float xs[SUB][MAXP];
    __shared__ float ds[SUB], as[SUB];
    __shared__ float yp[MAXNG][SUB][MAXP];
    __shared__ float lamAcc;

    const int c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int ng = tid / Pp, p = tid - ng * Pp;
    const bool act = p < P;
    const int g = h / (H / G);
    const int t0 = c * TC, t1 = min(t0 + TC, L);
    const float Ah = __ldg(A + h), bias = __ldg(dt_bias + h);
    const size_t sbase = ((((size_t)b * H + h) * nc + c) * P) * N;   // [b][h][c][p][n]

    float s[NPT];
    #pragma unroll
    for (int j = 0; j < NPT; ++j) {
        const int n = ng * NPT + j;
        s[j] = (OUT && c > 0 && act && n < N) ? Sbuf[sbase + (size_t)p * N + n] : 0.f;
    }
    if (tid == 0) lamAcc = 0.f;

    for (int ts = t0; ts < t1; ts += SUB) {
        const int cnt = min(SUB, t1 - ts);
        __syncthreads();                                      // previous sub-tile fully consumed
        const int NW = NG * NPT;                             // padded width: zero past N
        for (int e = tid; e < SUB * NW; e += nthr) {
            const int k = e / NW, n = e - k * NW, t = ts + k;
            const bool ok = (k < cnt) && (n < N);
            const size_t bi = (((size_t)b * L + t) * G + g) * N + n;
            Bs[k][n] = ok ? __ldg(B + bi) : 0.f;
            if (OUT) Cs[k][n] = ok ? __ldg(C + bi) : 0.f;
        }
        for (int e = tid; e < SUB * P; e += nthr) {
            const int k = e / P, pp = e - k * P, t = ts + k;
            xs[k][pp] = (k < cnt) ? __ldg(x + (((size_t)b * L + t) * H + h) * P + pp) : 0.f;
        }
        if (tid < SUB) {
            float dl = 0.f, a = 1.f;
            if (tid < cnt) {
                dl = softplus_t(__ldg(dt + ((size_t)b * L + ts + tid) * H + h) + bias);
                a = expf(dl * Ah);
            }
            ds[tid] = dl; as[tid] = a;
        }
        __syncthreads();

        if (!OUT && tid == 0) {
            float acc = lamAcc;
            for (int k = 0; k < cnt; ++k) acc += ds[k] * Ah;
            lamAcc = acc;
        }

        for (int k = 0; k < cnt; ++k) {
            const float a = as[k];
            const float u = ds[k] * (act ? xs[k][p] : 0.f);
            const float* Bk = &Bs[k][ng * NPT];
            float yv = 0.f;
            #pragma unroll
            for (int j = 0; j < NPT; ++j) {
                s[j] = fmaf(a, s[j], u * Bk[j]);              // B past N is 0 -> s stays 0
                if (OUT) yv = fmaf(s[j], Cs[k][ng * NPT + j], yv);
            }
            if (OUT) yp[ng][k][p < MAXP ? p : 0] = yv;
        }

        if (OUT) {
            __syncthreads();
            if (ng == 0 && act) {
                const float dsk = __ldg(Dsk + h);
                for (int k = 0; k < cnt; ++k) {
                    float v = 0.f;
                    for (int q = 0; q < NG; ++q) v += yp[q][k][p];
                    const size_t oi = (((size_t)b * L + ts + k) * H + h) * P + p;
                    const float zz = __ldg(z + oi);
                    v = fmaf(dsk, xs[k][p], v);
                    y[oi] = v * (zz / (1.f + expf(-zz)));     // * silu(z)
                }
            }
        }
    }

    if (!OUT) {
        #pragma unroll
        for (int j = 0; j < NPT; ++j) {
            const int n = ng * NPT + j;
            if (act && n < N) Sbuf[sbase + (size_t)p * N + n] = s[j];
        }
        __syncthreads();
        if (tid == 0) lam[((size_t)b * H + h) * nc + c] = lamAcc;
    }
}

// S_loc -> S_in (state entering each chunk), in place
__global__ void fold(float* __restrict__ Sbuf, const float* __restrict__ lam, int H, int P, int N,
                     int nc, size_t total)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // over (b, h, p, n)
    if (idx >= total) return;
    const size_t PN = (size_t)P * N;
    const size_t bh = idx / PN, e = idx - bh * PN;
    float sin = 0.f;
    for (int c = 0; c < nc; ++c) {
        float* sp = Sbuf + (bh * nc + c) * PN + e;
        const float loc = *sp;
        *sp = sin;
        sin = fmaf(expf(lam[bh * nc + c]), sin, loc);
    }
}

static float* g_ws = nullptr;
static size_t g_cap = 0;

// x, dt, dt_bias, A, B, C, D, z, y are device pointers
extern "C" void solve(const float* x, const float* dt, const float* dt_bias, const float* A,
                      const float* B, const float* C, const float* D, const float* z, float* y,
                      int batch, int seq_len, int num_heads, int head_dim, int num_groups,
                      int state_dim)
{
    const int bsz = batch, L = seq_len, H = num_heads, P = head_dim, G = num_groups, N = state_dim;
    if (bsz <= 0 || L <= 0 || H <= 0 || P <= 0 || N <= 0) return;

    const int Pp = ((P + 31) / 32) * 32;
    const int NG = (N + NPT - 1) / NPT;
    const int threads = Pp * NG;                         // ≤ 512

    int nc = (512 + bsz * H - 1) / (bsz * H);            // aim for ~512 blocks
    const int maxc = (L + 63) / 64;                      // chunks of at least 64 steps
    if (nc > maxc) nc = maxc;
    if (nc < 1) nc = 1;
    const int TC = (L + nc - 1) / nc;
    nc = (L + TC - 1) / TC;

    const size_t sElems = (size_t)bsz * H * nc * P * N;
    const size_t need = sElems + (size_t)bsz * H * nc;
    if (need > g_cap) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, need * sizeof(float));
        g_cap = need;
    }
    float* Sbuf = g_ws;
    float* lam = g_ws + sElems;

    dim3 grid(nc, H, bsz);
    if (nc > 1) {
        ssd_chunk<false><<<grid, threads>>>(x, dt, dt_bias, A, B, C, D, z, y, Sbuf, lam,
                                            L, H, P, G, N, Pp, NG, TC, nc);
        const size_t total = (size_t)bsz * H * P * N;
        fold<<<(unsigned)((total + 255) / 256), 256>>>(Sbuf, lam, H, P, N, nc, total);
    }
    ssd_chunk<true><<<grid, threads>>>(x, dt, dt_bias, A, B, C, D, z, y, Sbuf, lam,
                                       L, H, P, G, N, Pp, NG, TC, nc);
    cudaDeviceSynchronize();
}
