/*
Fused MoE SwiGLU Layer
x (T, H); topk_ids, topk_weights (T, K); w_gate, w_up (E, H, I); w_down (E, I, H);
output (T, H). Spec from the reference implementation (leetgpu-challenges #130),
tolerance 1e-3:
  out[t] = sum_k  w[t,k] * ( silu(x[t] @ Wg[e]) * (x[t] @ Wu[e]) ) @ Wd[e],   e = ids[t,k]
Weights are used as given (they may be unnormalized or negative); experts may get no tokens.
Constraints: T ≤ 8192, H, I ≤ 4096, K ≤ E ≤ 64. Performance at T = 4096, H = 1024,
I = 1408, E = 16, K = 2.
*/

/*
Approach (4 launches, FP32; only the routed (token, expert) pairs are computed):
 1. route        One 1024-thread block does a stable counting sort of the T*K
                 assignments by expert: a shared-memory histogram, a scan into
                 offsets, then rank-in-warp with __match_any_sync plus per-warp
                 offsets for a deterministic placement. Produces
                 off[E+1], tok[slot] and slot_of[t*K + k].
 2. gemm<GATEUP> Grouped GEMM over the expert-sorted slots. Each row tile of 128 slots
                 belongs to one expert; the device-side tile schedule is the same as
                 in GroupedGEMM, and surplus blocks exit. The A tile is gathered from
                 x[tok[slot]]. The same A tile multiplies BOTH w_gate[e] and w_up[e]
                 (64 columns each per block), and the epilogue writes
                 h = silu(g) * u to hbuf[slot][I]. So gate and up are never stored,
                 and A is loaded once for both.
 3. gemm<DOWN>   Grouped GEMM y[slot] = hbuf[slot] @ w_down[e] (128 x 128 tiles).
 4. combine      out[t] = sum_k w[t,k] * y[slot_of[t,k]], in fixed k order. Each
                 output element is written exactly once, so it's deterministic, with
                 no atomics and no memset needed.
GEMM core: 256 threads, BK = 8, A stored k-major, double-buffered shared memory with
register prefetch, 8 x 8 outputs per thread (8 x (4 + 4) in the gate/up kernel).
float4 global access when H % 4 == 0 and I % 4 == 0, guarded scalars otherwise.
FLOPs at the benchmark: (2*H*I*2 + I*H*2) * T*K ≈ 71 GFLOP.
*/

#include <cuda_runtime.h>

#define GT 256
#define GBK 8
#define BM 128
#define MAX_E 64
#define RT 1024

// ======================================================================== 1. routing
__global__ void __launch_bounds__(RT)
route(const int* __restrict__ ids, int TK, int K, int E,
      int* __restrict__ off, int* __restrict__ tok, int* __restrict__ slot_of)
{
    __shared__ int cnt[MAX_E], soff[MAX_E + 1], running[MAX_E];
    __shared__ int wc[RT / 32][MAX_E];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    if (tid < MAX_E) { cnt[tid] = 0; running[tid] = 0; }
    __syncthreads();
    for (int i = tid; i < TK; i += RT) atomicAdd(&cnt[ids[i]], 1);
    __syncthreads();
    if (tid == 0) {
        int a = 0;
        for (int e = 0; e < E; ++e) { soff[e] = a; off[e] = a; a += cnt[e]; }
        soff[E] = a; off[E] = a;
    }
    __syncthreads();

    const unsigned lt = (1u << lane) - 1u;
    for (int base = 0; base < TK; base += RT) {
        for (int i = tid; i < (RT / 32) * MAX_E; i += RT) (&wc[0][0])[i] = 0;
        __syncthreads();
        const int i = base + tid;
        const int e = (i < TK) ? ids[i] : -1;
        const unsigned peers = __match_any_sync(0xffffffff, e);
        const int rank = __popc(peers & lt);
        if (e >= 0 && rank == 0) wc[warp][e] = __popc(peers);
        __syncthreads();
        if (tid < E) {
            int run = running[tid];
            for (int w = 0; w < RT / 32; ++w) { const int c = wc[w][tid]; wc[w][tid] = run; run += c; }
            running[tid] = run;
        }
        __syncthreads();
        if (e >= 0) {
            const int slot = soff[e] + wc[warp][e] + rank;
            slot_of[i] = slot;
            tok[slot] = i / K;
        }
        __syncthreads();
    }
}

// ======================================================================== 2/3. grouped GEMMs
template <bool VEC>
__device__ __forceinline__ float4 ld4(const float* p, int col, int limit, bool ok)
{
    if (!ok) return make_float4(0.f, 0.f, 0.f, 0.f);
    if (VEC) return (col < limit) ? __ldg(reinterpret_cast<const float4*>(p + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 v;
    v.x = (col     < limit) ? __ldg(p + col)     : 0.f;
    v.y = (col + 1 < limit) ? __ldg(p + col + 1) : 0.f;
    v.z = (col + 2 < limit) ? __ldg(p + col + 2) : 0.f;
    v.w = (col + 3 < limit) ? __ldg(p + col + 3) : 0.f;
    return v;
}

// GATEUP: A = x[tok[slot]] (Kd = H), B0 = Wg[e], B1 = Wu[e] (H x I), out hbuf (ldo = I)
// DOWN  : A = hbuf[slot]   (Kd = I), B0 = Wd[e] (I x H),            out y    (ldo = H)
template <bool GATEUP, bool VEC>
__global__ void __launch_bounds__(GT)
moe_gemm(const float* __restrict__ A, const int* __restrict__ tok, const int* __restrict__ off,
         const float* __restrict__ B0, const float* __restrict__ B1, float* __restrict__ out,
         int E, int Kd, int Nd)
{
    constexpr int NBT = GATEUP ? 2 : 1;           // B tiles per block
    constexpr int BNW = GATEUP ? 64 : 128;        // width of each B tile
    constexpr int TNC = BNW / 16;                 // cols per thread per tile (4 or 8)
    __shared__ __align__(16) float As[2][GBK][BM];
    __shared__ __align__(16) float Bs[2][NBT][GBK][BNW];
    __shared__ int tileStart[MAX_E + 1];
    __shared__ int info[3];

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;

    // ---- schedule: blockIdx.y -> (expert, slot range)
    if (tid == 0) {
        int a = 0;
        for (int e = 0; e < E; ++e) { tileStart[e] = a; a += (off[e + 1] - off[e] + BM - 1) / BM; }
        tileStart[E] = a;
        const int t = blockIdx.y;
        info[0] = -1;
        if (t < a) {
            int e = 0;
            while (tileStart[e + 1] <= t) ++e;
            info[0] = e;
            info[1] = off[e] + (t - tileStart[e]) * BM;
            info[2] = off[e + 1];
        }
    }
    __syncthreads();
    const int e = info[0];
    if (e < 0) return;
    const int r0 = info[1], rEnd = info[2];
    const int n0 = blockIdx.x * BNW;
    const size_t bstride = (size_t)Kd * Nd;
    const float* Bp[2] = {B0 + (size_t)e * bstride, GATEUP ? B1 + (size_t)e * bstride : nullptr};

    // A load: row = tid/2, k4 = (tid&1)*4
    const int ar = tid >> 1, ak = (tid & 1) * 4;
    const bool aOk = r0 + ar < rEnd;
    const float* aRow = A;
    if (aOk) aRow = GATEUP ? A + (size_t)tok[r0 + ar] * Kd : A + (size_t)(r0 + ar) * Kd;
    // B load: tile bt, row kr, col c4 (one float4 per thread in both modes)
    const int bt = GATEUP ? (tid >> 7) : 0;
    const int bf = GATEUP ? (tid & 127) : tid;
    const int bkr = bf / (BNW / 4), bc = (bf % (BNW / 4)) * 4;

    float acc[8][NBT * TNC];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < NBT * TNC; ++j) acc[i][j] = 0.f;

    float4 ra = ld4<VEC>(aRow, ak, Kd, aOk);
    float4 rb = ld4<VEC>(Bp[bt] + (size_t)bkr * Nd, n0 + bc, Nd, bkr < Kd);
    As[0][ak][ar] = ra.x; As[0][ak + 1][ar] = ra.y; As[0][ak + 2][ar] = ra.z; As[0][ak + 3][ar] = ra.w;
    *reinterpret_cast<float4*>(&Bs[0][bt][bkr][bc]) = rb;
    __syncthreads();

    const int KT = (Kd + GBK - 1) / GBK;
    for (int kt = 0; kt < KT; ++kt) {
        const int cur = kt & 1;
        const bool more = kt + 1 < KT;
        if (more) {
            const int k0 = (kt + 1) * GBK;
            ra = ld4<VEC>(aRow, k0 + ak, Kd, aOk);
            rb = ld4<VEC>(Bp[bt] + (size_t)(k0 + bkr) * Nd, n0 + bc, Nd, k0 + bkr < Kd);
        }
        #pragma unroll
        for (int kk = 0; kk < GBK; ++kk) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[cur][kk][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[cur][kk][64 + ty * 4]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b[NBT * TNC];
            #pragma unroll
            for (int t = 0; t < NBT; ++t) {
                #pragma unroll
                for (int h = 0; h < TNC / 4; ++h) {
                    const float4 v = *reinterpret_cast<const float4*>(&Bs[cur][t][kk][h * 64 + tx * 4]);
                    b[t * TNC + h * 4] = v.x; b[t * TNC + h * 4 + 1] = v.y;
                    b[t * TNC + h * 4 + 2] = v.z; b[t * TNC + h * 4 + 3] = v.w;
                }
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < NBT * TNC; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (more) {
            const int nx = cur ^ 1;
            As[nx][ak][ar] = ra.x; As[nx][ak + 1][ar] = ra.y; As[nx][ak + 2][ar] = ra.z; As[nx][ak + 3][ar] = ra.w;
            *reinterpret_cast<float4*>(&Bs[nx][bt][bkr][bc]) = rb;
        }
        __syncthreads();
    }

    // ---- epilogue
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = r0 + ((i < 4) ? (ty * 4 + i) : (64 + ty * 4 + i - 4));
        if (r >= rEnd) continue;
        float* orow = out + (size_t)r * Nd;
        #pragma unroll
        for (int h = 0; h < TNC / 4; ++h) {
            const int c = n0 + h * 64 + tx * 4;
            float v[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (GATEUP) {
                    const float g = acc[i][h * 4 + j], u = acc[i][TNC + h * 4 + j];
                    v[j] = g / (1.f + expf(-g)) * u;              // silu(g) * u
                } else {
                    v[j] = acc[i][h * 4 + j];
                }
            }
            if (VEC) {
                if (c < Nd) *reinterpret_cast<float4*>(orow + c) = make_float4(v[0], v[1], v[2], v[3]);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) if (c + j < Nd) orow[c + j] = v[j];
            }
        }
    }
}

// ======================================================================== 4. combine
__global__ void combine(const float* __restrict__ y, const float* __restrict__ w,
                        const int* __restrict__ slot_of, float* __restrict__ out, int H, int K)
{
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            const int i = t * K + k;
            acc = fmaf(__ldg(w + i), y[(size_t)slot_of[i] * H + c], acc);
        }
        out[(size_t)t * H + c] = acc;
    }
}

// ======================================================================== host
static void*  g_ws = nullptr;
static size_t g_bytes = 0;

template <bool VEC>
static void run(const float* x, const float* wg, const float* wu, const float* wd,
                const int* off, const int* tok, float* hbuf, float* y, int TK, int H, int I, int E)
{
    const int rowTiles = (TK + BM - 1) / BM + E;          // ≥ sum_e ceil(cnt_e / BM)
    moe_gemm<true,  VEC><<<dim3((I + 63) / 64,   rowTiles), GT>>>(x, tok, off, wg, wu, hbuf, E, H, I);
    moe_gemm<false, VEC><<<dim3((H + 127) / 128, rowTiles), GT>>>(hbuf, tok, off, wd, nullptr, y, E, I, H);
}

// x, topk_ids, topk_weights, w_gate, w_up, w_down, output are device pointers
extern "C" void solve(const float* x, const int* topk_ids, const float* topk_weights,
                      const float* w_gate, const float* w_up, const float* w_down, float* output,
                      int T, int H, int I, int E, int K)
{
    if (T <= 0 || H <= 0 || I <= 0 || E <= 0 || K <= 0) return;
    const int TK = T * K;
    const size_t need = sizeof(int) * ((size_t)E + 1 + 2 * (size_t)TK) + 16
                      + sizeof(float) * ((size_t)TK * I + (size_t)TK * H);
    if (need > g_bytes) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, need);
        g_bytes = need;
    }
    float* hbuf = reinterpret_cast<float*>(g_ws);                 // TK x I
    float* y    = hbuf + (size_t)TK * I;                           // TK x H
    int* off     = reinterpret_cast<int*>(y + (size_t)TK * H);     // E + 1
    int* tok     = off + (E + 1);                                  // TK
    int* slot_of = tok + TK;                                       // TK

    route<<<1, RT>>>(topk_ids, TK, K, E, off, tok, slot_of);
    if ((H % 4 == 0) && (I % 4 == 0)) run<true >(x, w_gate, w_up, w_down, off, tok, hbuf, y, TK, H, I, E);
    else                              run<false>(x, w_gate, w_up, w_down, off, tok, hbuf, y, TK, H, I, E);
    combine<<<T, 256>>>(y, topk_weights, slot_of, output, H, K);
    cudaDeviceSynchronize();
}
