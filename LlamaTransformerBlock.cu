/*
Llama Transformer Block (d_model 512, 8 query heads / 2 KV heads, head_dim 64, SwiGLU 1408)
Spec taken from the challenge's reference implementation (leetgpu-challenges #93):
  xn     = RMSNorm(x) * w1                        eps = 1e-5
  q,k,v  = xn W_Q^T, xn W_K^T, xn W_V^T            weights stored (out, in)
  q,k    = RoPE half-split per head: [q1*c - q2*s, q1*s + q2*c], cos/sin: (seq, 32)
  attn   = causal softmax(q k^T / 8) v,  query head h uses KV head h / 4
  hidden = x + attn W_O^T
  hn     = RMSNorm(hidden) * w2
  out    = hidden + (silu(hn W_gate^T) * (hn W_up^T)) W_down^T
Tolerance 1e-3; performance measured at seq_len = 2048.
*/

/*
Pipeline (7 launches, all fp32):

  step                                     kernel
  xn  = RMSNorm1(x)                        rmsnorm: one 128-thread block per row (1 float4 each)
  qkv = xn [W_Q;W_K;W_V]^T  (seq x 768)    gemm_nt<64x64>: W_Q, W_K, W_V are adjacent in the
                                           buffer, so this is one (768 x 512) weight matrix
  RoPE on the 10 Q/K heads, in place       rope: one thread per rotated pair
  attn                                     flash attention: one block per (32 queries, q head),
                                           online softmax over 32-key tiles, causal, GQA
  hidden = x + attn W_O^T                  gemm_nt<64x64>, residual added in the epilogue
  hn  = RMSNorm2(hidden)                   rmsnorm
  gu  = hn [W_gate;W_up]^T  (seq x 2816)   gemm_nt<128x128>: gate and up are adjacent, so one GEMM
  out = hidden + (silu(g) * u) W_down^T    gemm_nt<64x64>; the A-tile loader computes
                                           silu(g) * u on the fly, so the 1408-wide activation
                                           is never written

GEMM: y = A W^T with both operands K-contiguous. A 256-thread register-tiled SGEMM,
BK = 8, both tiles stored k-major in double-buffered shared memory, the next tile
prefetched into registers. Each thread computes 8x8 (128x128 tiles) or 4x4 (64x64 tiles).
The narrow layers use 64x64 tiles so there are enough blocks (e.g. 32 x 8 = 256 at
seq 2048, N = 512). All N and K here are multiples of 64 and 8, so only M (seq_len)
needs bounds checks.
Attention: Q (with RoPE applied, pre-scaled by log2(e)/8) is loaded once per block. K
is stored transposed with an XOR swizzle; V and P are row-major / transposed-swizzled.
Shared memory is 20 KB. Query tiles are launched longest-first.
Scratch (xn/hn, qkv, attn, hidden, gu) is allocated once and grown when needed.
*/

#include <cuda_runtime.h>
#include <math.h>

#define D        512
#define NQH      8
#define NKVH     2
#define HD       64
#define QKV_N    (NQH * HD + 2 * NKVH * HD)      // 768
#define FFN      1408
#define GU_N     (2 * FFN)                       // 2816

#define O_RMS1   0
#define O_WQKV   (O_RMS1 + D)                    // W_Q, W_K, W_V contiguous: (768, 512)
#define O_WO     (O_WQKV + QKV_N * D)
#define O_RMS2   (O_WO + D * D)
#define O_WGU    (O_RMS2 + D)                    // W_gate, W_up contiguous: (2816, 512)
#define O_WDOWN  (O_WGU + GU_N * D)

// ======================================================================== RMSNorm
__global__ void __launch_bounds__(128)
rmsnorm(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ out)
{
    __shared__ float ws[4];
    const int row = blockIdx.x, t = threadIdx.x;
    const float4 v = __ldg(reinterpret_cast<const float4*>(in + (size_t)row * D) + t);
    float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if ((t & 31) == 0) ws[t >> 5] = s;
    __syncthreads();
    const float r = rsqrtf((ws[0] + ws[1] + ws[2] + ws[3]) * (1.0f / D) + 1e-5f);
    const float4 g = __ldg(reinterpret_cast<const float4*>(w) + t);
    reinterpret_cast<float4*>(out + (size_t)row * D)[t] =
        make_float4(v.x * r * g.x, v.y * r * g.y, v.z * r * g.z, v.w * r * g.w);
}

// ======================================================================== RoPE (in place on qkv)
__global__ void rope(float* __restrict__ qkv, const float* __restrict__ cs,
                     const float* __restrict__ sn, int seq)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;      // (s, head 0..9, j 0..31)
    if (idx >= seq * (NQH + NKVH) * (HD / 2)) return;
    const int j = idx & 31;
    const int hh = (idx >> 5) % (NQH + NKVH);
    const int s = idx / ((NQH + NKVH) * 32);
    float* p = qkv + (size_t)s * QKV_N + hh * HD;               // Q heads 0..7, then K heads
    const float c = __ldg(cs + s * 32 + j), si = __ldg(sn + s * 32 + j);
    const float a = p[j], b = p[j + 32];
    p[j]      = a * c - b * si;
    p[j + 32] = a * si + b * c;
}

// ======================================================================== GEMM  C = A W^T
// A: M x K (lda), W: N x K (row-major, ldw = K), C: M x N (ldc).
// ALOAD 1: A[m][k] = silu(G[m][k]) * G[m][k + FFN] with G = gu (lda = GU_N).
// EPI   1: C = R + acc (R has ldc).
#define GT 256
#define GBK 8

template <int BM, int BN, int ALOAD, int EPI>
__global__ void __launch_bounds__(GT)
gemm_nt(const float* __restrict__ A, int lda, const float* __restrict__ W,
        float* __restrict__ C, int ldc, const float* __restrict__ R, int M, int K)
{
    constexpr int TM = BM / 16, TN = BN / 16;         // 8 or 4
    constexpr int HM = TM / 4, HN = TN / 4;           // float4 groups per thread
    __shared__ __align__(16) float As[2][GBK][BM];
    __shared__ __align__(16) float Bs[2][GBK][BN];

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    float4 ra[(BM * 2 + GT - 1) / GT], rb[(BN * 2 + GT - 1) / GT];

    auto loadA = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < (BM * 2 + GT - 1) / GT; ++i) {
            const int f = tid + i * GT;
            if (f < BM * 2) {
                const int r = f >> 1, kk = k0 + (f & 1) * 4;
                const int m = m0 + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (m < M) {
                    if (ALOAD == 0) {
                        v = __ldg(reinterpret_cast<const float4*>(A + (size_t)m * lda + kk));
                    } else {
                        const float4 g = __ldg(reinterpret_cast<const float4*>(A + (size_t)m * lda + kk));
                        const float4 u = __ldg(reinterpret_cast<const float4*>(A + (size_t)m * lda + FFN + kk));
                        v.x = g.x / (1.f + __expf(-g.x)) * u.x;
                        v.y = g.y / (1.f + __expf(-g.y)) * u.y;
                        v.z = g.z / (1.f + __expf(-g.z)) * u.z;
                        v.w = g.w / (1.f + __expf(-g.w)) * u.w;
                    }
                }
                ra[i] = v;
            }
        }
    };
    auto loadB = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < (BN * 2 + GT - 1) / GT; ++i) {
            const int f = tid + i * GT;
            if (f < BN * 2) {
                const int r = f >> 1, kk = k0 + (f & 1) * 4;
                rb[i] = __ldg(reinterpret_cast<const float4*>(W + (size_t)(n0 + r) * K + kk));
            }
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int i = 0; i < (BM * 2 + GT - 1) / GT; ++i) {
            const int f = tid + i * GT;
            if (f < BM * 2) {
                const int r = f >> 1, kk = (f & 1) * 4;
                As[buf][kk][r] = ra[i].x; As[buf][kk + 1][r] = ra[i].y;
                As[buf][kk + 2][r] = ra[i].z; As[buf][kk + 3][r] = ra[i].w;
            }
        }
        #pragma unroll
        for (int i = 0; i < (BN * 2 + GT - 1) / GT; ++i) {
            const int f = tid + i * GT;
            if (f < BN * 2) {
                const int r = f >> 1, kk = (f & 1) * 4;
                Bs[buf][kk][r] = rb[i].x; Bs[buf][kk + 1][r] = rb[i].y;
                Bs[buf][kk + 2][r] = rb[i].z; Bs[buf][kk + 3][r] = rb[i].w;
            }
        }
    };

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    loadA(0); loadB(0); store(0);
    __syncthreads();

    const int KT = K / GBK;
    for (int kt = 0; kt < KT; ++kt) {
        const int cur = kt & 1;
        const bool more = kt + 1 < KT;
        if (more) { loadA((kt + 1) * GBK); loadB((kt + 1) * GBK); }

        #pragma unroll
        for (int kk = 0; kk < GBK; ++kk) {
            float a[TM], b[TN];
            #pragma unroll
            for (int h = 0; h < HM; ++h) {
                const float4 v = *reinterpret_cast<const float4*>(&As[cur][kk][h * (BM / 2) + ty * 4]);
                a[h * 4] = v.x; a[h * 4 + 1] = v.y; a[h * 4 + 2] = v.z; a[h * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int h = 0; h < HN; ++h) {
                const float4 v = *reinterpret_cast<const float4*>(&Bs[cur][kk][h * (BN / 2) + tx * 4]);
                b[h * 4] = v.x; b[h * 4 + 1] = v.y; b[h * 4 + 2] = v.z; b[h * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }

        if (more) store(cur ^ 1);
        __syncthreads();
    }

    // epilogue
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int m = m0 + (i / 4) * (BM / 2) + ty * 4 + (i & 3);
        if (m >= M) continue;
        #pragma unroll
        for (int h = 0; h < HN; ++h) {
            const int n = n0 + h * (BN / 2) + tx * 4;
            float4 v = make_float4(acc[i][h * 4], acc[i][h * 4 + 1], acc[i][h * 4 + 2], acc[i][h * 4 + 3]);
            if (EPI == 1) {
                const float4 r = __ldg(reinterpret_cast<const float4*>(R + (size_t)m * ldc + n));
                v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w;
            }
            *reinterpret_cast<float4*>(C + (size_t)m * ldc + n) = v;
        }
    }
}

// ======================================================================== attention
#define BQ 32
#define BKT 32
#define AT 128                                     // 16 x 8 threads

__device__ __forceinline__ int kc_idx(int d, int k) { return d * BKT + (k ^ (d & 31)); }
__device__ __forceinline__ int t32_idx(int a, int q) { return a * 32 + ((((q >> 2) ^ (a & 7))) << 2) + (q & 3); }

__global__ void __launch_bounds__(AT)
attention(const float* __restrict__ qkv, float* __restrict__ out, int seq, float qscale)
{
    __shared__ __align__(16) float Qs[HD * BQ];    // [d][q]
    __shared__ __align__(16) float KV[HD * BKT];   // Kc[d][k] or Vc[k][d]
    __shared__ __align__(16) float Pt[BKT * BQ];   // [k][q]

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int nQT = (seq + BQ - 1) / BQ;
    const int qt = nQT - 1 - blockIdx.x;           // longest rows first
    const int h = blockIdx.y, kvh = h / (NQH / NKVH);
    const int q0 = qt * BQ;
    const int qcol = h * HD, kcol = NQH * HD + kvh * HD, vcol = (NQH + NKVH) * HD + kvh * HD;

    for (int e = tid; e < BQ * HD; e += AT) {
        const int r = e >> 6, d = e & 63, q = q0 + r;
        Qs[t32_idx(d, r)] = (q < seq) ? __ldg(qkv + (size_t)q * QKV_N + qcol + d) * qscale : 0.f;
    }

    float o[4][4], m[4], l[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY; l[i] = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j] = 0.f;
    }

    for (int kt = 0; kt <= qt; ++kt) {
        const int k0 = kt * BKT;
        __syncthreads();                           // KV / Pt free (and Qs visible on kt = 0)
        for (int e = tid; e < BKT * HD; e += AT) {
            const int r = e >> 6, d = e & 63, k = k0 + r;
            KV[kc_idx(d, r)] = (k < seq) ? __ldg(qkv + (size_t)k * QKV_N + kcol + d) : 0.f;
        }
        __syncthreads();

        float s[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        #pragma unroll 8
        for (int d = 0; d < HD; ++d) {
            const float4 qv = *reinterpret_cast<const float4*>(&Qs[t32_idx(d, ty * 4)]);
            const float* kr = &KV[d * BKT];
            const int sw = d & 31;
            const float k0v = kr[tx ^ sw], k1v = kr[(tx + 16) ^ sw];
            s[0][0] = fmaf(qv.x, k0v, s[0][0]); s[0][1] = fmaf(qv.x, k1v, s[0][1]);
            s[1][0] = fmaf(qv.y, k0v, s[1][0]); s[1][1] = fmaf(qv.y, k1v, s[1][1]);
            s[2][0] = fmaf(qv.z, k0v, s[2][0]); s[2][1] = fmaf(qv.z, k1v, s[2][1]);
            s[3][0] = fmaf(qv.w, k0v, s[3][0]); s[3][1] = fmaf(qv.w, k1v, s[3][1]);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int qi = q0 + ty * 4 + i;
            float mx = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int k = k0 + tx + 16 * j;
                if (!(k <= qi && k < seq)) s[i][j] = -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            const float mNew = fmaxf(m[i], mx);
            float alpha = 1.f, sum = 0.f;
            if (mNew == -INFINITY) {
                s[i][0] = s[i][1] = 0.f;
            } else {
                alpha = exp2f(m[i] - mNew);
                #pragma unroll
                for (int j = 0; j < 2; ++j) { s[i][j] = exp2f(s[i][j] - mNew); sum += s[i][j]; }
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            m[i] = mNew;
            l[i] = l[i] * alpha + sum;
            #pragma unroll
            for (int j = 0; j < 4; ++j) o[i][j] *= alpha;
            #pragma unroll
            for (int j = 0; j < 2; ++j) Pt[t32_idx(tx + 16 * j, ty * 4 + i)] = s[i][j];
        }
        __syncthreads();                           // K reads done, Pt complete

        for (int e = tid; e < BKT * HD; e += AT) {
            const int r = e >> 6, d = e & 63, k = k0 + r;
            KV[r * HD + d] = (k < seq) ? __ldg(qkv + (size_t)k * QKV_N + vcol + d) : 0.f;
        }
        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < BKT; ++k) {
            const float4 pv = *reinterpret_cast<const float4*>(&Pt[t32_idx(k, ty * 4)]);
            const float4 vv = *reinterpret_cast<const float4*>(&KV[k * HD + tx * 4]);
            const float pa[4] = {pv.x, pv.y, pv.z, pv.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                o[i][0] = fmaf(pa[i], vv.x, o[i][0]);
                o[i][1] = fmaf(pa[i], vv.y, o[i][1]);
                o[i][2] = fmaf(pa[i], vv.z, o[i][2]);
                o[i][3] = fmaf(pa[i], vv.w, o[i][3]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = q0 + ty * 4 + i;
        if (q >= seq) continue;
        const float inv = 1.f / l[i];
        *reinterpret_cast<float4*>(out + (size_t)q * D + qcol + tx * 4) =
            make_float4(o[i][0] * inv, o[i][1] * inv, o[i][2] * inv, o[i][3] * inv);
    }
}

// ======================================================================== host
static float* g_ws = nullptr;
static size_t g_cap = 0;

// x, output, weights, cos, sin are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, const float* cos,
                      const float* sin, int seq_len)
{
    const int S = seq_len;
    if (S <= 0) return;

    const size_t need = (size_t)S * (D + QKV_N + D + D + GU_N);
    if (need > g_cap) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, need * sizeof(float));
        g_cap = need;
    }
    float* xn     = g_ws;                 // S x 512 (also hn)
    float* qkv    = xn + (size_t)S * D;   // S x 768
    float* attn   = qkv + (size_t)S * QKV_N;
    float* hidden = attn + (size_t)S * D;
    float* gu     = hidden + (size_t)S * D;   // S x 2816

    const int m64 = (S + 63) / 64, m128 = (S + 127) / 128;

    // attention sub-block
    rmsnorm<<<S, 128>>>(x, weights + O_RMS1, xn);
    gemm_nt<64, 64, 0, 0><<<dim3(QKV_N / 64, m64), GT>>>(xn, D, weights + O_WQKV, qkv, QKV_N, nullptr, S, D);
    const int pairs = S * (NQH + NKVH) * (HD / 2);
    rope<<<(pairs + 255) / 256, 256>>>(qkv, cos, sin, S);
    const float qscale = 1.4426950408889634f / 8.0f;           // log2(e) / sqrt(64)
    attention<<<dim3((S + BQ - 1) / BQ, NQH), AT>>>(qkv, attn, S, qscale);
    gemm_nt<64, 64, 0, 1><<<dim3(D / 64, m64), GT>>>(attn, D, weights + O_WO, hidden, D, x, S, D);

    // FFN sub-block
    rmsnorm<<<S, 128>>>(hidden, weights + O_RMS2, xn);
    gemm_nt<128, 128, 0, 0><<<dim3(GU_N / 128, m128), GT>>>(xn, D, weights + O_WGU, gu, GU_N, nullptr, S, D);
    gemm_nt<64, 64, 1, 1><<<dim3(D / 64, m64), GT>>>(gu, GU_N, weights + O_WDOWN, output, D, hidden, S, FFN);

    cudaDeviceSynchronize();
}
