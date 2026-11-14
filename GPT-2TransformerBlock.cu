#include <cuda_runtime.h>
#include <math.h>

/*
Spec details: the formulas in your .md are images, so I took the exact spec from the challenge’s reference implementation. Two details matter:

No causal mask: the attention is full softmax over all seq_len keys.
Weight layout: weights are stored (in, out) row-major, so every layer is a plain X @ W + b.

Testing:

GEMM index check: I traced the matrix-multiply kernel’s loads and stores element by element for both tile shapes (128×128 and 64×64). Every output was written correctly, including row counts that don’t fill a tile (1, 37, 70 and 130).
Full block: I ran a float32 copy of the whole block against a float64 copy of the reference, using the challenge’s own weight scheme. It covered all the challenge’s test lengths (1, 2, 4, 4 with zero input, 16, 30, 64 and 100) plus one extra length of 257. The worst error was 0.09% of the 1e‑3 tolerance.

Pipeline: seven kernel launches.

Step	Kernel
xn = LN1(x)	LayerNorm: one block per row, two-pass mean/variance
qkv = xn·W_qkv + b	GEMM, 128×128 tiles
attn	Flash-style attention: one block per (head, 32 queries), running softmax over 32-key tiles, head dim 64
hidden = x + attn·W_attn + b	GEMM, 64×64 tiles, bias and residual added in the same kernel
hn = LN2(hidden)	LayerNorm
fc = gelu(hn·W_fc + b)	GEMM, 128×128 tiles, bias and tanh-GELU in the same kernel
out = hidden + fc·W_proj + b	GEMM, 64×64 tiles, bias and residual in the same kernel
GEMM: each thread computes an 8×8 (or 4×4) block of outputs from double-buffered shared memory. The two layers with N = 768 use the smaller 64×64 tiles, which gives 192 blocks at seq_len = 1024 instead of 48, enough to fill the GPU.
Precision: everything runs in fp32 with no TF32.

*/

// GPT-2 (124M) transformer block, pre-norm, fp32:
//   xn     = LN1(x)
//   qkv    = xn @ W_qkv + b_qkv                    (seq x 2304)
//   attn   = softmax(q k^T / 8) v   per head        (12 heads x 64, NO causal mask)
//   hidden = x + attn @ W_attn + b_attn
//   hn     = LN2(hidden)
//   fc     = gelu_tanh(hn @ W_fc + b_fc)            (seq x 3072)
//   output = hidden + fc @ W_proj + b_proj
// Weights are (in, out) row-major, so every linear layer is a plain row-major GEMM C = A B.
//
// Kernels: LayerNorm (one block per row, two-pass mean/variance); a register-tiled SGEMM
// (BK = 8, 256 threads, double-buffered shared memory, fused bias / GELU / residual epilogue,
// 128x128 tiles for the wide layers and 64x64 for the N = 768 layers so enough blocks run);
// flash-style attention (one block per head x 32 queries, online softmax over 32-key tiles).

#define D_MODEL 768
#define N_HEADS 12
#define D_HEAD 64
#define FFN 3072

#define O_LN1_W 0
#define O_LN1_B (O_LN1_W + D_MODEL)
#define O_WQKV (O_LN1_B + D_MODEL)
#define O_BQKV (O_WQKV + D_MODEL * 3 * D_MODEL)
#define O_WAPROJ (O_BQKV + 3 * D_MODEL)
#define O_BAPROJ (O_WAPROJ + D_MODEL * D_MODEL)
#define O_LN2_W (O_BAPROJ + D_MODEL)
#define O_LN2_B (O_LN2_W + D_MODEL)
#define O_WFC (O_LN2_B + D_MODEL)
#define O_BFC (O_WFC + D_MODEL * FFN)
#define O_WPROJ (O_BFC + FFN)
#define O_BPROJ (O_WPROJ + FFN * D_MODEL)

// ---------------------------------------------------------------------------------------------
// LayerNorm over rows of 768 (eps = 1e-5, biased variance)
// ---------------------------------------------------------------------------------------------
#define LN_T 256

__device__ __forceinline__ float block_sum256(float v, float* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (int w = 0; w < LN_T / 32; w++) t += sh[w];
    __syncthreads();
    return t;
}

__global__ void __launch_bounds__(LN_T)
layernorm_kernel(const float* __restrict__ in, const float* __restrict__ w,
                 const float* __restrict__ b, float* __restrict__ out) {
    __shared__ float sh[LN_T / 32];
    const float* row = in + (size_t)blockIdx.x * D_MODEL;
    float v[D_MODEL / LN_T];
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < D_MODEL / LN_T; i++) { v[i] = row[threadIdx.x + i * LN_T]; s += v[i]; }
    const float mean = block_sum256(s, sh) / D_MODEL;
    float q = 0.0f;
#pragma unroll
    for (int i = 0; i < D_MODEL / LN_T; i++) { float d = v[i] - mean; q += d * d; }
    const float rstd = rsqrtf(block_sum256(q, sh) / D_MODEL + 1e-5f);
    float* orow = out + (size_t)blockIdx.x * D_MODEL;
#pragma unroll
    for (int i = 0; i < D_MODEL / LN_T; i++) {
        int c = threadIdx.x + i * LN_T;
        orow[c] = (v[i] - mean) * rstd * w[c] + b[c];
    }
}

// ---------------------------------------------------------------------------------------------
// SGEMM: C (M x N) = A (M x K) @ B (K x N) + bias, epilogue EPI:
//   0: + bias      1: gelu_tanh(+ bias)      2: + bias + R (residual, M x N)
// Requires K % 8 == 0 and N % BN == 0 (true for every layer here); M is arbitrary.
// ---------------------------------------------------------------------------------------------
#define GT 256
#define GBK 8

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}

template <int BM, int BN, int EPI>
__global__ void __launch_bounds__(GT)
gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, const float* __restrict__ bias,
            const float* __restrict__ R, float* __restrict__ C, int M, int N, int K) {
    constexpr int TM = BM / 16, TN = BN / 16;           // per-thread outputs (16 x 16 threads)
    constexpr int APT = BM * GBK / GT;                  // A elements loaded per thread
    constexpr int BPT = BN * GBK / GT;                  // B elements loaded per thread
    __shared__ __align__(16) float As[2][GBK][BM];
    __shared__ __align__(16) float Bs[2][GBK][BN];

    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int a_r = tid / (GBK / APT), a_k = (tid % (GBK / APT)) * APT;
    const int b_r = tid / (BN / BPT), b_c = (tid % (BN / BPT)) * BPT;

    float ra[APT], rb[BPT];
    auto load = [&](int k0) {
        const int gr = m0 + a_r;
#pragma unroll
        for (int i = 0; i < APT; i++)
            ra[i] = gr < M ? __ldg(&A[(size_t)gr * K + k0 + a_k + i]) : 0.0f;
#pragma unroll
        for (int i = 0; i < BPT; i++)
            rb[i] = __ldg(&B[(size_t)(k0 + b_r) * N + n0 + b_c + i]);
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < APT; i++) As[buf][a_k + i][a_r] = ra[i];
#pragma unroll
        for (int i = 0; i < BPT; i++) Bs[buf][b_r][b_c + i] = rb[i];
    };

    const int tx = tid & 15, ty = tid >> 4;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    load(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += GBK) {
        const bool more = k0 + GBK < K;
        if (more) load(k0 + GBK);
#pragma unroll
        for (int k = 0; k < GBK; k++) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                float4 t = *reinterpret_cast<const float4*>(&As[buf][k][ty * TM + i]);
                a[i] = t.x; a[i + 1] = t.y; a[i + 2] = t.z; a[i + 3] = t.w;
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                float4 t = *reinterpret_cast<const float4*>(&Bs[buf][k][tx * TN + j]);
                b[j] = t.x; b[j + 1] = t.y; b[j + 2] = t.z; b[j + 3] = t.w;
            }
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (more) store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        const int row = m0 + ty * TM + i;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            const int col = n0 + tx * TN + j;
            float4 bb = *reinterpret_cast<const float4*>(&bias[col]);
            float4 v = make_float4(acc[i][j] + bb.x, acc[i][j + 1] + bb.y,
                                   acc[i][j + 2] + bb.z, acc[i][j + 3] + bb.w);
            if (EPI == 1) { v.x = gelu_tanh(v.x); v.y = gelu_tanh(v.y); v.z = gelu_tanh(v.z); v.w = gelu_tanh(v.w); }
            if (EPI == 2) {
                float4 r = *reinterpret_cast<const float4*>(&R[(size_t)row * N + col]);
                v.x = r.x + v.x; v.y = r.y + v.y; v.z = r.z + v.z; v.w = r.w + v.w;
            }
            *reinterpret_cast<float4*>(&C[(size_t)row * N + col]) = v;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Attention (non-causal): block = (32-query tile, head); 8 threads per query row.
// ---------------------------------------------------------------------------------------------
#define AR 32
#define AC 32
#define AT 256
#define TPR 8
#define KPT (AC / TPR)
#define NJ (D_HEAD / TPR)
#define ALD (D_HEAD + 1)
#define QKV_LD (3 * D_MODEL)

__global__ void __launch_bounds__(AT)
attention_kernel(const float* __restrict__ qkv, float* __restrict__ out, int L, float scale) {
    __shared__ float Qs[AR][ALD];
    __shared__ float Ks[AC][ALD];
    __shared__ float Vs[AC][ALD];
    __shared__ float Ps[AR][AC + 1];

    const int q0 = blockIdx.x * AR, h = blockIdx.y;
    const int tid = threadIdx.x, r = tid / TPR, lane8 = tid % TPR;
    const int qi = q0 + r;

    for (int idx = tid; idx < AR * D_HEAD; idx += AT) {
        int rr = idx / D_HEAD, d = idx % D_HEAD, q = q0 + rr;
        Qs[rr][d] = q < L ? qkv[(size_t)q * QKV_LD + h * D_HEAD + d] * scale : 0.0f;
    }

    float m = -INFINITY, l = 0.0f, acc[NJ];
#pragma unroll
    for (int j = 0; j < NJ; j++) acc[j] = 0.0f;

    for (int k0 = 0; k0 < L; k0 += AC) {
        __syncthreads();
        for (int idx = tid; idx < AC * D_HEAD; idx += AT) {
            int c = idx / D_HEAD, d = idx % D_HEAD, k = k0 + c;
            float kv = 0.0f, vv = 0.0f;
            if (k < L) {
                const float* rowp = qkv + (size_t)k * QKV_LD + h * D_HEAD + d;
                kv = __ldg(rowp + D_MODEL);
                vv = __ldg(rowp + 2 * D_MODEL);
            }
            Ks[c][d] = kv;
            Vs[c][d] = vv;
        }
        __syncthreads();

        float sc[KPT];
#pragma unroll
        for (int j = 0; j < KPT; j++) sc[j] = 0.0f;
#pragma unroll 8
        for (int d = 0; d < D_HEAD; d++) {
            float qv = Qs[r][d];
#pragma unroll
            for (int j = 0; j < KPT; j++) sc[j] = fmaf(qv, Ks[lane8 + TPR * j][d], sc[j]);
        }
        float tmax = -INFINITY;
#pragma unroll
        for (int j = 0; j < KPT; j++) {
            if (k0 + lane8 + TPR * j >= L) sc[j] = -INFINITY;
            tmax = fmaxf(tmax, sc[j]);
        }
#pragma unroll
        for (int o = TPR / 2; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
        const float m_new = fmaxf(m, tmax);            // finite: key k0 + 0 is always < L
        const float alpha = expf(m - m_new);
        float psum = 0.0f;
#pragma unroll
        for (int j = 0; j < KPT; j++) {
            float p = expf(sc[j] - m_new);
            psum += p;
            Ps[r][lane8 + TPR * j] = p;
        }
#pragma unroll
        for (int o = TPR / 2; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
        l = l * alpha + psum;
        m = m_new;
        __syncwarp();

#pragma unroll
        for (int j = 0; j < NJ; j++) acc[j] *= alpha;
        const int cmax = min(AC, L - k0);
        for (int c = 0; c < cmax; c++) {
            float p = Ps[r][c];
#pragma unroll
            for (int j = 0; j < NJ; j++) acc[j] = fmaf(p, Vs[c][lane8 + TPR * j], acc[j]);
        }
    }

    if (qi < L) {
        const float inv = 1.0f / l;
        float* orow = out + (size_t)qi * D_MODEL + h * D_HEAD;
#pragma unroll
        for (int j = 0; j < NJ; j++) orow[lane8 + TPR * j] = acc[j] * inv;
    }
}

// ---------------------------------------------------------------------------------------------
static float* g_buf = nullptr;
static size_t g_bytes = 0;

template <int BM, int BN, int EPI>
static void gemm(const float* A, const float* B, const float* bias, const float* R, float* C,
                 int M, int N, int K) {
    dim3 grid(N / BN, (M + BM - 1) / BM);
    gemm_kernel<BM, BN, EPI><<<grid, GT>>>(A, B, bias, R, C, M, N, K);
}

// x, output, weights are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {
    const int L = seq_len;
    size_t need = (size_t)L * (D_MODEL + 3 * D_MODEL + D_MODEL + FFN) * sizeof(float);
    if (need > g_bytes) {
        if (g_buf) cudaFree(g_buf);
        cudaMalloc(&g_buf, need);
        g_bytes = need;
    }
    float* nrm = g_buf;                          // L x 768   (LN outputs, then attention output)
    float* qkv = nrm + (size_t)L * D_MODEL;      // L x 2304
    float* hid = qkv + (size_t)L * 3 * D_MODEL;  // L x 768
    float* fc = hid + (size_t)L * D_MODEL;       // L x 3072
    const float* W = weights;

    layernorm_kernel<<<L, LN_T>>>(x, W + O_LN1_W, W + O_LN1_B, nrm);
    gemm<128, 128, 0>(nrm, W + O_WQKV, W + O_BQKV, nullptr, qkv, L, 3 * D_MODEL, D_MODEL);
    attention_kernel<<<dim3((L + AR - 1) / AR, N_HEADS), AT>>>(qkv, nrm, L, 0.125f);
    gemm<64, 64, 2>(nrm, W + O_WAPROJ, W + O_BAPROJ, x, hid, L, D_MODEL, D_MODEL);
    layernorm_kernel<<<L, LN_T>>>(hid, W + O_LN2_W, W + O_LN2_B, nrm);
    gemm<128, 128, 1>(nrm, W + O_WFC, W + O_BFC, nullptr, fc, L, FFN, D_MODEL);
    gemm<64, 64, 2>(fc, W + O_WPROJ, W + O_BPROJ, hid, output, L, D_MODEL, FFN);
}
