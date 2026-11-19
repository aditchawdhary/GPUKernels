#include <cuda_runtime.h>
#include <math.h>

// DiT block (adaLN-Zero), FP32 throughout. d = 512, 8 heads x 64, MLP 2048.
//   mod = SiLU(c) W_ada^T + b_ada  -> [shift_msa | scale_msa | gate_msa | shift_mlp | scale_mlp | gate_mlp]
//   h   = LN(x)  * (1 + scale_msa) + shift_msa
//   x1  = x  + gate_msa * (Attn(h) W_o^T + b_o)
//   h2  = LN(x1) * (1 + scale_mlp) + shift_mlp
//   out = x1 + gate_mlp * (GELU_tanh(h2 W_fc1^T + b_fc1) W_fc2^T + b_fc2)
//
// Pipeline (8 launches):
//   1. mod_kernel             : the B x 3072 modulation vectors (one warp per output).
//   2. ln_mod_kernel          : h = modulated LayerNorm of x (one warp per token row).
//   3. gemm<BIAS>             : qkv = h W_qkv^T + b_qkv                  (BT x 1536)
//   4. attn_kernel            : bidirectional FlashAttention per (batch, head), FP32
//   5. gemm<GATED_RESIDUAL>   : x1 = x + gate_msa * (attn W_o^T + b_o)
//   6. ln_mod_kernel          : h2 from x1
//   7. gemm<BIAS_GELU>        : u = GELU(h2 W_fc1^T + b_fc1)             (BT x 2048)
//   8. gemm<GATED_RESIDUAL>   : out = x1 + gate_mlp * (u W_fc2^T + b_fc2)
// The GEMM is a 128 x 128 x 8 register-tiled FP32 SIMT kernel (8 x 8 outputs
// per thread, double-buffered smem, transposed tiles for float4 reads). Its
// epilogues fuse bias, GELU and the gated residual, so no extra elementwise passes
// are needed. N and K are fixed multiples of 128 / 8 here; M = B*T is guarded.

#define D 512
#define NH 8
#define DH 64
#define MLP 2048

#define O_WADA 0
#define O_BADA 1572864
#define O_WQKV 1575936
#define O_BQKV 2362368
#define O_WO   2363904
#define O_BO   2626048
#define O_WFC1 2626560
#define O_BFC1 3675136
#define O_WFC2 3677184
#define O_BFC2 4725760

// ---------------- 1. adaLN modulation ----------------
__global__ void mod_kernel(const float* __restrict__ c, const float* __restrict__ W,
                           const float* __restrict__ bias, float* __restrict__ mod, int B)
{
    const int lane = threadIdx.x & 31;
    const int o = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;      // over B * 3072
    if (o >= B * 6 * D) return;
    const int b = o / (6 * D), n = o % (6 * D);
    const float* cr = c + (size_t)b * D;
    const float* wr = W + (size_t)n * D;
    float acc = 0.f;
    for (int k = lane; k < D; k += 32) {
        const float z = __ldg(cr + k);
        acc = fmaf(z / (1.f + expf(-z)), __ldg(wr + k), acc);      // SiLU(c) . W[n]
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0) mod[o] = acc + bias[n];
}

// ---------------- 2/6. LayerNorm (no affine) + modulate ----------------
// shift = mod[b][sh*512 ..], scale = mod[b][(sh+1)*512 ..]
__global__ void ln_mod_kernel(const float* __restrict__ x, const float* __restrict__ mod,
                              float* __restrict__ h, int rows, int T, int sh)
{
    const int lane = threadIdx.x & 31;
    const int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (r >= rows) return;
    const int b = r / T;
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)r * D);
    float4 v[4];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] = xr[lane + 32 * i]; s += v[i].x + v[i].y + v[i].z + v[i].w; }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    const float mu = s * (1.f / D);
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float a = v[i].x - mu, bb = v[i].y - mu, cc = v[i].z - mu, dd = v[i].w - mu;
        q += a * a + bb * bb + cc * cc + dd * dd;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) q += __shfl_xor_sync(0xffffffff, q, off);
    const float rs = rsqrtf(q * (1.f / D) + 1e-6f);
    const float4* shf = reinterpret_cast<const float4*>(mod + (size_t)b * 6 * D + sh * D);
    const float4* scl = reinterpret_cast<const float4*>(mod + (size_t)b * 6 * D + (sh + 1) * D);
    float4* hr = reinterpret_cast<float4*>(h + (size_t)r * D);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = lane + 32 * i;
        const float4 a = shf[k], g = scl[k];
        float4 o;
        o.x = (v[i].x - mu) * rs * (1.f + g.x) + a.x;
        o.y = (v[i].y - mu) * rs * (1.f + g.y) + a.y;
        o.z = (v[i].z - mu) * rs * (1.f + g.z) + a.z;
        o.w = (v[i].w - mu) * rs * (1.f + g.w) + a.w;
        hr[k] = o;
    }
}

// ---------------- 3/5/7/8. GEMM: C = A W^T + bias, fused epilogue ----------------
enum { EPI_BIAS = 0, EPI_GELU = 1, EPI_GATED = 2 };

#define BM 128
#define BN 128
#define BK 8
#define GT 256

template <int EPI>
__global__ void __launch_bounds__(GT, 2)
gemm_kernel(const float* __restrict__ A, const float* __restrict__ W, const float* __restrict__ bias,
            float* __restrict__ C, int M, int N, int K,
            const float* __restrict__ R, const float* __restrict__ mod, int gateOff, int T)
{
    __shared__ __align__(16) float As[2][BK][BM + 4];
    __shared__ __align__(16) float Ws[2][BK][BN + 4];

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int lr = tid >> 1, lk = (tid & 1) * 4;          // loader: row, k offset

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

    auto loadA = [&](int k0) -> float4 {
        const int gr = m0 + lr;
        return (gr < M) ? __ldg(reinterpret_cast<const float4*>(A + (size_t)gr * K + k0 + lk))
                        : make_float4(0.f, 0.f, 0.f, 0.f);
    };
    auto loadW = [&](int k0) -> float4 {
        return __ldg(reinterpret_cast<const float4*>(W + (size_t)(n0 + lr) * K + k0 + lk));
    };
    auto store = [&](float (*S)[BM + 4], float4 v) {
        S[lk][lr] = v.x; S[lk + 1][lr] = v.y; S[lk + 2][lr] = v.z; S[lk + 3][lr] = v.w;
    };

    float4 ra = loadA(0), rw = loadW(0);
    store(As[0], ra); store(Ws[0], rw);
    __syncthreads();

    const int nk = K / BK;
    for (int t = 0; t < nk; ++t) {
        const int cur = t & 1;
        if (t + 1 < nk) { ra = loadA((t + 1) * BK); rw = loadW((t + 1) * BK); }
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[cur][k][64 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&Ws[cur][k][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Ws[cur][k][64 + tx * 4]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (t + 1 < nk) { store(As[cur ^ 1], ra); store(Ws[cur ^ 1], rw); }
        __syncthreads();
    }

    // ---- epilogue ----
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
        if (row >= M) continue;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int col = n0 + half * 64 + tx * 4;
            const float4 bv = *reinterpret_cast<const float4*>(bias + col);
            float v[4] = {acc[i][half * 4 + 0] + bv.x, acc[i][half * 4 + 1] + bv.y,
                          acc[i][half * 4 + 2] + bv.z, acc[i][half * 4 + 3] + bv.w};
            if (EPI == EPI_GELU) {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float z = v[u];
                    v[u] = 0.5f * z * (1.f + tanhf(0.7978845608028654f * (z + 0.044715f * z * z * z)));
                }
            } else if (EPI == EPI_GATED) {
                const int b = row / T;
                const float4 g = *reinterpret_cast<const float4*>(mod + (size_t)b * 6 * D + gateOff + col);
                const float4 r = *reinterpret_cast<const float4*>(R + (size_t)row * N + col);
                v[0] = fmaf(g.x, v[0], r.x); v[1] = fmaf(g.y, v[1], r.y);
                v[2] = fmaf(g.z, v[2], r.z); v[3] = fmaf(g.w, v[3], r.w);
            }
            *reinterpret_cast<float4*>(C + (size_t)row * N + col) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
}

// ---------------- 4. bidirectional attention (FlashAttention-style, FP32) ----------------
// qkv rows: [q (512) | k (512) | v (512)], head h at column h*64 within each part.
#define AQ 32
#define AK 32
#define AW 8
#define AQPW (AQ / AW)
#define APITCH (DH + 4)

__global__ void __launch_bounds__(AW * 32)
attn_kernel(const float* __restrict__ qkv, float* __restrict__ out, int T)
{
    __shared__ __align__(16) float sQ[AQ][APITCH];
    __shared__ __align__(16) float sK[AK][APITCH];
    __shared__ __align__(16) float sV[AK][DH];
    __shared__ __align__(16) float sP[AW][AQPW][AK];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int h = blockIdx.y, b = blockIdx.z;
    const int q0 = blockIdx.x * AQ;
    const size_t base = (size_t)b * T * (3 * D);
    const int ldq = 3 * D;
    const float scale = 0.125f;                               // 1 / sqrt(64)

    for (int idx = tid; idx < AQ * (DH / 4); idx += AW * 32) {
        const int r = idx / (DH / 4), c4 = (idx % (DH / 4)) * 4;
        const int g = q0 + r;
        const float4 v = (g < T) ? *reinterpret_cast<const float4*>(qkv + base + (size_t)g * ldq + h * DH + c4)
                                 : make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(&sQ[r][c4]) = v;
    }

    float m[AQPW], l[AQPW], o[AQPW][2];                       // lane owns dims 2*lane, 2*lane+1
#pragma unroll
    for (int a = 0; a < AQPW; ++a) { m[a] = -INFINITY; l[a] = 0.f; o[a][0] = o[a][1] = 0.f; }

    const int nT = (T + AK - 1) / AK;
    for (int t = 0; t < nT; ++t) {
        const int k0 = t * AK;
        __syncthreads();
        for (int idx = tid; idx < AK * (DH / 4); idx += AW * 32) {
            const int r = idx / (DH / 4), c4 = (idx % (DH / 4)) * 4;
            const int g = k0 + r;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (g < T) {
                const float* rowp = qkv + base + (size_t)g * ldq + h * DH + c4;
                kv = *reinterpret_cast<const float4*>(rowp + D);
                vv = *reinterpret_cast<const float4*>(rowp + 2 * D);
            }
            *reinterpret_cast<float4*>(&sK[r][c4]) = kv;
            *reinterpret_cast<float4*>(&sV[r][c4]) = vv;
        }
        __syncthreads();

        float s[AQPW] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
        for (int c = 0; c < DH; c += 4) {
            const float4 kv = *reinterpret_cast<const float4*>(&sK[lane][c]);
#pragma unroll
            for (int a = 0; a < AQPW; ++a) {
                const float4 qv = *reinterpret_cast<const float4*>(&sQ[warp * AQPW + a][c]);
                s[a] = fmaf(qv.x, kv.x, fmaf(qv.y, kv.y, fmaf(qv.z, kv.z, fmaf(qv.w, kv.w, s[a]))));
            }
        }
        const bool ok = (k0 + lane) < T;
#pragma unroll
        for (int a = 0; a < AQPW; ++a) {
            const float x = ok ? s[a] * scale : -INFINITY;
            float tmax = x;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
            const float mNew = fmaxf(m[a], tmax);              // finite: every tile has >= 1 valid key
            const float p = ok ? expf(x - mNew) : 0.f;
            const float alpha = expf(m[a] - mNew);             // exp(-inf) = 0 on the first tile
            float ps = p;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) ps += __shfl_xor_sync(0xffffffff, ps, off);
            l[a] = l[a] * alpha + ps;
            m[a] = mNew;
            o[a][0] *= alpha; o[a][1] *= alpha;
            sP[warp][a][lane] = p;
        }
        __syncwarp();
#pragma unroll 8
        for (int j = 0; j < AK; ++j) {
            const float2 vv = *reinterpret_cast<const float2*>(&sV[j][2 * lane]);
#pragma unroll
            for (int a = 0; a < AQPW; ++a) {
                const float p = sP[warp][a][j];
                o[a][0] = fmaf(p, vv.x, o[a][0]);
                o[a][1] = fmaf(p, vv.y, o[a][1]);
            }
        }
        __syncwarp();
    }

#pragma unroll
    for (int a = 0; a < AQPW; ++a) {
        const int i = q0 + warp * AQPW + a;
        if (i >= T) continue;
        const float inv = 1.f / l[a];
        *reinterpret_cast<float2*>(out + ((size_t)b * T + i) * D + h * DH + 2 * lane) =
            make_float2(o[a][0] * inv, o[a][1] * inv);
    }
}

// ---------------- host ----------------
static float* g_buf = nullptr;
static size_t g_cap = 0;

template <int EPI>
static void gemm(const float* A, const float* W, const float* bias, float* C, int M, int N, int K,
                 const float* R = nullptr, const float* mod = nullptr, int gateOff = 0, int T = 1)
{
    dim3 grid(N / BN, (M + BM - 1) / BM);
    gemm_kernel<EPI><<<grid, GT>>>(A, W, bias, C, M, N, K, R, mod, gateOff, T);
}

// x, c, output, weights are device pointers
extern "C" void solve(const float* x, const float* c, float* output, const float* weights,
                      int batch_size, int seq_len) {
    const int B = batch_size, T = seq_len, M = B * T;

    // scratch: mod | h | qkv | attn | x1 | u
    const size_t nMod = (size_t)B * 6 * D, nH = (size_t)M * D, nQKV = (size_t)M * 3 * D, nU = (size_t)M * MLP;
    const size_t need = nMod + nH + nQKV + nH + nH + nU;
    if (need > g_cap) {
        if (g_buf) cudaFree(g_buf);
        cudaMalloc(&g_buf, need * sizeof(float));
        g_cap = need;
    }
    float* mod  = g_buf;
    float* h    = mod + nMod;
    float* qkv  = h + nH;
    float* attn = qkv + nQKV;
    float* x1   = attn + nH;
    float* u    = x1 + nH;

    const float* W = weights;
    mod_kernel<<<(int)((nMod * 32 + 255) / 256), 256>>>(c, W + O_WADA, W + O_BADA, mod, B);

    const int lnBlocks = (M * 32 + 255) / 256;
    ln_mod_kernel<<<lnBlocks, 256>>>(x, mod, h, M, T, 0);                       // shift_msa, scale_msa
    gemm<EPI_BIAS>(h, W + O_WQKV, W + O_BQKV, qkv, M, 3 * D, D);

    dim3 ag((T + AQ - 1) / AQ, NH, B);
    attn_kernel<<<ag, AW * 32>>>(qkv, attn, T);

    gemm<EPI_GATED>(attn, W + O_WO, W + O_BO, x1, M, D, D, x, mod, 2 * D, T);     // gate_msa
    ln_mod_kernel<<<lnBlocks, 256>>>(x1, mod, h, M, T, 3);                      // shift_mlp, scale_mlp
    gemm<EPI_GELU>(h, W + O_WFC1, W + O_BFC1, u, M, MLP, D);
    gemm<EPI_GATED>(u, W + O_WFC2, W + O_BFC2, output, M, D, MLP, x1, mod, 5 * D, T);  // gate_mlp
    cudaDeviceSynchronize();
}
