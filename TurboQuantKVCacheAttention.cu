#include <cuda_runtime.h>
#include <math.h>

// TurboQuant KV-cache attention scores:
//   K~_s   = codebook[K_idx_s] @ Pi + c * gamma_s * (sigma_s @ S_mat),   c = sqrt(pi/2) / D
//   scores = Q @ K~^T                                                    (B x S)
//
// Never dequantize the keys. Move the D x D matrices onto the queries instead:
//   Q_b . (y_s @ Pi)        = y_s . (Pi @ Q_b)     =: y_s . Qr_b
//   Q_b . (sigma_s @ S_mat) = sigma_s . (S_mat @ Q_b) =: sigma_s . Qs_b
// so
//   scores[b, s] = sum_d codebook[K_idx[s, d]] * Qr[b, d] + c * gamma_s * sum_d sigma[s, d] * Qs[b, d].
//
// Kernel 1 (tiny): Qr, Qs = Pi @ Q_b, S_mat @ Q_b, stored transposed [d][b] and
//   zero-padded to Dp x 32 so the main kernel needs no bounds checks on them.
// Kernel 2: a skinny GEMM, (S x 2D) bytes against (2D x B) floats. Each block
//   handles 128 keys x 32 queries; 256 threads each own 4 keys x 4 queries with
//   two accumulators (MSE part, QJL part). Per 32-wide d chunk, the uint8
//   indices are decoded through a smem codebook and the int8 signs converted to
//   +-1, both stored transposed [d][key] for float4 reads. The gamma scaling
//   is applied once in the epilogue. Everything is FP32.

#define TK 128          // keys per block
#define TB 32           // queries per block (B <= 32)
#define BD 32           // d chunk
#define THREADS 256

// ---------------- Kernel 1: rotate the queries ----------------
__global__ void rotate_q_kernel(const float* __restrict__ Q, const float* __restrict__ Pi,
                                const float* __restrict__ Smat, float* __restrict__ QrT,
                                float* __restrict__ QsT, int B, int D, int Dp)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;      // over Dp * TB
    if (idx >= Dp * TB) return;
    const int d = idx / TB, b = idx % TB;
    float r = 0.f, s = 0.f;
    if (d < D && b < B) {
        const float* q  = Q + (size_t)b * D;
        const float* pr = Pi + (size_t)d * D;
        const float* sr = Smat + (size_t)d * D;
        for (int e = 0; e < D; ++e) {
            const float qe = __ldg(q + e);
            r = fmaf(__ldg(pr + e), qe, r);
            s = fmaf(__ldg(sr + e), qe, s);
        }
    }
    QrT[idx] = r;
    QsT[idx] = s;
}

// ---------------- Kernel 2: scores ----------------
__global__ void __launch_bounds__(THREADS)
tq_scores_kernel(const unsigned char* __restrict__ K_idx, const signed char* __restrict__ sgn,
                 const float* __restrict__ gamma, const float* __restrict__ codebook,
                 const float* __restrict__ QrT, const float* __restrict__ QsT,
                 float* __restrict__ scores, int B, int S, int D, int C, int Dp, float c, bool vec16)
{
    __shared__ float cb[256];
    __shared__ __align__(16) float As[BD][TK];       // codebook values, [d][key]
    __shared__ __align__(16) float Ss[BD][TK];       // signs as +-1,     [d][key]
    __shared__ __align__(16) float Qr[BD][TB];
    __shared__ __align__(16) float Qs[BD][TB];

    const int tid = threadIdx.x;
    const int s0 = blockIdx.x * TK;
    for (int i = tid; i < C; i += THREADS) cb[i] = codebook[i];

    const int tk = tid & 31, tq = tid >> 5;          // keys tk*4.., queries tq*4.. (warp-uniform)
    float am[4][4], aq[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) am[i][j] = aq[i][j] = 0.f;

    // Loader role: key lk (consecutive threads -> consecutive keys), 16-wide d half lh.
    const int lk = tid & (TK - 1), lh = tid >> 7;
    const int gs = s0 + lk;

    for (int d0 = 0; d0 < Dp; d0 += BD) {
        __syncthreads();                              // previous chunk consumed (and cb ready)
        const int dd = d0 + lh * 16;
        unsigned char ib[16];
        signed char   sb[16];
        if (vec16 && gs < S && dd < D) {              // D % 16 == 0: whole 16-byte run is valid
            const uint4 iv = __ldg(reinterpret_cast<const uint4*>(K_idx + (size_t)gs * D + dd));
            const uint4 sv = __ldg(reinterpret_cast<const uint4*>(sgn   + (size_t)gs * D + dd));
            *reinterpret_cast<uint4*>(ib) = iv;
            *reinterpret_cast<uint4*>(sb) = sv;
        } else {
#pragma unroll
            for (int u = 0; u < 16; ++u) {
                const bool ok = gs < S && dd + u < D;
                ib[u] = ok ? __ldg(K_idx + (size_t)gs * D + dd + u) : 0;
                sb[u] = ok ? __ldg(sgn   + (size_t)gs * D + dd + u) : 0;
            }
        }
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const bool ok = gs < S && dd + u < D;
            As[lh * 16 + u][lk] = ok ? cb[ib[u]] : 0.f;
            Ss[lh * 16 + u][lk] = (float)sb[u];       // 0 for padding
        }
        for (int i = tid; i < BD * TB; i += THREADS) {
            (&Qr[0][0])[i] = QrT[(size_t)d0 * TB + i];
            (&Qs[0][0])[i] = QsT[(size_t)d0 * TB + i];
        }
        __syncthreads();

#pragma unroll 8
        for (int d = 0; d < BD; ++d) {
            const float4 a  = *reinterpret_cast<const float4*>(&As[d][tk * 4]);
            const float4 sg = *reinterpret_cast<const float4*>(&Ss[d][tk * 4]);
            const float4 qr = *reinterpret_cast<const float4*>(&Qr[d][tq * 4]);
            const float4 qs = *reinterpret_cast<const float4*>(&Qs[d][tq * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w};
            const float sv[4] = {sg.x, sg.y, sg.z, sg.w};
            const float rv[4] = {qr.x, qr.y, qr.z, qr.w};
            const float qv[4] = {qs.x, qs.y, qs.z, qs.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    am[i][j] = fmaf(av[i], rv[j], am[i][j]);
                    aq[i][j] = fmaf(sv[i], qv[j], aq[i][j]);
                }
        }
    }

    // ---- epilogue: score = mse + c * gamma_s * qjl ----
    const int sk = s0 + tk * 4;
    float g[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) g[i] = (sk + i < S) ? c * __ldg(gamma + sk + i) : 0.f;
    const bool vecOut = ((S & 3) == 0) && (sk + 3 < S);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int b = tq * 4 + j;
        if (b >= B) break;
        float r[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) r[i] = fmaf(g[i], aq[i][j], am[i][j]);
        float* dst = scores + (size_t)b * S + sk;
        if (vecOut) {
            *reinterpret_cast<float4*>(dst) = make_float4(r[0], r[1], r[2], r[3]);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) if (sk + i < S) dst[i] = r[i];
        }
    }
}

// ---------------- host ----------------
static float* g_q = nullptr;
static size_t g_qCap = 0;

// Q, K_idx, qjl_signs, gamma, Pi, S_mat, codebook, scores are device pointers
extern "C" void solve(const float* Q, const unsigned char* K_idx, const signed char* qjl_signs,
                      const float* gamma, const float* Pi, const float* S_mat,
                      const float* codebook, float* scores, int B, int S, int D, int C) {
    const int Dp = (D + BD - 1) / BD * BD;
    const size_t need = 2 * (size_t)Dp * TB;
    if (need > g_qCap) {
        if (g_q) cudaFree(g_q);
        cudaMalloc(&g_q, need * sizeof(float));
        g_qCap = need;
    }
    float* QrT = g_q;
    float* QsT = g_q + (size_t)Dp * TB;

    rotate_q_kernel<<<(Dp * TB + 255) / 256, 256>>>(Q, Pi, S_mat, QrT, QsT, B, D, Dp);

    const bool vec16 = (D % 16 == 0) &&
                       (((reinterpret_cast<size_t>(K_idx) | reinterpret_cast<size_t>(qjl_signs)) & 15) == 0);
    const float c = sqrtf((float)M_PI / 2.0f) / (float)D;
    tq_scores_kernel<<<(S + TK - 1) / TK, THREADS>>>(K_idx, qjl_signs, gamma, codebook, QrT, QsT,
                                                     scores, B, S, D, C, Dp, c, vec16);
    cudaDeviceSynchronize();
}
