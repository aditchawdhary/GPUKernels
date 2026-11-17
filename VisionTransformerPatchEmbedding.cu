/*
Vision Transformer Patch Embedding
images (B, C, H, W); patch_weight (D, C, P, P); patch_bias (D); cls_token (D);
pos_embed (N+1, D); output (B, N+1, D), N = (H/P)(W/P), float32.
Spec from the reference implementation (leetgpu-challenges #118), tolerance 1e-4:
  patch vector of patch n = (gy, gx): x[c, gy*P + py, gx*P + px] in (c, py, px) order
  output[b, 0]   = cls_token + pos_embed[0]
  output[b, 1+n] = patch_vec(b, n) . patch_weight[d, :] + bias[d] + pos_embed[1+n, d]
Performance at B = 64, C = 3, H = W = 224, P = 16, D = 768 (ViT-B/16).
*/

/*
Approach: one implicit-GEMM kernel plus a tiny CLS kernel.

 GEMM view: rows m = (b, n) (B*N = 12,544 at the benchmark), K = C*P*P (768),
 columns = D (768). The weight is (D, K) row-major, so this is C = A W^T with both
 operands K-contiguous.
 - The A loader gathers patch pixels straight from the image (im2col on the fly). A
   thread decodes its tile row's (b, gy, gx) once. Each 4-wide k group is (c, py, px)
   with px % 4 == 0 when P % 4 == 0, which is 4 adjacent pixels of one image row, so
   one aligned float4 load. Otherwise guarded scalar gathers. No patches buffer is
   ever written.
 - FP32 register-tiled SGEMM: 128x128 tiles (64x64 when that gives too few blocks),
   BK = 8, 256 threads (8x8 or 4x4 outputs each), both tiles k-major in
   double-buffered shared memory with register prefetch.
 - Epilogue: + bias[d] + pos_embed[1 + n, d], stored to output row b*(N+1) + 1 + n
   (float4 when D % 4 == 0).
 - cls_rows: output[b, 0, :] = cls + pos_embed[0].
 At the benchmark: 98 x 6 = 588 blocks, 14.8 GFLOP of FP32.
 Edge cases: any P (e.g. 5, 6), any K or D. Rows, columns and k past the ends are
 zero-filled and never stored.
*/

#include <cuda_runtime.h>

#define GT 256
#define GBK 8

template <int BM, int BN, bool VEC>
__global__ void __launch_bounds__(GT)
patch_gemm(const float* __restrict__ img, const float* __restrict__ Wt,
           const float* __restrict__ bias, const float* __restrict__ pos, float* __restrict__ out,
           int C, int H, int Wd, int P, int D, int gw, int Np, int Mtot, int K)
{
    constexpr int TM = BM / 16, TN = BN / 16, HM = TM / 4, HN = TN / 4;
    constexpr int NA = (BM * 2 + GT - 1) / GT, NB = (BN * 2 + GT - 1) / GT;
    __shared__ __align__(16) float As[2][GBK][BM];
    __shared__ __align__(16) float Bs[2][GBK][BN];

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int PP = P * P;

    // per-thread A rows: image base of the patch (top-left pixel, channel 0)
    const float* aBase[NA];
    bool aOk[NA];
    #pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int f = tid + i * GT;
        const int m = m0 + (f >> 1);
        aOk[i] = (f < BM * 2) && (m < Mtot);
        aBase[i] = img;
        if (aOk[i]) {
            const int b = m / Np, n = m - b * Np, gy = n / gw, gx = n - gy * gw;
            aBase[i] = img + ((size_t)b * C * H + (size_t)gy * P) * Wd + (size_t)gx * P;
        }
    }

    float4 ra[NA], rb[NB];
    auto load = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < NA; ++i) {
            const int f = tid + i * GT;
            if (f < BM * 2) {
                const int kk = k0 + (f & 1) * 4;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (aOk[i]) {
                    if (VEC) {                          // 4 adjacent pixels of one image row
                        if (kk < K) {
                            const int c = kk / PP, rem = kk - c * PP, py = rem / P, px = rem - py * P;
                            v = __ldg(reinterpret_cast<const float4*>(aBase[i] + ((size_t)c * H + py) * Wd + px));
                        }
                    } else {
                        float e[4];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int k = kk + j;
                            e[j] = 0.f;
                            if (k < K) {
                                const int c = k / PP, rem = k - c * PP, py = rem / P, px = rem - py * P;
                                e[j] = __ldg(aBase[i] + ((size_t)c * H + py) * Wd + px);
                            }
                        }
                        v = make_float4(e[0], e[1], e[2], e[3]);
                    }
                }
                ra[i] = v;
            }
        }
        #pragma unroll
        for (int i = 0; i < NB; ++i) {
            const int f = tid + i * GT;
            if (f < BN * 2) {
                const int r = f >> 1, kk = k0 + (f & 1) * 4, n = n0 + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (n < D) {
                    const float* wr = Wt + (size_t)n * K;
                    if (VEC) {
                        if (kk < K) v = __ldg(reinterpret_cast<const float4*>(wr + kk));
                    } else {
                        v.x = (kk     < K) ? __ldg(wr + kk)     : 0.f;
                        v.y = (kk + 1 < K) ? __ldg(wr + kk + 1) : 0.f;
                        v.z = (kk + 2 < K) ? __ldg(wr + kk + 2) : 0.f;
                        v.w = (kk + 3 < K) ? __ldg(wr + kk + 3) : 0.f;
                    }
                }
                rb[i] = v;
            }
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int i = 0; i < NA; ++i) {
            const int f = tid + i * GT;
            if (f < BM * 2) {
                const int r = f >> 1, kk = (f & 1) * 4;
                As[buf][kk][r] = ra[i].x; As[buf][kk + 1][r] = ra[i].y;
                As[buf][kk + 2][r] = ra[i].z; As[buf][kk + 3][r] = ra[i].w;
            }
        }
        #pragma unroll
        for (int i = 0; i < NB; ++i) {
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

    load(0); store(0);
    __syncthreads();
    const int KT = (K + GBK - 1) / GBK;
    for (int kt = 0; kt < KT; ++kt) {
        const int cur = kt & 1;
        const bool more = kt + 1 < KT;
        if (more) load((kt + 1) * GBK);
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

    // ---- epilogue: + bias + pos_embed[1+n], write token row
    const bool vecD = (D & 3) == 0;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int m = m0 + (i / 4) * (BM / 2) + ty * 4 + (i & 3);
        if (m >= Mtot) continue;
        const int b = m / Np, n = m - b * Np;
        float* orow = out + ((size_t)b * (Np + 1) + 1 + n) * D;
        const float* prow = pos + (size_t)(1 + n) * D;
        #pragma unroll
        for (int h = 0; h < HN; ++h) {
            const int c = n0 + h * (BN / 2) + tx * 4;
            if (vecD) {
                if (c < D) {
                    const float4 bb = __ldg(reinterpret_cast<const float4*>(bias + c));
                    const float4 pp = __ldg(reinterpret_cast<const float4*>(prow + c));
                    *reinterpret_cast<float4*>(orow + c) =
                        make_float4(acc[i][h * 4] + bb.x + pp.x, acc[i][h * 4 + 1] + bb.y + pp.y,
                                    acc[i][h * 4 + 2] + bb.z + pp.z, acc[i][h * 4 + 3] + bb.w + pp.w);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (c + j < D) orow[c + j] = acc[i][h * 4 + j] + __ldg(bias + c + j) + __ldg(prow + c + j);
            }
        }
    }
}

__global__ void cls_rows(const float* __restrict__ cls, const float* __restrict__ pos,
                         float* __restrict__ out, int B, int Np, int D)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;
    const int b = idx / D, d = idx - b * D;
    out[(size_t)b * (Np + 1) * D + d] = cls[d] + pos[d];
}

template <int BM, int BN>
static void launch(const float* img, const float* w, const float* bias, const float* pos, float* out,
                   int C, int H, int Wd, int P, int D, int gw, int Np, int Mtot, int K, bool vec)
{
    dim3 grid((D + BN - 1) / BN, (Mtot + BM - 1) / BM);
    if (vec) patch_gemm<BM, BN, true ><<<grid, GT>>>(img, w, bias, pos, out, C, H, Wd, P, D, gw, Np, Mtot, K);
    else     patch_gemm<BM, BN, false><<<grid, GT>>>(img, w, bias, pos, out, C, H, Wd, P, D, gw, Np, Mtot, K);
}

// images, patch_weight, patch_bias, cls_token, pos_embed, output are device pointers
extern "C" void solve(const float* images, const float* patch_weight, const float* patch_bias,
                      const float* cls_token, const float* pos_embed, float* output, int B, int C,
                      int H, int W, int P, int D)
{
    if (B <= 0 || C <= 0 || H <= 0 || W <= 0 || P <= 0 || D <= 0) return;
    const int gh = H / P, gw = W / P, Np = gh * gw;
    const int Mtot = B * Np, K = C * P * P;

    cls_rows<<<(B * D + 255) / 256, 256>>>(cls_token, pos_embed, output, B, Np, D);
    if (Mtot > 0) {
        int dev = 0, sms = 80;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        const bool vec = (P % 4) == 0;                 // => W % 4 == 0 and K % 4 == 0
        const long long big = (long long)((Mtot + 127) / 128) * ((D + 127) / 128);
        if (big >= 2LL * sms) launch<128, 128>(images, patch_weight, patch_bias, pos_embed, output, C, H, W, P, D, gw, Np, Mtot, K, vec);
        else                  launch<64, 64>(images, patch_weight, patch_bias, pos_embed, output, C, H, W, P, D, gw, Np, Mtot, K, vec);
    }
    cudaDeviceSynchronize();
}
