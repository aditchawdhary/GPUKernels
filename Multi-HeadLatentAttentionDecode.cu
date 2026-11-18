/*
Multi-Head Latent Attention (MLA) Decode — DeepSeek-V2/V3 style, weight-absorbed.
q (H, hd + rd); kv_cache (S, R + rd) = [c_kv | k_pe]; W_UK (H, hd, R); W_UV (H, R, hd);
output (H, hd). Spec from the reference implementation (leetgpu-challenges #114),
tolerance 1e-3:
  q_latent[h] = q_nope[h] @ W_UK[h]                            (R)
  score[h,s]  = (q_latent[h] . c_kv[s] + q_pe[h] . k_pe[s]) / sqrt(hd + rd)
  latent[h]   = softmax_s(score[h]) @ c_kv                       (R)
  out[h]      = latent[h] @ W_UV[h]                               (hd)
Constraints: H ≤ 128, S ≤ 8192, R ≤ 512, hd ≤ 128, rd ≤ 64.
Performance at H = 128, S = 4096, R = 512, hd = 128, rd = 64.
*/

/*
Approach (3 launches, FP32):
 1. absorb_q    one block per head. qa[h] = [q_nope[h] @ W_UK[h], q_pe[h]] * scale * log2(e).
                Threads run over r (coalesced reads of W_UK[h][d][r]) and loop over d.
                Because the score is exactly qa[h] . kv_cache[s] over all R + rd
                columns, the cache is used as stored and never split or expanded.
 2. mla_attn    flash decoding with heads as the query rows. All heads share the
                same cache, so a 32-head tile reuses every cache tile 32 times.
                grid = (head tiles of 32, sequence splits of 128 keys, output-column
                groups of 128). For each 32-key tile: S = qa . kv over 64-wide
                chunks of the R + rd score columns, online softmax (exp2), then
                O += P c_kv over this block's 128 latent columns. Splitting the
                output columns across blocks recomputes the cheap scores (R + rd ≈
                576 MACs per pair) but keeps only 32 accumulators per thread and
                gives 4 x 32 x 4 = 512 blocks at the benchmark.
                Each block writes unnormalized partials (m, l, O) per (split, head).
 3. combine_up  one block per head: merges the splits by log-sum-exp
                (M = max m, weights 2^(m - M)), normalizes into latent[R] in shared
                memory, then out = latent @ W_UV[h] as a GEMV. Threads are
                (column j, r-group); reads of W_UV[h][r][j] are coalesced over j, and
                the r-groups are reduced through shared memory.
Tile machinery as in the other attention kernels: 32 x 32 tiles, 128 threads
(16 x 8), each thread owns 4 rows; row operand [d][r] swizzled in float4 units,
column operand [d][k] XOR-swizzled, probabilities [k][r], V chunk row-major.
20 KB of shared memory.
Traffic at the benchmark: W_UK and W_UV are 32 MB each (read once), the cache is
9.4 MB (about 4 reads, mostly from L2), and attention is about 2.5 GFLOP, so the
total time is tens of µs.
*/

#include <cuda_runtime.h>
#include <math.h>

#define TS 32
#define DC 64
#define THREADS 128
#define OG_COLS 128                          // latent columns per block (2 chunks)
#define NCO (OG_COLS / DC)
#define KPS 128                              // keys per sequence split
#define MAXR 512

__device__ __forceinline__ int col_idx(int d, int c) { return d * TS + (c ^ (d & 31)); }
__device__ __forceinline__ int t32_idx(int a, int r) { return a * 32 + ((((r >> 2) ^ (a & 7))) << 2) + (r & 3); }

// ======================================================================== 1. absorb
__global__ void absorb_q(const float* __restrict__ q, const float* __restrict__ Wuk, float* __restrict__ qa,
                         int hd, int rd, int R, float c2)
{
    const int h = blockIdx.x;
    const int DQ = hd + rd, DS = R + rd;
    const float* qh = q + (size_t)h * DQ;
    const float* W = Wuk + (size_t)h * hd * R;
    for (int r = threadIdx.x; r < R; r += blockDim.x) {
        float acc = 0.f;
        for (int d = 0; d < hd; ++d) acc = fmaf(__ldg(qh + d), __ldg(W + (size_t)d * R + r), acc);
        qa[(size_t)h * DS + r] = acc * c2;
    }
    for (int j = threadIdx.x; j < rd; j += blockDim.x)
        qa[(size_t)h * DS + R + j] = __ldg(qh + hd + j) * c2;
}

// ======================================================================== 2. attention
__global__ void __launch_bounds__(THREADS)
mla_attn(const float* __restrict__ qa, const float* __restrict__ kv, float* __restrict__ part,
         int H, int S, int R, int rd)
{
    __shared__ __align__(16) float A[DC * TS], B[DC * TS], Pm[TS * TS];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int h0 = blockIdx.x * TS, sp = blockIdx.y, og = blockIdx.z;
    const int DS = R + rd;
    const int s0 = sp * KPS, s1 = min(s0 + KPS, S);
    const int nScoreCh = (DS + DC - 1) / DC;

    float o[NCO][4][4], m[4], l[4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        m[a] = -INFINITY; l[a] = 0.f;
        #pragma unroll
        for (int c = 0; c < NCO; ++c)
            #pragma unroll
            for (int j = 0; j < 4; ++j) o[c][a][j] = 0.f;
    }

    for (int k0 = s0; k0 < s1; k0 += TS) {
        float s[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
        for (int c = 0; c < nScoreCh; ++c) {
            const int db = c * DC, dw = min(DC, DS - db);
            __syncthreads();
            for (int e = tid; e < TS * DC; e += THREADS) {
                const int r = e >> 6, dd = e & 63;
                const int h = h0 + r, k = k0 + r;
                A[t32_idx(dd, r)] = (h < H && dd < dw) ? __ldg(qa + (size_t)h * DS + db + dd) : 0.f;
                B[col_idx(dd, r)] = (k < s1 && dd < dw) ? __ldg(kv + (size_t)k * DS + db + dd) : 0.f;
            }
            __syncthreads();
            #pragma unroll 4
            for (int dd = 0; dd < dw; ++dd) {
                const float4 av = *reinterpret_cast<const float4*>(&A[t32_idx(dd, ty * 4)]);
                const float* br = &B[dd * TS];
                const int sw = dd & 31;
                const float b0 = br[tx ^ sw], b1 = br[(tx + 16) ^ sw];
                s[0][0] = fmaf(av.x, b0, s[0][0]); s[0][1] = fmaf(av.x, b1, s[0][1]);
                s[1][0] = fmaf(av.y, b0, s[1][0]); s[1][1] = fmaf(av.y, b1, s[1][1]);
                s[2][0] = fmaf(av.z, b0, s[2][0]); s[2][1] = fmaf(av.z, b1, s[2][1]);
                s[3][0] = fmaf(av.w, b0, s[3][0]); s[3][1] = fmaf(av.w, b1, s[3][1]);
            }
        }

        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            float mx = -INFINITY;
            #pragma unroll
            for (int b = 0; b < 2; ++b) {
                if (k0 + tx + 16 * b >= s1) s[a][b] = -INFINITY;
                mx = fmaxf(mx, s[a][b]);
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            const float mNew = fmaxf(m[a], mx);            // finite: tile has ≥ 1 key
            const float alpha = exp2f(m[a] - mNew);
            float sum = 0.f;
            #pragma unroll
            for (int b = 0; b < 2; ++b) { s[a][b] = exp2f(s[a][b] - mNew); sum += s[a][b]; }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            m[a] = mNew;
            l[a] = l[a] * alpha + sum;
            #pragma unroll
            for (int c = 0; c < NCO; ++c)
                #pragma unroll
                for (int j = 0; j < 4; ++j) o[c][a][j] *= alpha;
            #pragma unroll
            for (int b = 0; b < 2; ++b) Pm[t32_idx(tx + 16 * b, ty * 4 + a)] = s[a][b];
        }

        #pragma unroll
        for (int c = 0; c < NCO; ++c) {
            const int cb = og * OG_COLS + c * DC;          // latent column base
            const int cw = min(DC, R - cb);
            if (cw <= 0) break;
            __syncthreads();                               // score reads done, Pm ready
            for (int e = tid; e < TS * DC; e += THREADS) {
                const int r = e >> 6, dd = e & 63, k = k0 + r;
                B[r * DC + dd] = (k < s1 && dd < cw) ? __ldg(kv + (size_t)k * DS + cb + dd) : 0.f;
            }
            __syncthreads();
            #pragma unroll 8
            for (int k = 0; k < TS; ++k) {
                const float4 p = *reinterpret_cast<const float4*>(&Pm[t32_idx(k, ty * 4)]);
                const float4 v = *reinterpret_cast<const float4*>(&B[k * DC + tx * 4]);
                const float pa[4] = {p.x, p.y, p.z, p.w};
                #pragma unroll
                for (int a = 0; a < 4; ++a) {
                    o[c][a][0] = fmaf(pa[a], v.x, o[c][a][0]); o[c][a][1] = fmaf(pa[a], v.y, o[c][a][1]);
                    o[c][a][2] = fmaf(pa[a], v.z, o[c][a][2]); o[c][a][3] = fmaf(pa[a], v.w, o[c][a][3]);
                }
            }
        }
    }

    // ---- unnormalized partials: [m, l, O(R)] per (split, head)
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int h = h0 + ty * 4 + a;
        if (h >= H) continue;
        float* p = part + ((size_t)sp * H + h) * (R + 2);
        if (og == 0 && tx == 0) { p[0] = m[a]; p[1] = l[a]; }
        #pragma unroll
        for (int c = 0; c < NCO; ++c)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = og * OG_COLS + c * DC + tx * 4 + j;
                if (col < R) p[2 + col] = o[c][a][j];
            }
    }
}

// ======================================================================== 3. combine + W_UV
#define CT 256
__global__ void __launch_bounds__(CT)
combine_up(const float* __restrict__ part, const float* __restrict__ Wuv, float* __restrict__ out,
           int H, int R, int hd, int nSplit)
{
    __shared__ float lat[MAXR];
    __shared__ float red[CT];
    __shared__ float sM, sInvL;
    const int h = blockIdx.x, tid = threadIdx.x;

    if (tid == 0) {
        float M = -INFINITY;
        for (int s = 0; s < nSplit; ++s) M = fmaxf(M, part[((size_t)s * H + h) * (R + 2)]);
        float L = 0.f;
        for (int s = 0; s < nSplit; ++s) {
            const float* p = part + ((size_t)s * H + h) * (R + 2);
            L += p[1] * exp2f(p[0] - M);
        }
        sM = M; sInvL = 1.f / L;
    }
    __syncthreads();
    const float M = sM, invL = sInvL;
    for (int r = tid; r < R; r += CT) {
        float acc = 0.f;
        for (int s = 0; s < nSplit; ++s) {
            const float* p = part + ((size_t)s * H + h) * (R + 2);
            acc = fmaf(p[2 + r], exp2f(p[0] - M), acc);
        }
        lat[r] = acc * invL;
    }
    __syncthreads();

    // out[h][j] = sum_r lat[r] * W_UV[h][r][j];   threads = (j, r-group)
    const int G = CT / hd;                          // hd ≤ 128 -> G ≥ 2
    const int j = tid % hd, g = tid / hd;
    float acc = 0.f;
    if (g < G) {
        const float* W = Wuv + (size_t)h * R * hd + j;
        for (int r = g; r < R; r += G) acc = fmaf(lat[r], __ldg(W + (size_t)r * hd), acc);
    }
    red[tid] = acc;
    __syncthreads();
    if (tid < hd) {
        float s = 0.f;
        for (int gg = 0; gg < G; ++gg) s += red[gg * hd + tid];
        out[(size_t)h * hd + tid] = s;
    }
}

// ======================================================================== host
static float* g_ws = nullptr;
static size_t g_cap = 0;

// q, kv_cache, W_UK, W_UV, output are device pointers
extern "C" void solve(const float* q, const float* kv_cache, const float* W_UK, const float* W_UV,
                      float* output, int num_heads, int seq_len, int kv_lora_rank, int head_dim,
                      int rope_dim)
{
    const int H = num_heads, S = seq_len, R = kv_lora_rank, hd = head_dim, rd = rope_dim;
    if (H <= 0 || S <= 0 || R <= 0 || hd <= 0) return;
    const int DS = R + rd;
    const int nSplit = (S + KPS - 1) / KPS;
    const int nOG = (R + OG_COLS - 1) / OG_COLS;

    const size_t need = (size_t)H * DS + (size_t)nSplit * H * (R + 2);
    if (need > g_cap) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, need * sizeof(float));
        g_cap = need;
    }
    float* qa = g_ws;
    float* part = g_ws + (size_t)H * DS;

    const float c2 = (float)(1.4426950408889634 / sqrt((double)(hd + rd)));   // log2(e)/sqrt(hd+rd)
    absorb_q<<<H, 256>>>(q, W_UK, qa, hd, rd, R, c2);
    mla_attn<<<dim3((H + TS - 1) / TS, nSplit, nOG), THREADS>>>(qa, kv_cache, part, H, S, R, rd);
    combine_up<<<H, CT>>>(part, W_UV, output, H, R, hd, nSplit);
    cudaDeviceSynchronize();
}
