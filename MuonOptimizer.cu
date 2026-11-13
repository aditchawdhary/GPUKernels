#include <cuda_runtime.h>
#include <math.h>

// Muon update: V = momentum*Buf + G ; X = V / (||V||_F + 1e-7)
// 5 x { A = X X^T ; B = b*A + c*A A ; X = a*X + B X }   (on X^T when M > N)
// output = X * sqrt(max(M, N))
//
// Layout: the iteration runs on an r x c matrix with r = min(M,N) (the Gram dim) and c = max(M,N).
// All GEMMs are fp32 SIMT with fp32 FMA accumulation (no TF32 / tensor cores).
//
// GEMM tricks:
//   * A = X X^T and A A (= A A^T, since A is symmetric) are "NT" GEMMs whose outputs are
//     symmetric: only the upper-triangular tiles are computed and mirrored (1/3 fewer FLOPs
//     per step). Each element is accumulated over k in the same order as its mirror, so the
//     results are exactly symmetric.
//   * The final step's epilogue applies the sqrt(max(M,N)) scale and the transpose back,
//     writing straight into `output`.

#define TILE 64          // output tile is TILE x TILE
#define BK 16            // reduction depth per shared-memory stage
#define NT 256           // threads per block, each computes a 4x4 micro-tile
#define PAD 4            // shared padding (keeps float4 alignment, breaks bank conflicts)

static const float NS_A = 3.4445f, NS_B = -4.7750f, NS_C = 2.0315f;

// ---------------------------------------------------------------------------------------------
// Frobenius norm (two-level: per row of V, then over rows), accumulated in double.
// ---------------------------------------------------------------------------------------------
__global__ void row_sumsq(const float* __restrict__ G, const float* __restrict__ Buf,
                          double* __restrict__ rowsum, int N, float momentum) {
    int row = blockIdx.x;
    const float* g = G + (size_t)row * N;
    const float* b = Buf + (size_t)row * N;
    double s = 0.0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float v = __fadd_rn(__fmul_rn(momentum, b[j]), g[j]);
        s += (double)v * (double)v;
    }
    __shared__ double sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) rowsum[row] = sh[0];
}

__global__ void finish_norm(const double* __restrict__ rowsum, int M, float* __restrict__ denom) {
    double s = 0.0;
    for (int i = threadIdx.x; i < M; i += blockDim.x) s += rowsum[i];
    __shared__ double sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) *denom = __fadd_rn((float)sqrt(sh[0]), 1e-7f);
}

// X (r x c) = V / denom, transposed when M > N.
__global__ void build_x(const float* __restrict__ G, const float* __restrict__ Buf,
                        const float* __restrict__ denom, float* __restrict__ X,
                        int M, int N, float momentum, bool transposed) {
    size_t total = (size_t)M * N;
    float d = *denom;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        // idx walks the destination (r x c) row-major for coalesced writes
        int c = transposed ? M : N;
        int i = (int)(idx / c), j = (int)(idx % c);
        size_t src = transposed ? (size_t)j * N + i : idx;
        float v = __fadd_rn(__fmul_rn(momentum, Buf[src]), G[src]);
        X[idx] = __fdiv_rn(v, d);
    }
}

// ---------------------------------------------------------------------------------------------
// Tiled fp32 GEMM with epilogue.
//   acc[i][j] = sum_k A[i][k] * (TB ? B[j][k] : B[k][j])
//   val       = beta * D[i][j] + alpha * acc      (D read only if HAS_D)
//   val      *= post
//   C[i*ldc + j] = val   (or C[j*ldc + i] if trans_out)
// A: R x K row-major.  B: (TB ? Cn x K : K x Cn) row-major.  D: R x Cn row-major.
// SYM: output is symmetric (R == Cn); compute only tiles with ti <= tj and mirror.
// ---------------------------------------------------------------------------------------------
template <bool TB, bool SYM, bool HAS_D>
__global__ void __launch_bounds__(NT)
gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, const float* __restrict__ D,
            float* __restrict__ C, int R, int Cn, int K, float alpha, float beta, float post,
            bool trans_out, int ldc) {
    int ti, tj;
    if (SYM) {
        int T = (R + TILE - 1) / TILE;
        int idx = blockIdx.x;
        ti = 0;
        while (idx >= T - ti) { idx -= T - ti; ti++; }
        tj = ti + idx;
    } else {
        ti = blockIdx.y;
        tj = blockIdx.x;
    }
    const int r0 = ti * TILE, c0 = tj * TILE;

    __shared__ __align__(16) float As[2][BK][TILE + PAD];
    __shared__ __align__(16) float Bs[2][BK][TILE + PAD];

    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;

    // load mappings
    const int ak = tid % BK, arow = tid / BK;         // A-like tiles: k fastest (coalesced on k)
    const int bcol = tid % TILE, bk = tid / TILE;     // NN B tile: col fastest

    float ra[4], rb[4];
    auto load = [&](int k0) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = r0 + arow + r * 16, k = k0 + ak;
            ra[r] = (row < R && k < K) ? __ldg(&A[(size_t)row * K + k]) : 0.0f;
        }
        if (TB) {
#pragma unroll
            for (int r = 0; r < 4; r++) {
                int row = c0 + arow + r * 16, k = k0 + ak;
                rb[r] = (row < Cn && k < K) ? __ldg(&B[(size_t)row * K + k]) : 0.0f;
            }
        } else {
#pragma unroll
            for (int r = 0; r < 4; r++) {
                int k = k0 + bk + r * 4, col = c0 + bcol;
                rb[r] = (k < K && col < Cn) ? __ldg(&B[(size_t)k * Cn + col]) : 0.0f;
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int r = 0; r < 4; r++) As[buf][ak][arow + r * 16] = ra[r];
        if (TB) {
#pragma unroll
            for (int r = 0; r < 4; r++) Bs[buf][ak][arow + r * 16] = rb[r];
        } else {
#pragma unroll
            for (int r = 0; r < 4; r++) Bs[buf][bk + r * 4][bcol] = rb[r];
        }
    };

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    load(0);
    store(0);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        bool more = k0 + BK < K;
        if (more) load(k0 + BK);              // global prefetch overlaps compute

#pragma unroll
        for (int k = 0; k < BK; k++) {
            float4 a4 = *reinterpret_cast<const float4*>(&As[buf][k][ty * 4]);
            float4 b4 = *reinterpret_cast<const float4*>(&Bs[buf][k][tx * 4]);
            float a[4] = {a4.x, a4.y, a4.z, a4.w};
            float b[4] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }

        if (more) store(buf ^ 1);             // other buffer: no hazard with current reads
        __syncthreads();
        buf ^= 1;
    }

    // epilogue
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = r0 + ty * 4 + i;
        if (row >= R) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = c0 + tx * 4 + j;
            if (col >= Cn) continue;
            float v = __fmul_rn(alpha, acc[i][j]);
            if (HAS_D) v = __fadd_rn(__fmul_rn(beta, D[(size_t)row * Cn + col]), v);
            v = __fmul_rn(v, post);
            if (trans_out) C[(size_t)col * ldc + row] = v;
            else           C[(size_t)row * ldc + col] = v;
            if (SYM && ti != tj) C[(size_t)col * ldc + row] = v;   // mirror (D symmetric too)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Scratch memory (allocated once, grown if needed)
// ---------------------------------------------------------------------------------------------
static char*  g_scratch = nullptr;
static size_t g_scratch_bytes = 0;

static void* scratch(size_t bytes) {
    if (bytes > g_scratch_bytes) {
        if (g_scratch) cudaFree(g_scratch);
        cudaMalloc(&g_scratch, bytes);
        g_scratch_bytes = bytes;
    }
    return g_scratch;
}

static size_t align256(size_t b) { return (b + 255) & ~(size_t)255; }

// G, Buf, output are device pointers
extern "C" void solve(const float* G, const float* Buf, float* output, int M, int N,
                      float momentum) {
    const bool transposed = M > N;
    const int r = transposed ? N : M;      // Gram dimension (smaller)
    const int c = transposed ? M : N;

    size_t szX = align256((size_t)r * c * sizeof(float));
    size_t szA = align256((size_t)r * r * sizeof(float));
    size_t szR = align256((size_t)M * sizeof(double));
    char* base = (char*)scratch(2 * szX + 2 * szA + szR + 256);
    float*  X0     = (float*)base;
    float*  X1     = (float*)(base + szX);
    float*  Am     = (float*)(base + 2 * szX);
    float*  Bm     = (float*)(base + 2 * szX + szA);
    double* rowsum = (double*)(base + 2 * szX + 2 * szA);
    float*  denom  = (float*)(base + 2 * szX + 2 * szA + szR);

    // 1. ||V||_F and X0 = V / (||V|| + eps)
    row_sumsq<<<M, 256>>>(G, Buf, rowsum, N, momentum);
    finish_norm<<<1, 256>>>(rowsum, M, denom);
    {
        size_t total = (size_t)M * N;
        int blocks = (int)((total + 255) / 256);
        if (blocks > 4096) blocks = 4096;
        build_x<<<blocks, 256>>>(G, Buf, denom, X0, M, N, momentum, transposed);
    }

    const int Tr = (r + TILE - 1) / TILE, Tc = (c + TILE - 1) / TILE;
    const int symBlocks = Tr * (Tr + 1) / 2;
    const float scale = sqrtf((float)c);   // sqrt(max(M, N))

    float* Xc = X0;
    float* Xn = X1;
    for (int step = 0; step < 5; step++) {
        // A = X X^T                         (r x r, reduce over c)
        gemm_kernel<true, true, false><<<symBlocks, NT>>>(
            Xc, Xc, nullptr, Am, r, r, c, 1.0f, 0.0f, 1.0f, false, r);
        // B = b*A + c*(A A^T)               (r x r, reduce over r)
        gemm_kernel<true, true, true><<<symBlocks, NT>>>(
            Am, Am, Am, Bm, r, r, r, NS_C, NS_B, 1.0f, false, r);
        // X = a*X + B X                     (r x c, reduce over r)
        dim3 grid(Tc, Tr);
        if (step < 4) {
            gemm_kernel<false, false, true><<<grid, NT>>>(
                Bm, Xc, Xc, Xn, r, c, r, 1.0f, NS_A, 1.0f, false, c);
            float* t = Xc; Xc = Xn; Xn = t;
        } else {
            // last step: scale by sqrt(max(M,N)) and write (transposed back if needed) to output
            gemm_kernel<false, false, true><<<grid, NT>>>(
                Bm, Xc, Xc, output, r, c, r, 1.0f, NS_A, scale, transposed,
                transposed ? r : c);
        }
    }
}
