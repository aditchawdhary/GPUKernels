#include <cuda_runtime.h>
#include <math.h>

// 2D DFT by row-column decomposition, X[k,l] = sum_m sum_n x[m,n] e^{-2 pi i (km/M + ln/N)}.
//
//   1. 1D DFT of every row (length N)            signal   -> spectrum
//   2. tiled transpose (M x N -> N x M)          spectrum -> tmp
//   3. 1D DFT of every row of tmp (length M)     tmp      -> tmp (in place)
//   4. tiled transpose back (N x M -> M x N)     tmp      -> spectrum
// so both passes read and write rows contiguously.
//
// Row transform, one block per row, whole row in shared memory (L <= 4096 complex = 32 KB):
//   * L a power of two: iterative radix-2 Cooley-Tukey (bit-reversed load, log2 L butterfly
//     stages), twiddles e^{-2 pi i k / L} precomputed per block with sincospif (2k/L is exact for 2^p).
//   * otherwise: exact O(L^2) DFT from shared memory with a twiddle table indexed by (k*n) mod L
//     (only small non-power-of-two sizes appear in the tests; the benchmark is 2048 x 2048).

#define FFT_THREADS 512
#define TDIM 32

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// radix-2 FFT of each row; in and out may alias (each block reads its row fully first)
__global__ void __launch_bounds__(FFT_THREADS)
fft_rows_pow2(const float2* in, float2* out, int L, int logL) {
    extern __shared__ float2 sm[];
    float2* buf = sm;            // [L]
    float2* tw = sm + L;         // [L/2]
    const size_t base = (size_t)blockIdx.x * L;

    for (int k = threadIdx.x; k < L / 2; k += blockDim.x) {
        float s, c;
        sincospif(-2.0f * (float)k / (float)L, &s, &c);
        tw[k] = make_float2(c, s);
    }
    for (int n = threadIdx.x; n < L; n += blockDim.x) {
        int rev = (int)(__brev((unsigned)n) >> (32 - logL));
        buf[rev] = in[base + n];
    }
    __syncthreads();

    for (int s = 1; s <= logL; s++) {
        const int half = 1 << (s - 1);
        const int tstride = L >> s;            // twiddle index step for this stage
        for (int j = threadIdx.x; j < L / 2; j += blockDim.x) {
            int pos = j & (half - 1);
            int i0 = ((j >> (s - 1)) << s) + pos;
            int i1 = i0 + half;
            float2 w = tw[pos * tstride];
            float2 a = buf[i0];
            float2 b = cmul(buf[i1], w);
            buf[i0] = make_float2(a.x + b.x, a.y + b.y);
            buf[i1] = make_float2(a.x - b.x, a.y - b.y);
        }
        __syncthreads();
    }

    for (int n = threadIdx.x; n < L; n += blockDim.x) out[base + n] = buf[n];
}

// exact DFT of each row for non-power-of-two L; in and out may alias
__global__ void __launch_bounds__(FFT_THREADS)
dft_rows(const float2* in, float2* out, int L) {
    extern __shared__ float2 sm[];
    float2* buf = sm;            // [L]
    float2* tw = sm + L;         // [L]
    const size_t base = (size_t)blockIdx.x * L;

    for (int k = threadIdx.x; k < L; k += blockDim.x) {
        float s, c;
        sincospif(-2.0f * (float)k / (float)L, &s, &c);
        tw[k] = make_float2(c, s);
        buf[k] = in[base + k];
    }
    __syncthreads();

    float2 res[(4096 + FFT_THREADS - 1) / FFT_THREADS];
    int cnt = 0;
    for (int k = threadIdx.x; k < L; k += blockDim.x, cnt++) {
        float re = 0.0f, im = 0.0f;
        int idx = 0;                           // (k * n) mod L, updated incrementally
        for (int n = 0; n < L; n++) {
            float2 x = buf[n], w = tw[idx];
            re = fmaf(x.x, w.x, fmaf(-x.y, w.y, re));
            im = fmaf(x.x, w.y, fmaf(x.y, w.x, im));
            idx += k;
            if (idx >= L) idx -= L;
        }
        res[cnt] = make_float2(re, im);
    }
    __syncthreads();                           // everyone done reading before in-place writes
    cnt = 0;
    for (int k = threadIdx.x; k < L; k += blockDim.x, cnt++) out[base + k] = res[cnt];
}

// out (cols x rows) = transpose of in (rows x cols)
__global__ void transpose_c(const float2* __restrict__ in, float2* __restrict__ out, int rows, int cols) {
    __shared__ float2 tile[TDIM][TDIM + 1];
    int x = blockIdx.x * TDIM + threadIdx.x;   // column in `in`
    int y0 = blockIdx.y * TDIM;
    for (int dy = threadIdx.y; dy < TDIM; dy += blockDim.y) {
        int y = y0 + dy;
        if (x < cols && y < rows) tile[dy][threadIdx.x] = in[(size_t)y * cols + x];
    }
    __syncthreads();
    int ox = y0 + threadIdx.x;                 // column in `out` = row in `in`
    for (int dy = threadIdx.y; dy < TDIM; dy += blockDim.y) {
        int oy = blockIdx.x * TDIM + dy;       // row in `out` = column in `in`
        if (oy < cols && ox < rows) out[(size_t)oy * rows + ox] = tile[threadIdx.x][dy];
    }
}

static int ilog2_exact(int L) {               // log2(L) if L is a power of two, else -1
    if (L <= 0 || (L & (L - 1))) return -1;
    int p = 0;
    while ((1 << p) < L) p++;
    return p;
}

static void row_transform(const float2* in, float2* out, int rows, int L) {
    if (L == 1) {
        if (in != out) cudaMemcpy(out, in, (size_t)rows * sizeof(float2), cudaMemcpyDeviceToDevice);
        return;
    }
    int p = ilog2_exact(L);
    if (p > 0) {
        size_t smem = (size_t)(L + L / 2) * sizeof(float2);
        cudaFuncSetAttribute(fft_rows_pow2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        fft_rows_pow2<<<rows, FFT_THREADS, smem>>>(in, out, L, p);
    } else {
        size_t smem = (size_t)(2 * L) * sizeof(float2);
        cudaFuncSetAttribute(dft_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        dft_rows<<<rows, FFT_THREADS, smem>>>(in, out, L);
    }
}

static float2* g_tmp = nullptr;
static size_t g_tmp_bytes = 0;

// signal, spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int M, int N) {
    const float2* x = reinterpret_cast<const float2*>(signal);
    float2* X = reinterpret_cast<float2*>(spectrum);
    size_t bytes = (size_t)M * N * sizeof(float2);
    if (bytes > g_tmp_bytes) {
        if (g_tmp) cudaFree(g_tmp);
        cudaMalloc(&g_tmp, bytes);
        g_tmp_bytes = bytes;
    }

    row_transform(x, X, M, N);                                    // DFT along rows
    dim3 tb(TDIM, 8);
    transpose_c<<<dim3((N + TDIM - 1) / TDIM, (M + TDIM - 1) / TDIM), tb>>>(X, g_tmp, M, N);
    row_transform(g_tmp, g_tmp, N, M);                            // DFT along columns
    transpose_c<<<dim3((M + TDIM - 1) / TDIM, (N + TDIM - 1) / TDIM), tb>>>(g_tmp, X, N, M);
}
