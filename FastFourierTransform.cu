#include <cuda_runtime.h>
#include <math.h>

// 1-D complex FFT, X_k = sum_n x_n e^{-2 pi i k n / N}, any 1 <= N <= 262144.
//
// Power-of-two length L (iterative radix-2 decimation-in-time):
//   1. bit-reversal permutation                       (in -> out, out of place)
//   2. stages 1..log2(B) inside shared memory, B = min(L, 4096): one block per B-chunk
//   3. remaining stages as global butterfly passes   (6 passes for L = 262144)
//   Twiddles e^{-i pi pos/half} come from sincospif, whose argument pos/half is exact.
//
// Other N (Bluestein / chirp-z): with w_n = e^{-i pi n^2 / N},
//   X_k = w_k * sum_n (x_n w_n) conj(w_{k-n})  = a circular convolution of length M >= 2N-1
//   (M a power of two) -> FFT(a), FFT(b), multiply, inverse FFT (via conjugation), scale.
//   n^2 is reduced mod 2N in 64-bit integers before forming the angle, so chirps stay exact.

#define LOCAL_N 4096
#define LOCAL_T 512

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__global__ void bitrev_permute(const float2* __restrict__ in, float2* __restrict__ out, int L, int logL) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < L; n += gridDim.x * blockDim.x)
        out[__brev((unsigned)n) >> (32 - logL)] = in[n];
}

// stages 1..logB on each contiguous chunk of B elements (data already bit-reversed)
__global__ void __launch_bounds__(LOCAL_T)
fft_local(float2* __restrict__ data, int B, int logB) {
    __shared__ float2 buf[LOCAL_N];
    float2* chunk = data + (size_t)blockIdx.x * B;
    for (int i = threadIdx.x; i < B; i += blockDim.x) buf[i] = chunk[i];
    __syncthreads();
    for (int s = 1; s <= logB; s++) {
        const int half = 1 << (s - 1);
        for (int j = threadIdx.x; j < B / 2; j += blockDim.x) {
            int pos = j & (half - 1);
            int i0 = ((j >> (s - 1)) << s) + pos, i1 = i0 + half;
            float sn, cs;
            sincospif(-(float)pos / (float)half, &sn, &cs);
            float2 a = buf[i0], b = cmul(buf[i1], make_float2(cs, sn));
            buf[i0] = make_float2(a.x + b.x, a.y + b.y);
            buf[i1] = make_float2(a.x - b.x, a.y - b.y);
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < B; i += blockDim.x) chunk[i] = buf[i];
}

// one radix-2 stage s over the whole array
__global__ void fft_global_stage(float2* __restrict__ data, int L, int s) {
    const int half = 1 << (s - 1);
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < L / 2; j += gridDim.x * blockDim.x) {
        int pos = j & (half - 1);
        int i0 = ((j >> (s - 1)) << s) + pos, i1 = i0 + half;
        float sn, cs;
        sincospif(-(float)pos / (float)half, &sn, &cs);
        float2 a = data[i0], b = cmul(data[i1], make_float2(cs, sn));
        data[i0] = make_float2(a.x + b.x, a.y + b.y);
        data[i1] = make_float2(a.x - b.x, a.y - b.y);
    }
}

static int grid_for(int n) { int g = (n + 255) / 256; return g > 4096 ? 4096 : (g < 1 ? 1 : g); }

// out = FFT(in), L = 2^logL >= 2, in != out
static void fft_pow2(const float2* in, float2* out, int L, int logL) {
    bitrev_permute<<<grid_for(L), 256>>>(in, out, L, logL);
    int logB = logL < 12 ? logL : 12;
    int B = 1 << logB;
    fft_local<<<L / B, LOCAL_T>>>(out, B, logB);
    for (int s = logB + 1; s <= logL; s++) fft_global_stage<<<grid_for(L / 2), 256>>>(out, L, s);
}

// ---------------- Bluestein helpers ----------------
__device__ __forceinline__ float2 chirp(long long n, int N) {      // e^{-i pi n^2 / N}
    long long r = (n * n) % (2LL * N);
    double s, c;
    sincospi(-(double)r / (double)N, &s, &c);
    return make_float2((float)c, (float)s);
}

__global__ void blu_make_a(const float2* __restrict__ x, float2* __restrict__ a, int N, int M) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < M; n += gridDim.x * blockDim.x)
        a[n] = n < N ? cmul(x[n], chirp(n, N)) : make_float2(0.f, 0.f);
}

__global__ void blu_make_b(float2* __restrict__ b, int N, int M) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < M; n += gridDim.x * blockDim.x) {
        float2 v = make_float2(0.f, 0.f);
        int m = -1;
        if (n < N) m = n;
        else if (n > M - N) m = M - n;
        if (m >= 0) { float2 w = chirp(m, N); v = make_float2(w.x, -w.y); }   // conj(w_m)
        b[n] = v;
    }
}

// c = conj(A * B)  (so FFT(c) = conj(M * IFFT(A*B)))
__global__ void blu_mul_conj(const float2* __restrict__ A, const float2* __restrict__ B,
                             float2* __restrict__ c, int M) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < M; n += gridDim.x * blockDim.x) {
        float2 p = cmul(A[n], B[n]);
        c[n] = make_float2(p.x, -p.y);
    }
}

// X_k = w_k * conj(F_k) / M
__global__ void blu_finish(const float2* __restrict__ F, float2* __restrict__ X, int N, int M) {
    const float invM = 1.0f / (float)M;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x) {
        float2 f = F[k];
        float2 y = make_float2(f.x * invM, -f.y * invM);
        X[k] = cmul(y, chirp(k, N));
    }
}

static float2* g_buf = nullptr;
static size_t g_bytes = 0;

// signal and spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int N) {
    const float2* x = reinterpret_cast<const float2*>(signal);
    float2* X = reinterpret_cast<float2*>(spectrum);

    if (N == 1) {
        cudaMemcpy(X, x, sizeof(float2), cudaMemcpyDeviceToDevice);
        return;
    }
    int logN = 0;
    while ((1 << logN) < N) logN++;
    if ((1 << logN) == N) {                       // power of two
        fft_pow2(x, X, N, logN);
        return;
    }

    // Bluestein
    int logM = 0;
    while ((1 << logM) < 2 * N - 1) logM++;
    const int M = 1 << logM;
    size_t bytes = (size_t)3 * M * sizeof(float2);
    if (bytes > g_bytes) {
        if (g_buf) cudaFree(g_buf);
        cudaMalloc(&g_buf, bytes);
        g_bytes = bytes;
    }
    float2* T = g_buf;          // staging
    float2* A = g_buf + M;
    float2* B = g_buf + 2 * (size_t)M;
    const int g = grid_for(M);

    blu_make_a<<<g, 256>>>(x, T, N, M);
    fft_pow2(T, A, M, logM);
    blu_make_b<<<g, 256>>>(T, N, M);
    fft_pow2(T, B, M, logM);
    blu_mul_conj<<<g, 256>>>(A, B, T, M);
    fft_pow2(T, A, M, logM);
    blu_finish<<<grid_for(N), 256>>>(A, X, N, M);
}
