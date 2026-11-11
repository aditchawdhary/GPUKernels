#include <cuda_runtime.h>
#include <math.h>

#define GT 16
#define MAXIT 100
#define LAMBDA 1e-6   // objective: -loglik + (LAMBDA/2) * ||beta||^2

__device__ __forceinline__ double sigmoid_d(double z) {
    if (z >= 0.0) return 1.0 / (1.0 + exp(-z));
    double e = exp(z);
    return e / (1.0 + e);
}

__device__ __forceinline__ void atomicAddD(double* addr, double val) {
    unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
}

// In-place Cholesky on lower triangle of A (p x p row-major), solves A x = b (x -> b).
// Called by all threads of one block.
__device__ void chol_solve(double* A, double* b, int p) {
    const int tid = threadIdx.x, bd = blockDim.x;
    for (int k = 0; k < p; ++k) {
        double d = sqrt(A[k * p + k]);
        __syncthreads();
        if (tid == 0) A[k * p + k] = d;
        for (int i = k + 1 + tid; i < p; i += bd) A[i * p + k] /= d;
        __syncthreads();
        int m = p - k - 1;
        for (int t = tid; t < m * m; t += bd) {
            int i = k + 1 + t / m, j = k + 1 + t % m;
            if (j <= i) A[i * p + j] -= A[i * p + k] * A[j * p + k];
        }
        __syncthreads();
    }
    for (int k = 0; k < p; ++k) {
        double zk = b[k] / A[k * p + k];
        __syncthreads();
        if (tid == 0) b[k] = zk;
        for (int i = k + 1 + tid; i < p; i += bd) b[i] -= A[i * p + k] * zk;
        __syncthreads();
    }
    for (int k = p - 1; k >= 0; --k) {
        double xk = b[k] / A[k * p + k];
        __syncthreads();
        if (tid == 0) b[k] = xk;
        for (int i = tid; i < k; i += bd) b[i] -= A[k * p + i] * xk;
        __syncthreads();
    }
}

// ============ Small path: full Newton loop in one block ============
#define SP 32
#define SN 2048
__global__ void __launch_bounds__(1024)
logreg_small_kernel(const float* __restrict__ X, const float* __restrict__ y,
                    float* __restrict__ beta, int n, int p) {
    __shared__ double A[SP * SP];
    __shared__ double g[SP];
    __shared__ double bet[SP];
    __shared__ double w[SN];
    __shared__ double r[SN];
    __shared__ int done;
    const int tid = threadIdx.x, bd = blockDim.x;

    for (int i = tid; i < p; i += bd) bet[i] = 0.0;
    if (tid == 0) done = 0;
    __syncthreads();

    for (int it = 0; it < MAXIT; ++it) {
        for (int s = tid; s < n; s += bd) {
            double eta = 0.0;
            for (int j = 0; j < p; ++j) eta += (double)X[(size_t)s * p + j] * bet[j];
            double pr = sigmoid_d(eta);
            w[s] = fmax(pr * (1.0 - pr), 1e-12);
            r[s] = (double)y[s] - pr;
        }
        __syncthreads();

        for (int idx = tid; idx < p * p; idx += bd) {
            int i = idx / p, j = idx % p;
            if (j > i) continue;
            double acc = 0.0;
            for (int s = 0; s < n; ++s)
                acc += w[s] * (double)X[(size_t)s * p + i] * (double)X[(size_t)s * p + j];
            if (i == j) acc += LAMBDA;
            A[i * p + j] = acc;
        }
        for (int i = tid; i < p; i += bd) {
            double acc = 0.0;
            for (int s = 0; s < n; ++s) acc += r[s] * (double)X[(size_t)s * p + i];
            g[i] = acc - LAMBDA * bet[i];
        }
        __syncthreads();

        chol_solve(A, g, p);                       // g := delta

        if (tid == 0) {
            double md = 0.0, mb = 0.0;
            for (int i = 0; i < p; ++i) {
                bet[i] += g[i];
                md = fmax(md, fabs(g[i]));
                mb = fmax(mb, fabs(bet[i]));
            }
            done = (md < 1e-7 * (1.0 + mb)) ? 1 : 0;
        }
        __syncthreads();
        if (done) break;
    }
    for (int i = tid; i < p; i += bd) beta[i] = (float)bet[i];
}

// ============ General path ============
struct Ctl { int done; };

// One warp per sample: w[s] = p(1-p), r[s] = y - p.
__global__ void prob_kernel(const float* __restrict__ X, const float* __restrict__ y,
                            const double* __restrict__ bet, double* __restrict__ w,
                            double* __restrict__ r, int n, int p, const Ctl* ctl) {
    if (ctl->done) return;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n) return;
    const float* row = X + (size_t)warp * p;
    double eta = 0.0;
    for (int j = lane; j < p; j += 32) eta += (double)row[j] * bet[j];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) eta += __shfl_xor_sync(0xffffffffu, eta, o);
    if (lane == 0) {
        double pr = sigmoid_d(eta);
        w[warp] = fmax(pr * (1.0 - pr), 1e-12);
        r[warp] = (double)y[warp] - pr;
    }
}

// H[i][j] += sum_s w_s X[s][i] X[s][j] (lower triangle), g[i] += sum_s r_s X[s][i]
__global__ void __launch_bounds__(GT * GT)
hess_kernel(const float* __restrict__ X, const double* __restrict__ w,
            const double* __restrict__ r, double* H, double* g,
            int n, int p, int chunk, const Ctl* ctl) {
    if (ctl->done) return;
    __shared__ float Xi[GT][GT + 1], Xj[GT][GT + 1];
    __shared__ double ws[GT], rs[GT];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int ti = blockIdx.y * GT, tj = blockIdx.x * GT;
    if (tj > ti + GT - 1) return;
    const int s0 = blockIdx.z * chunk;
    const int s1 = min(s0 + chunk, n);
    const bool doG = (blockIdx.x == 0);

    double acc = 0.0, gacc = 0.0;
    for (int s = s0; s < s1; s += GT) {
        int sr = s + ty;
        Xi[ty][tx] = (sr < s1 && ti + tx < p) ? X[(size_t)sr * p + ti + tx] : 0.0f;
        Xj[ty][tx] = (sr < s1 && tj + tx < p) ? X[(size_t)sr * p + tj + tx] : 0.0f;
        if (ty == 0) {
            bool ok = (s + tx) < s1;
            ws[tx] = ok ? w[s + tx] : 0.0;
            rs[tx] = ok ? r[s + tx] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < GT; ++k)
            acc += ws[k] * (double)Xi[k][ty] * (double)Xj[k][tx];
        if (doG && ty == 0) {
            #pragma unroll
            for (int k = 0; k < GT; ++k) gacc += rs[k] * (double)Xi[k][tx];
        }
        __syncthreads();
    }
    int gi = ti + ty, gj = tj + tx;
    if (gi < p && gj < p && gj <= gi) atomicAddD(&H[(size_t)gi * p + gj], acc);
    if (doG && ty == 0 && ti + tx < p) atomicAddD(&g[ti + tx], gacc);
}

// Add ridge term, solve H delta = g, update beta, test convergence, re-zero H/g.
__global__ void __launch_bounds__(1024)
newton_step_kernel(double* H, double* g, double* bet, float* beta, int p, Ctl* ctl) {
    if (ctl->done) return;
    for (int i = threadIdx.x; i < p; i += blockDim.x) {
        H[(size_t)i * p + i] += LAMBDA;
        g[i] -= LAMBDA * bet[i];
    }
    __syncthreads();
    chol_solve(H, g, p);
    if (threadIdx.x == 0) {
        double md = 0.0, mb = 0.0;
        for (int i = 0; i < p; ++i) {
            bet[i] += g[i];
            md = fmax(md, fabs(g[i]));
            mb = fmax(mb, fabs(bet[i]));
        }
        if (md < 1e-7 * (1.0 + mb)) ctl->done = 1;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < p; i += blockDim.x) { beta[i] = (float)bet[i]; g[i] = 0.0; }
    for (size_t i = threadIdx.x; i < (size_t)p * p; i += blockDim.x) H[i] = 0.0;
}

extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    const int n = n_samples, p = n_features;
    if (n <= 0 || p <= 0) return;

    if (p <= SP && n <= SN) {
        logreg_small_kernel<<<1, 1024>>>(X, y, beta, n, p);
        cudaDeviceSynchronize();
        return;
    }

    // Scratch: H (p*p), g (p), bet (p), w (n), r (n), ctl.
    static char* mem = nullptr;
    static size_t cap = 0;
    size_t nd = (size_t)p * p + 2 * (size_t)p + 2 * (size_t)n;
    size_t need = nd * sizeof(double) + 256;
    if (cap < need) {
        if (mem) cudaFree(mem);
        cudaMalloc(&mem, need);
        cap = need;
    }
    double* H   = reinterpret_cast<double*>(mem);
    double* g   = H + (size_t)p * p;
    double* bet = g + p;
    double* w   = bet + p;
    double* r   = w + n;
    Ctl* ctl    = reinterpret_cast<Ctl*>(mem + nd * sizeof(double));
    cudaMemsetAsync(mem, 0, need);                 // H = g = beta = 0, done = 0

    int tiles1d = (p + GT - 1) / GT;
    int tiles = tiles1d * (tiles1d + 1) / 2;
    int splits = (512 + tiles - 1) / tiles;
    int maxSplits = (n + 255) / 256;
    if (splits > maxSplits) splits = maxSplits;
    if (splits < 1) splits = 1;
    int chunk = (n + splits - 1) / splits;
    chunk = (chunk + GT - 1) / GT * GT;
    splits = (n + chunk - 1) / chunk;

    dim3 hgrid(tiles1d, tiles1d, splits), hblock(GT, GT);
    int probBlocks = (int)(((size_t)n * 32 + 255) / 256);

    for (int it = 0; it < MAXIT; ++it) {
        prob_kernel<<<probBlocks, 256>>>(X, y, bet, w, r, n, p, ctl);
        hess_kernel<<<hgrid, hblock>>>(X, w, r, H, g, n, p, chunk, ctl);
        newton_step_kernel<<<1, 1024>>>(H, g, bet, beta, p, ctl);
    }
    cudaDeviceSynchronize();
}
