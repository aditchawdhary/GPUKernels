#include <cuda_runtime.h>
#include <math.h>

#define GT 16

// Double atomicAdd via CAS (works on every architecture).
__device__ __forceinline__ void atomicAddD(double* addr, double val) {
    unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
}

// In-place Cholesky (lower triangle of A, row-major p x p) + solve A x = b.
// Solution overwrites b. Must be called by all threads of one block.
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
    // Forward: L z = b
    for (int k = 0; k < p; ++k) {
        double zk = b[k] / A[k * p + k];
        __syncthreads();
        if (tid == 0) b[k] = zk;
        for (int i = k + 1 + tid; i < p; i += bd) b[i] -= A[i * p + k] * zk;
        __syncthreads();
    }
    // Backward: L^T x = z
    for (int k = p - 1; k >= 0; --k) {
        double xk = b[k] / A[k * p + k];
        __syncthreads();
        if (tid == 0) b[k] = xk;
        for (int i = tid; i < k; i += bd) b[i] -= A[k * p + i] * xk;
        __syncthreads();
    }
}

// ---------------- Small path: everything in one block, shared memory ----------------
#define SMALL_P 64
__global__ void __launch_bounds__(1024)
ols_small_kernel(const float* __restrict__ X, const float* __restrict__ y,
                 float* __restrict__ beta, int n, int p) {
    __shared__ double A[SMALL_P * SMALL_P];
    __shared__ double b[SMALL_P];
    const int tid = threadIdx.x, bd = blockDim.x;

    for (int idx = tid; idx < p * p; idx += bd) {
        int i = idx / p, j = idx % p;
        if (j > i) continue;                       // lower triangle only
        double s = 0.0;
        for (int r = 0; r < n; ++r)
            s += (double)X[(size_t)r * p + i] * (double)X[(size_t)r * p + j];
        A[i * p + j] = s;
    }
    for (int i = tid; i < p; i += bd) {
        double s = 0.0;
        for (int r = 0; r < n; ++r) s += (double)X[(size_t)r * p + i] * (double)y[r];
        b[i] = s;
    }
    __syncthreads();

    chol_solve(A, b, p);
    for (int i = tid; i < p; i += bd) beta[i] = (float)b[i];
}

// ---------------- General path ----------------
// G[i][j] += sum_s X[s][i] * X[s][j],  bvec[i] += sum_s X[s][i] * y[s]
__global__ void __launch_bounds__(GT * GT)
gram_kernel(const float* __restrict__ X, const float* __restrict__ y,
            double* G, double* bvec, int n, int p, int chunk) {
    __shared__ float Xi[GT][GT + 1], Xj[GT][GT + 1];
    __shared__ float ys[GT];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int ti = blockIdx.y * GT, tj = blockIdx.x * GT;
    if (tj > ti + GT - 1) return;                  // tile fully above diagonal: skip
    const int s0 = blockIdx.z * chunk;
    const int s1 = min(s0 + chunk, n);
    const bool doB = (blockIdx.x == 0);

    double acc = 0.0, bacc = 0.0;
    for (int s = s0; s < s1; s += GT) {
        int sr = s + ty;
        Xi[ty][tx] = (sr < s1 && ti + tx < p) ? X[(size_t)sr * p + ti + tx] : 0.0f;
        Xj[ty][tx] = (sr < s1 && tj + tx < p) ? X[(size_t)sr * p + tj + tx] : 0.0f;
        if (ty == 0) ys[tx] = (s + tx < s1) ? y[s + tx] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < GT; ++r)
            acc += (double)Xi[r][ty] * (double)Xj[r][tx];
        if (doB && ty == 0) {
            #pragma unroll
            for (int r = 0; r < GT; ++r) bacc += (double)Xi[r][tx] * (double)ys[r];
        }
        __syncthreads();
    }
    int gi = ti + ty, gj = tj + tx;
    if (gi < p && gj < p && gj <= gi) atomicAddD(&G[(size_t)gi * p + gj], acc);
    if (doB && ty == 0 && ti + tx < p) atomicAddD(&bvec[ti + tx], bacc);
}

__global__ void __launch_bounds__(1024)
chol_kernel(double* G, double* bvec, float* beta, int p) {
    chol_solve(G, bvec, p);
    for (int i = threadIdx.x; i < p; i += blockDim.x) beta[i] = (float)bvec[i];
}

extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    const int n = n_samples, p = n_features;
    if (n <= 0 || p <= 0) return;

    if (p <= SMALL_P && (long long)n * p * p <= (1LL << 22)) {
        ols_small_kernel<<<1, 1024>>>(X, y, beta, n, p);
        cudaDeviceSynchronize();
        return;
    }

    static double* buf = nullptr;
    static size_t cap = 0;
    size_t need = (size_t)p * p + p;
    if (cap < need) {
        if (buf) cudaFree(buf);
        cudaMalloc(&buf, need * sizeof(double));
        cap = need;
    }
    double* G = buf;
    double* bvec = buf + (size_t)p * p;
    cudaMemsetAsync(buf, 0, need * sizeof(double));

    int tiles1d = (p + GT - 1) / GT;
    int tiles = tiles1d * (tiles1d + 1) / 2;       // lower-triangle tiles actually doing work
    int splits = (512 + tiles - 1) / tiles;        // aim for enough blocks to fill the GPU
    int maxSplits = (n + 255) / 256;               // keep >= ~256 samples per split
    if (splits > maxSplits) splits = maxSplits;
    if (splits < 1) splits = 1;
    int chunk = (n + splits - 1) / splits;
    chunk = (chunk + GT - 1) / GT * GT;
    splits = (n + chunk - 1) / chunk;

    dim3 grid(tiles1d, tiles1d, splits);
    dim3 block(GT, GT);
    gram_kernel<<<grid, block>>>(X, y, G, bvec, n, p, chunk);
    chol_kernel<<<1, 1024>>>(G, bvec, beta, p);
    cudaDeviceSynchronize();
}
