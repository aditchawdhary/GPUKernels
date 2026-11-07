#include <cuda_runtime.h>

#define TX 32          // columns per block (one per lane -> coalesced)
#define TY 8           // threads along rows
#define ROWS_PER_THREAD 4

// Pass 1: per-column shifted sums, split across row chunks
__global__ void bn_stats(const float* __restrict__ input,
                         double* __restrict__ colSum,
                         double* __restrict__ colSumSq,
                         int N, int C, int rowsPerBlock)
{
    __shared__ float shS[TY][TX + 1];
    __shared__ float shQ[TY][TX + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int c  = blockIdx.x * TX + tx;
    const int r0 = blockIdx.y * rowsPerBlock;
    const int r1 = min(N, r0 + rowsPerBlock);

    float s = 0.0f, q = 0.0f;
    if (c < C) {
        const float shift = __ldg(&input[c]);          // row 0 value
        for (int r = r0 + ty; r < r1; r += TY) {
            float d = __ldg(&input[(size_t)r * C + c]) - shift;
            s += d;
            q = fmaf(d, d, q);
        }
    }
    shS[ty][tx] = s;
    shQ[ty][tx] = q;
    __syncthreads();

    if (ty == 0 && c < C) {
        double ds = 0.0, dq = 0.0;
        #pragma unroll
        for (int k = 0; k < TY; ++k) { ds += shS[k][tx]; dq += shQ[k][tx]; }
        atomicAdd(&colSum[c],   ds);
        atomicAdd(&colSumSq[c], dq);
    }
}

// Pass 2: normalize, scale, shift
__global__ void bn_apply(const float* __restrict__ input,
                         const float* __restrict__ gamma,
                         const float* __restrict__ beta,
                         const double* __restrict__ colSum,
                         const double* __restrict__ colSumSq,
                         float* __restrict__ output,
                         int N, int C, float eps)
{
    __shared__ float sMean[TX], sScale[TX], sBeta[TX];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int c  = blockIdx.x * TX + tx;

    if (ty == 0 && c < C) {
        double m   = colSum[c] / N;                    // mean of shifted values
        double var = colSumSq[c] / N - m * m;          // biased variance
        if (var < 0.0) var = 0.0;
        double inv = 1.0 / sqrt(var + (double)eps);
        sMean[tx]  = (float)((double)input[c] + m);
        sScale[tx] = (float)(inv * gamma[c]);
        sBeta[tx]  = beta[c];
    }
    __syncthreads();
    if (c >= C) return;

    const float mean = sMean[tx], scale = sScale[tx], b = sBeta[tx];
    const int rBase = blockIdx.y * (TY * ROWS_PER_THREAD) + ty;

    #pragma unroll
    for (int k = 0; k < ROWS_PER_THREAD; ++k) {
        int r = rBase + k * TY;
        if (r < N) {
            size_t idx = (size_t)r * C + c;
            output[idx] = fmaf(__ldg(&input[idx]) - mean, scale, b);
        }
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta,
                      float* output, int N, int C, float eps)
{
    static double* stats = nullptr;      // [colSum | colSumSq], C <= 1024
    static int numSMs = 0;
    if (!stats) {
        cudaMalloc(&stats, 2 * 1024 * sizeof(double));
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
    }
    double* colSum   = stats;
    double* colSumSq = stats + 1024;
    cudaMemset(stats, 0, 2 * 1024 * sizeof(double));

    dim3 block(TX, TY);
    const int colBlocks = (C + TX - 1) / TX;

    // Split rows so the stats pass fills the GPU (~8 blocks per SM)
    int rowSplits = (numSMs * 8) / colBlocks;
    if (rowSplits < 1) rowSplits = 1;
    int maxSplits = (N + TY - 1) / TY;
    if (rowSplits > maxSplits) rowSplits = maxSplits;
    int rowsPerBlock = (N + rowSplits - 1) / rowSplits;
    rowSplits = (N + rowsPerBlock - 1) / rowsPerBlock;

    bn_stats<<<dim3(colBlocks, rowSplits), block>>>(input, colSum, colSumSq,
                                                    N, C, rowsPerBlock);

    const int rowsPerApplyBlock = TY * ROWS_PER_THREAD;
    dim3 gridApply(colBlocks, (N + rowsPerApplyBlock - 1) / rowsPerApplyBlock);
    bn_apply<<<gridApply, block>>>(input, gamma, beta, colSum, colSumSq,
                                   output, N, C, eps);
    cudaDeviceSynchronize();
}
