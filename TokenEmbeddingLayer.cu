#include <cuda_runtime.h>

#define THREADS 256

// One block handles one (b, t) row of length D.
__global__ void embedding_layernorm(const int* token_ids, const int* position_ids,
                                    const float* token_embeddings, const float* position_embeddings,
                                    const float* gamma, const float* beta,
                                    float* output, int T, int D, float eps) {
    int row = blockIdx.x;          // row = b * T + t
    int t   = row % T;
    int tid = threadIdx.x;

    // Find the two embedding rows to add together.
    const float* tok_row = token_embeddings    + (size_t)token_ids[row]  * D;
    const float* pos_row = position_embeddings + (size_t)position_ids[t] * D;
    float*       out_row = output              + (size_t)row * D;

    __shared__ float partial[THREADS];

    // Step 1: s = token + position, written to output as scratch space; sum it.
    float sum = 0.0f;
    for (int d = tid; d < D; d += THREADS) {
        float s = tok_row[d] + pos_row[d];
        out_row[d] = s;
        sum += s;
    }

    // Reduce the per-thread sums to get the mean.
    partial[tid] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float mean = partial[0] / D;
    __syncthreads();   // everyone has read partial[0] before we reuse it

    // Step 2: variance = average of (s - mean)^2.
    float sq = 0.0f;
    for (int d = tid; d < D; d += THREADS) {
        float diff = out_row[d] - mean;
        sq += diff * diff;
    }

    partial[tid] = sq;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float variance = partial[0] / D;

    // Step 3: normalize, then scale and shift.
    float inv_std = 1.0f / sqrtf(variance + eps);
    for (int d = tid; d < D; d += THREADS) {
        out_row[d] = gamma[d] * (out_row[d] - mean) * inv_std + beta[d];
    }
}

// token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device
// pointers
extern "C" void solve(const int* token_ids, const int* position_ids, const float* token_embeddings,
                      const float* position_embeddings, const float* gamma, const float* beta,
                      float* output, int B, int T, int V, int P, int D, float eps) {
    int rows = B * T;
    embedding_layernorm<<<rows, THREADS>>>(token_ids, position_ids, token_embeddings,
                                           position_embeddings, gamma, beta, output, T, D, eps);
}
