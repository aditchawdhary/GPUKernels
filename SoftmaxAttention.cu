#include <cuda_runtime.h>
#include <math.h>

#define WARPS_PER_BLOCK 8
#define MAX_PER_LANE 4          // supports d <= 128

// Sum a value across the warp; every lane gets the result.
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// One warp computes one output row using online softmax.
__global__ void attention_kernel(const float* Q, const float* K, const float* V,
                                 float* output, int M, int N, int d) {
    int row  = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= M) return;

    float scale = 1.0f / sqrtf((float)d);

    // Lane holds elements lane, lane+32, lane+64, lane+96.
    float q[MAX_PER_LANE], acc[MAX_PER_LANE];
    for (int i = 0; i < MAX_PER_LANE; i++) {
        int k = lane + i * 32;
        q[i]   = (k < d) ? Q[(size_t)row * d + k] : 0.0f;
        acc[i] = 0.0f;
    }

    float m = -INFINITY;   // running max score
    float l = 0.0f;        // running sum of exp(score - m)

    for (int j = 0; j < N; j++) {
        const float* k_row = K + (size_t)j * d;
        const float* v_row = V + (size_t)j * d;

        // Score s = (q . k_j) / sqrt(d)
        float partial = 0.0f;
        for (int i = 0; i < MAX_PER_LANE; i++) {
            int k = lane + i * 32;
            if (k < d) partial += q[i] * k_row[k];
        }
        float s = warp_sum(partial) * scale;

        // Online softmax update
        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);   // rescales old totals
        float p     = expf(s - m_new);   // weight of this key
        l = l * alpha + p;

        for (int i = 0; i < MAX_PER_LANE; i++) {
            int k = lane + i * 32;
            if (k < d) acc[i] = acc[i] * alpha + p * v_row[k];
        }
        m = m_new;
    }

    // Final normalization
    for (int i = 0; i < MAX_PER_LANE; i++) {
        int k = lane + i * 32;
        if (k < d) output[(size_t)row * d + k] = acc[i] / l;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int M, int N, int d) {
    int threads = WARPS_PER_BLOCK * 32;
    int blocks  = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    attention_kernel<<<blocks, threads>>>(Q, K, V, output, M, N, d);
}
