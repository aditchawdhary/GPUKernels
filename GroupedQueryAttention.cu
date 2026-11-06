#include <cuda_runtime.h>
#include <math.h>

#define WARPS     8      // query rows per block
#define TILE      16     // keys loaded into shared memory at a time
#define MAX_DIM   256
#define PER_LANE  (MAX_DIM / 32)   // 8 elements per lane max

__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// grid = (ceil(seq_len / WARPS), num_q_heads); one warp = one query row.
__global__ void gqa_kernel(const float* Q, const float* K, const float* V, float* output,
                           int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    __shared__ float Ks[TILE * MAX_DIM];
    __shared__ float Vs[TILE * MAX_DIM];

    int q_head  = blockIdx.y;
    int group   = num_q_heads / num_kv_heads;
    int kv_head = q_head / group;              // the KV head this query head shares

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row  = blockIdx.x * WARPS + warp;      // query position
    bool active = row < seq_len;               // can't return early: we use __syncthreads

    const float* q_ptr  = Q + ((size_t)q_head  * seq_len + row) * head_dim;
    const float* K_head = K + (size_t)kv_head * seq_len * head_dim;
    const float* V_head = V + (size_t)kv_head * seq_len * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    // Each lane owns elements lane, lane+32, lane+64, ...
    float q[PER_LANE], acc[PER_LANE];
    for (int i = 0; i < PER_LANE; i++) {
        int k = lane + i * 32;
        q[i]   = (active && k < head_dim) ? q_ptr[k] : 0.0f;
        acc[i] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;

    for (int j0 = 0; j0 < seq_len; j0 += TILE) {
        int tile_len = min(TILE, seq_len - j0);

        // All 256 threads cooperatively load a tile of K and V.
        __syncthreads();
        for (int x = threadIdx.x; x < tile_len * head_dim; x += blockDim.x) {
            Ks[x] = K_head[(size_t)j0 * head_dim + x];
            Vs[x] = V_head[(size_t)j0 * head_dim + x];
        }
        __syncthreads();

        if (active) {
            for (int jj = 0; jj < tile_len; jj++) {
                const float* k_row = Ks + jj * head_dim;
                const float* v_row = Vs + jj * head_dim;

                // Score
                float partial = 0.0f;
                for (int i = 0; i < PER_LANE; i++) {
                    int k = lane + i * 32;
                    if (k < head_dim) partial += q[i] * k_row[k];
                }
                float s = warp_sum(partial) * scale;

                // Online softmax update
                float m_new = fmaxf(m, s);
                float alpha = expf(m - m_new);
                float p     = expf(s - m_new);
                l = l * alpha + p;
                for (int i = 0; i < PER_LANE; i++) {
                    int k = lane + i * 32;
                    if (k < head_dim) acc[i] = acc[i] * alpha + p * v_row[k];
                }
                m = m_new;
            }
        }
    }

    if (active) {
        float* o_ptr = output + ((size_t)q_head * seq_len + row) * head_dim;
        for (int i = 0; i < PER_LANE; i++) {
            int k = lane + i * 32;
            if (k < head_dim) o_ptr[k] = acc[i] / l;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    dim3 grid((seq_len + WARPS - 1) / WARPS, num_q_heads);
    dim3 block(WARPS * 32);
    gqa_kernel<<<grid, block>>>(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim);
}
