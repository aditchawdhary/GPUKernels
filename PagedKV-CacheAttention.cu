#include <cuda_runtime.h>
#include <math.h>

#define WARPS    16          // warps per (sequence, head)
#define MAX_DIM  256
#define PER_LANE 2           // float4s per lane: 256 / 4 / 32

__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// grid = (num_heads, batch_size): one block per (sequence s, head h).
__global__ void paged_attention_kernel(const float* Q, const float* K_cache, const float* V_cache,
                                       const int* block_table, const int* context_lens,
                                       float* output, int num_heads, int head_dim,
                                       int block_size, int max_blocks_per_seq) {
    __shared__ float m_s[WARPS], l_s[WARPS];
    __shared__ float acc_s[WARPS][MAX_DIM];

    int h    = blockIdx.x;
    int s    = blockIdx.y;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    int ctx  = context_lens[s];
    int D4   = head_dim / 4;                  // head_dim is a multiple of 4
    float scale = 1.0f / sqrtf((float)head_dim);
    const int* table = block_table + (size_t)s * max_blocks_per_seq;

    // Load this (s, h)'s query; lane owns float4s lane and lane+32.
    const float4* q4 = reinterpret_cast<const float4*>(Q + ((size_t)s * num_heads + h) * head_dim);
    float4 q[PER_LANE], acc[PER_LANE];
    for (int i = 0; i < PER_LANE; i++) {
        int k = lane + i * 32;
        q[i]   = (k < D4) ? q4[k] : make_float4(0, 0, 0, 0);
        acc[i] = make_float4(0, 0, 0, 0);
    }
    float m = -INFINITY, l = 0.0f;

    // Each warp handles tokens warp, warp+16, warp+32, ...
    for (int t = warp; t < ctx; t += WARPS) {
        // Paged lookup: logical token -> physical location
        int phys   = table[t / block_size];
        int offset = t % block_size;
        size_t base = (((size_t)phys * block_size + offset) * num_heads + h) * head_dim;
        const float4* k4 = reinterpret_cast<const float4*>(K_cache + base);
        const float4* v4 = reinterpret_cast<const float4*>(V_cache + base);

        // Score
        float partial = 0.0f;
        for (int i = 0; i < PER_LANE; i++) {
            int k = lane + i * 32;
            if (k < D4) {
                float4 kv = k4[k];
                partial += q[i].x * kv.x + q[i].y * kv.y + q[i].z * kv.z + q[i].w * kv.w;
            }
        }
        float score = warp_sum(partial) * scale;

        // Online softmax update
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p     = expf(score - m_new);
        l = l * alpha + p;
        for (int i = 0; i < PER_LANE; i++) {
            int k = lane + i * 32;
            if (k < D4) {
                float4 vv = v4[k];
                acc[i].x = acc[i].x * alpha + p * vv.x;
                acc[i].y = acc[i].y * alpha + p * vv.y;
                acc[i].z = acc[i].z * alpha + p * vv.z;
                acc[i].w = acc[i].w * alpha + p * vv.w;
            }
        }
        m = m_new;
    }

    // Write each warp's partial result to shared memory.
    if (lane == 0) { m_s[warp] = m; l_s[warp] = l; }
    for (int i = 0; i < PER_LANE; i++) {
        int k = lane + i * 32;
        if (k < D4) reinterpret_cast<float4*>(acc_s[warp])[k] = acc[i];
    }
    __syncthreads();

    // Merge the 16 partial softmaxes.
    float M = -INFINITY;
    for (int w = 0; w < WARPS; w++) M = fmaxf(M, m_s[w]);

    float L = 0.0f;
    for (int w = 0; w < WARPS; w++) L += l_s[w] * expf(m_s[w] - M);   // empty warps: exp(-inf) = 0

    float* out = output + ((size_t)s * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int w = 0; w < WARPS; w++) sum += acc_s[w][d] * expf(m_s[w] - M);
        out[d] = sum / L;
    }
}

// Q, K_cache, V_cache, block_table, context_lens, output are device pointers
extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, float* output,
                      int batch_size, int num_heads, int head_dim, int block_size,
                      int max_blocks_per_seq) {
    dim3 grid(num_heads, batch_size);
    dim3 block(WARPS * 32);
    paged_attention_kernel<<<grid, block>>>(Q, K_cache, V_cache, block_table, context_lens,
                                            output, num_heads, head_dim, block_size,
                                            max_blocks_per_seq);
}
