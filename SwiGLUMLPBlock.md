#### SwiGLU MLP Block

Implement the SwiGLU MLP block — the feedforward network used in LLaMA, Mistral, Gemma, and most modern large language models. 

Given an input matrix `x` of shape `[M, d_model]` and three weight matrices `W_gate`, `W_up` (each `[d_model, d_ffn]`), and `W_down` (`[d_ffn, d_model]`), compute: 
`output = (SiLU(x × W_gate) ⊙ (x × W_up)) × W_down`, where `SiLU(z) = z × sigmoid(z)` and `⊙` denotes element-wise multiplication. All tensors are `float32`.

<img width="1405" height="446" alt="image" src="https://github.com/user-attachments/assets/d22001c4-3b27-4011-b4e7-cdb9de907cfa" />


#### Implementation Requirements
 - Implement the solve function with the signature unchanged.
 - Do not use external libraries beyond the framework provided.
 - Write the result into output in-place.

#### Example
Input: M = 2, d_model = 2, d_ffn = 4

<img width="1234" height="740" alt="image" src="https://github.com/user-attachments/assets/2cbdef0d-251a-4323-b6be-6c771e265bd1" />


#### Constraints
 - 1 ≤ M ≤ 65,536
 - 1 ≤ d_model ≤ 8,192
 - 1 ≤ d_ffn ≤ 32,768
 - All tensors are float32 on the GPU.
 - Input values are in the range [-10, 10].
 - Performance is measured with M = 512, d_model = 4,096, d_ffn = 14,336


### The plan

There are two kernel launches:

Fused gate/up kernel: one GEMM kernel computes `G` and `U` at the same time, reading each tile of x once for both. It applies SiLU and the multiply before writing `H`. Neither `G` nor `U` is ever stored.
Down kernel: the same GEMM kernel, without the activation, computes `H`,`W_down`.

`H` needs a temporary buffer: 512 × 14,336 floats ≈ 29 MB at the benchmark size. It’s allocated once and reused on later calls.

```
#include <cuda_runtime.h>
#include <math.h>

#define BM  64     // rows of C per block
#define BN  64     // cols of C per block
#define BK  16     // K-slice loaded per step
#define TPB 256    // threads per block; each computes a 4x4 patch

__device__ __forceinline__ float silu(float z) { return z / (1.0f + expf(-z)); }

// GLU = true :  C = silu(A @ B1) * (A @ B2)
// GLU = false:  C = A @ B1
// A is M x K, B1/B2 are K x N, C is M x N, all row-major.
template <bool GLU>
__global__ void __launch_bounds__(TPB)
gemm_kernel(const float* __restrict__ A, const float* __restrict__ B1,
            const float* __restrict__ B2, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BK][BM + 1];          // A tile, stored transposed (+1 avoids bank conflicts)
    __shared__ float B1s[BK][BN];
    __shared__ float B2s[GLU ? BK : 1][BN];

    int tid = threadIdx.x;
    int tx  = tid % 16, ty = tid / 16;        // this thread owns rows ty+16i, cols tx+16j
    int bm  = blockIdx.y * BM, bn = blockIdx.x * BN;

    float acc1[4][4] = {}, acc2[4][4] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load of the A tile (64 x 16): 4 elements per thread.
        for (int r = 0; r < 4; r++) {
            int e = tid + r * TPB, row = e / BK, col = e % BK;
            int gr = bm + row, gc = k0 + col;
            As[col][row] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
        }
        // Cooperative load of the B tile(s) (16 x 64): 4 elements per thread.
        for (int r = 0; r < 4; r++) {
            int e = tid + r * TPB, row = e / BN, col = e % BN;
            int gr = k0 + row, gc = bn + col;
            bool ok = gr < K && gc < N;
            B1s[row][col] = ok ? B1[(size_t)gr * N + gc] : 0.0f;
            if (GLU) B2s[row][col] = ok ? B2[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();

        // Multiply the tiles: each thread updates its 4x4 patch.
#pragma unroll
        for (int k = 0; k < BK; k++) {
            float a[4], b1[4], b2[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[k][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                b1[j] = B1s[k][tx + 16 * j];
                if (GLU) b2[j] = B2s[k][tx + 16 * j];
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc1[i][j] += a[i] * b1[j];
                    if (GLU) acc2[i][j] += a[i] * b2[j];
                }
        }
        __syncthreads();
    }

    // Write results (activation fused in for the GLU pass).
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = bm + ty + 16 * i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int c = bn + tx + 16 * j;
            if (c >= N) continue;
            C[(size_t)r * N + c] = GLU ? silu(acc1[i][j]) * acc2[i][j] : acc1[i][j];
        }
    }
}

// x, W_gate, W_up, W_down, output are device pointers
extern "C" void solve(const float* x, const float* W_gate, const float* W_up,
                      const float* W_down, float* output, int M, int d_model, int d_ffn) {
    // Scratch buffer for H (M x d_ffn), grown only when needed.
    static float* H = nullptr;
    static size_t capacity = 0;
    size_t need = (size_t)M * d_ffn;
    if (need > capacity) {
        if (H) cudaFree(H);
        cudaMalloc(&H, need * sizeof(float));
        capacity = need;
    }

    dim3 block(TPB);

    // 1. H = silu(x @ W_gate) * (x @ W_up)
    dim3 grid1((d_ffn + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<true><<<grid1, block>>>(x, W_gate, W_up, H, M, d_ffn, d_model);

    // 2. output = H @ W_down
    dim3 grid2((d_model + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<false><<<grid2, block>>>(H, W_down, nullptr, output, M, d_model, d_ffn);
}
```



### How the GEMM works
 - **Tiling:** each block computes a 64 × 64 patch of the output. It steps along the shared dimension in slices of 16. At each step, all 256 threads load a 64 × 16 tile of A and a 16 × 64 tile of B into shared memory. Every value loaded is then reused 64 times from fast memory instead of being re-read from DRAM.
 - **Register blocking:** each thread computes a 4 × 4 patch of outputs. For every 8 values read from shared memory it does 16 multiply-adds, so arithmetic rather than memory access is the bottleneck.
 - **No bank conflicts:** the A tile is stored transposed with one column of padding, so the loads into it don’t conflict. Reading it is a broadcast within each warp.
 - **Fusion:** the first kernel keeps two sets of accumulators against the same A tile, so x is read once for both projections. SiLU is applied in registers just before the single write of $H$, which avoids writing out and re-reading $G$ and $U$ (about 58 MB of extra traffic).
 - **Edge handling:** tiles that run past the matrix edge are padded with zeros, so any $M$, d_model and d_ffn work, not just multiples of 64.

### Performance

The benchmark is about 180 GFLOP of math, three GEMMs of 512 × 4,096 × 14,336, so it’s compute-bound. This kernel should reach a reasonable fraction of FP32 peak. To go further:
 - **Bigger tiles:** 128 × 128 per block with 8 × 8 per thread, plus float4 loads.
 - **Double buffering:** load the next slice of tiles while computing on the current one.
 - **TF32 tensor cores:** use wmma/mma instructions, often 5–10× faster. TF32 lowers precision slightly, so check it’s within the checker’s tolerance.

