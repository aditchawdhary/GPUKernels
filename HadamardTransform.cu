/*
Hadamard Transform
Apply the orthonormal Walsh–Hadamard transform (natural / Sylvester ordering,
scaled by 1/sqrt(N)) to every row of a B x N float32 matrix, with N a power of two.
  e.g. [1, 2, 3, 4] -> [5, -1, -2, 0]

Constraints
 - 1 ≤ B ≤ 16,384;  1 ≤ N ≤ 4096, N a power of two;  tolerance 1e-4
 - Performance is measured with B = 8192, N = 4096
*/

/*
Approach: an in-register fast Walsh–Hadamard transform (FWHT), one block per row,
for N ≥ 512. A small shared-memory kernel handles smaller N.

The butterfly for bit b of the element index maps (a, b) -> (a + b, a - b). The
bits are processed in three tiers.

fwht_row<LOGN> (N = 2^LOGN ≥ 512, T = N/16 threads, 16 elements per thread):
 1. Load: thread t holds elements 16t .. 16t+15 (4 float4 loads).
 2. Register stages: element bits 0-3 are within a thread, giving 4 butterfly
    stages in registers.
 3. Shuffle stages: element bits 4-8 are the 5 lane bits. Each stage is one
    __shfl_xor; the upper lane takes partner - mine and the lower lane takes
    mine + partner.
 4. Warp stages (N > 512): element bits 9 and up are the warp bits. The row is
    written to padded shared memory and read back transposed, so each thread now
    holds the W = T/32 elements that differ only in those high bits. The
    remaining log2(W) stages then run in registers.
 5. Store: write back to shared memory, then do coalesced stores scaled by
    1/sqrt(N).
 The padding i + (i >> 5) makes both the blocked writes and the transposed reads
 free of bank conflicts. Shared memory is 16.5 KB at N = 4096.

Cost: one read and one write of the matrix (128 MB each at the benchmark) and
about N log N adds, so it's bandwidth-bound. N < 512 goes to fwht_small: 4096-float
tiles holding several rows, one shared-memory stage per bit.
*/

#include <cuda_runtime.h>
#include <math.h>

#define PAD(i) ((i) + ((i) >> 5))

template <int LOGN>
__global__ void __launch_bounds__((1 << LOGN) / 16)
fwht_row(const float* __restrict__ in, float* __restrict__ out, float scale)
{
    constexpr int N = 1 << LOGN;
    constexpr int T = N / 16;                 // threads
    constexpr int W = T / 32;                 // warps (1..8)
    constexpr int R = 16 / W;                 // low-space entries per thread in layout 2
    constexpr int LS = N / W;                 // low-space size

    __shared__ float sm[N + N / 32];

    const int t = threadIdx.x, lane = t & 31;
    const float* src = in  + (size_t)blockIdx.x * N;
    float*       dst = out + (size_t)blockIdx.x * N;

    // 1. load 16 consecutive elements
    float x[16];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(src + t * 16) + q);
        x[4 * q] = v.x; x[4 * q + 1] = v.y; x[4 * q + 2] = v.z; x[4 * q + 3] = v.w;
    }

    // 2. element bits 0..3: in registers
    #pragma unroll
    for (int h = 1; h < 16; h <<= 1)
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            if (!(j & h)) { const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }

    // 3. element bits 4..8: lane bits
    #pragma unroll
    for (int bit = 1; bit < 32; bit <<= 1) {
        const bool upper = lane & bit;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float p = __shfl_xor_sync(0xffffffff, x[j], bit);
            x[j] = upper ? (p - x[j]) : (x[j] + p);
        }
    }

    if (W == 1) {                              // N = 512: done
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            reinterpret_cast<float4*>(dst + t * 16)[q] =
                make_float4(x[4 * q] * scale, x[4 * q + 1] * scale, x[4 * q + 2] * scale, x[4 * q + 3] * scale);
        return;
    }

    // 4. element bits 9.. : warp bits, via a transpose through shared memory
    #pragma unroll
    for (int j = 0; j < 16; ++j) sm[PAD(t * 16 + j)] = x[j];
    __syncthreads();

    float z[16];                               // z[k*R + r] = element k*LS + t*R + r
    #pragma unroll
    for (int k = 0; k < W; ++k)
        #pragma unroll
        for (int r = 0; r < R; ++r) z[k * R + r] = sm[PAD(k * LS + t * R + r)];

    #pragma unroll
    for (int h = 1; h < W; h <<= 1)
        #pragma unroll
        for (int k = 0; k < W; ++k)
            if (!(k & h))
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float a = z[k * R + r], b = z[(k + h) * R + r];
                    z[k * R + r] = a + b; z[(k + h) * R + r] = a - b;
                }

    __syncthreads();                           // all transposed reads done
    #pragma unroll
    for (int k = 0; k < W; ++k)
        #pragma unroll
        for (int r = 0; r < R; ++r) sm[PAD(k * LS + t * R + r)] = z[k * R + r];
    __syncthreads();

    // 5. coalesced scaled store
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        const int i = n * T + t;
        dst[i] = sm[PAD(i)] * scale;
    }
}

// Small N (< 512): a block transforms a 4096-float tile holding 4096/N whole rows.
#define SMALL_TILE 4096
#define SMALL_THREADS 256

__global__ void __launch_bounds__(SMALL_THREADS)
fwht_small(const float* __restrict__ in, float* __restrict__ out, size_t total, int N, float scale)
{
    __shared__ float sm[SMALL_TILE];
    const size_t base = (size_t)blockIdx.x * SMALL_TILE;
    const int cnt = (int)min((size_t)SMALL_TILE, total - base);   // a multiple of N

    for (int i = threadIdx.x; i < cnt; i += SMALL_THREADS) sm[i] = __ldg(in + base + i);
    __syncthreads();

    for (int h = 1; h < N; h <<= 1) {
        for (int p = threadIdx.x; p < cnt / 2; p += SMALL_THREADS) {
            const int i = ((p & ~(h - 1)) << 1) | (p & (h - 1));   // bit h of i is clear
            const float a = sm[i], b = sm[i + h];
            sm[i] = a + b; sm[i + h] = a - b;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < cnt; i += SMALL_THREADS) out[base + i] = sm[i] * scale;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int B, int N)
{
    if (B <= 0 || N <= 0) return;
    const float scale = (float)(1.0 / sqrt((double)N));

    switch (N) {
        case 4096: fwht_row<12><<<B, 256>>>(input, output, scale); break;
        case 2048: fwht_row<11><<<B, 128>>>(input, output, scale); break;
        case 1024: fwht_row<10><<<B,  64>>>(input, output, scale); break;
        case  512: fwht_row< 9><<<B,  32>>>(input, output, scale); break;
        default: {
            const size_t total = (size_t)B * N;
            const unsigned blocks = (unsigned)((total + SMALL_TILE - 1) / SMALL_TILE);
            fwht_small<<<blocks, SMALL_THREADS>>>(input, output, total, N, scale);
        }
    }
    cudaDeviceSynchronize();
}
