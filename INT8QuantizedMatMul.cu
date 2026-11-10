#include <cuda_runtime.h>
#include <stdint.h>

// C = clamp( rint( (sA*sB/sC) * sum_k (A[m,k]-zA)(B[k,n]-zB) ) + zC, -128, 127 )
//
// Expand the zero points so the hot loop is a plain int8 x int8 GEMM:
//   sum (A-zA)(B-zB) = sum A*B - zB*rowA[m] - zA*colB[n] + K*zA*zB
//
// 1. Prep: transpose B (K x N) -> Bt (N x Kp), so both operands are
//    K-contiguous (what int8 tensor-core MMAs want); Kp = K rounded up to 16,
//    zero padded. If K % 16 != 0, A is also copied into a padded buffer.
//    rowA = row sums of A, colB = row sums of Bt.
// 2. GEMM: 128x128x64 block tile, 8 warps (64x32 each), 3-stage cp.async
//    pipeline, XOR-swizzled smem, ldmatrix + mma.sync.m16n8k32.s8 (sm_80+).
//    Fallback for older GPUs: smem-tiled __dp4a kernel.
// 3. Epilogue (fused): zero-point correction, float scale, rintf, +zC, clamp.

// ============================================================ prep kernels
// Bt[n][k] = B[k][n] for k < K, 0 for K <= k < Kp
__global__ void transpose_pad_B(const int8_t* __restrict__ B, int8_t* __restrict__ Bt,
                                int K, int N, int Kp)
{
    __shared__ int8_t t[32][33];
    const int n0 = blockIdx.x * 32, k0 = blockIdx.y * 32;
    for (int y = threadIdx.y; y < 32; y += 8) {
        const int k = k0 + y, n = n0 + threadIdx.x;
        t[y][threadIdx.x] = (k < K && n < N) ? B[(size_t)k * N + n] : (int8_t)0;
    }
    __syncthreads();
    for (int y = threadIdx.y; y < 32; y += 8) {
        const int n = n0 + y, k = k0 + threadIdx.x;
        if (n < N && k < Kp) Bt[(size_t)n * Kp + k] = t[threadIdx.x][y];
    }
}

// Ap[m][k] = A[m][k] for k < K, 0 for K <= k < Kp  (only used when K % 16 != 0)
__global__ void pad_A(const int8_t* __restrict__ A, int8_t* __restrict__ Ap,
                      int M, int K, int Kp)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * Kp) return;
    const int m = (int)(i / Kp), k = (int)(i % Kp);
    Ap[i] = (k < K) ? A[(size_t)m * K + k] : (int8_t)0;
}

// out[r] = sum of row r of X (R x Kp, Kp % 16 == 0). One warp per row.
__global__ void row_sums(const int8_t* __restrict__ X, int* __restrict__ out, int R, int Kp)
{
    const int row  = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= R) return;
    const int4* p = reinterpret_cast<const int4*>(X + (size_t)row * Kp);
    int s = 0;
    for (int i = lane; i < Kp / 16; i += 32) {
        const int4 v = __ldg(p + i);
        s = __dp4a(v.x, 0x01010101, s);
        s = __dp4a(v.y, 0x01010101, s);
        s = __dp4a(v.z, 0x01010101, s);
        s = __dp4a(v.w, 0x01010101, s);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if (lane == 0) out[row] = s;
}

// ============================================================ epilogue
struct Quant {
    float scale;       // sA * sB / sC
    int zA, zB, zC, Kzz;
};

__device__ __forceinline__ int8_t requant(int acc, int ra, int cb, const Quant& q)
{
    const int v = acc - q.zB * ra - q.zA * cb + q.Kzz;
    int r = (int)rintf((float)v * q.scale) + q.zC;      // round half to even
    r = max(-128, min(127, r));
    return (int8_t)r;
}

// ============================================================ tensor-core GEMM
#define BM 128
#define BN 128
#define BK 64              // bytes of K per stage (two k32 MMA steps)
#define STAGES 3
#define GEMM_THREADS 256

// Tile rows are 64 B = four 16 B chunks. XOR the chunk with (row>>1)&3 so the
// 8 rows read by one ldmatrix 8x16B matrix land in 8 distinct 16 B bank
// groups (no conflicts), and cp.async writes stay conflict-free too.
__device__ __forceinline__ int swz(int r, int c) { return r * BK + ((c ^ ((r >> 1) & 3)) << 4); }

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid)
{
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    const int n = valid ? 16 : 0;                          // 0 -> zero fill
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N_)); }

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem)
{
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void mma_s8(int (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A: M x Kp (row-major), Bt: N x Kp (row-major), both 16 B aligned rows.
__global__ void __launch_bounds__(GEMM_THREADS, 2)
gemm_s8_mma(const int8_t* __restrict__ A, const int8_t* __restrict__ Bt,
            const int* __restrict__ rowA, const int* __restrict__ colB,
            int8_t* __restrict__ C, int M, int N, int Kp, Quant q)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) int8_t sA[STAGES][BM * BK];
    __shared__ __align__(128) int8_t sB[STAGES][BN * BK];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2;          // 0..1 -> 64-row slab
    const int wn = warp & 3;           // 0..3 -> 32-col slab
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    const int KT = (Kp + BK - 1) / BK;

    auto load_tile = [&](int stage, int kt) {
        const int kb = kt * BK;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {                  // 512 chunks / 256 threads
            const int id = tid + i * GEMM_THREADS;
            const int r = id >> 2, c = id & 3;
            const int k = kb + c * 16;
            const int gm = bm + r, gn = bn + r;
            const bool va = (gm < M) && (k < Kp);
            const bool vb = (gn < N) && (k < Kp);
            cp_async16(&sA[stage][swz(r, c)], va ? (const void*)(A  + (size_t)gm * Kp + k) : (const void*)A,  va);
            cp_async16(&sB[stage][swz(r, c)], vb ? (const void*)(Bt + (size_t)gn * Kp + k) : (const void*)Bt, vb);
        }
    };

    int acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0;

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_tile(s, s);
        cp_async_commit();
    }

    const int lr  = lane & 7;          // row within an 8x16B matrix
    const int mat = lane >> 3;         // which of the 4 matrices this lane addresses

    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();               // tile kt visible; stage being refilled is free

        const int nk = kt + STAGES - 1;
        if (nk < KT) load_tile(nk % STAGES, nk);
        cp_async_commit();

        const int8_t* a_s = sA[kt % STAGES];
        const int8_t* b_s = sB[kt % STAGES];

        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {               // two k32 steps per 64 B
            unsigned af[4][4], bf[2][4];
            // A 16x32 fragment: m0 rows 0-7/k 0-15, m1 rows 8-15/k 0-15,
            //                   m2 rows 0-7/k16-31, m3 rows 8-15/k16-31
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int r = wm * 64 + mi * 16 + lr + (mat & 1) * 8;
                const int c = ks * 2 + (mat >> 1);
                ldmatrix_x4(af[mi], a_s + swz(r, c));
            }
            // B (two n8 tiles per x4): m0 n 0-7/k 0-15, m1 n 0-7/k16-31,
            //                          m2 n 8-15/k 0-15, m3 n 8-15/k16-31
            #pragma unroll
            for (int nj = 0; nj < 2; ++nj) {
                const int r = wn * 32 + nj * 16 + lr + (mat >> 1) * 8;
                const int c = ks * 2 + (mat & 1);
                ldmatrix_x4(bf[nj], b_s + swz(r, c));
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma_s8(acc[mi][ni], af[mi], bf[ni >> 1][(ni & 1) * 2], bf[ni >> 1][(ni & 1) * 2 + 1]);
        }
    }
    cp_async_wait<0>();

    // Epilogue. m16n8 accumulator: c0,c1 -> (g, 2t..2t+1), c2,c3 -> (g+8, 2t..2t+1)
    const int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int m = bm + wm * 64 + mi * 16 + g + h * 8;
            if (m >= M) continue;
            const int ra = __ldg(rowA + m);
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int n = bn + wn * 32 + ni * 8 + t * 2;
                #pragma unroll
                for (int e = 0; e < 2; ++e)
                    if (n + e < N)
                        C[(size_t)m * N + n + e] = requant(acc[mi][ni][h * 2 + e], ra, __ldg(colB + n + e), q);
            }
        }
    }
#endif
}

// ============================================================ dp4a fallback (< sm_80)
#define FT 64
__global__ void __launch_bounds__(256)
gemm_s8_dp4a(const int8_t* __restrict__ A, const int8_t* __restrict__ Bt,
             const int* __restrict__ rowA, const int* __restrict__ colB,
             int8_t* __restrict__ C, int M, int N, int Kp, Quant q)
{
    __shared__ int As[FT][17], Bs[FT][17];      // 64 rows x 16 ints (64 B of K), padded
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int bm = blockIdx.y * FT, bn = blockIdx.x * FT;
    int acc[4][4] = {};

    for (int k0 = 0; k0 < Kp; k0 += 64) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int id = threadIdx.x + i * 256;     // 0..1023
            const int r = id >> 4, w = id & 15;
            const int k = k0 + w * 4;
            const int gm = bm + r, gn = bn + r;
            As[r][w] = (gm < M && k < Kp) ? *reinterpret_cast<const int*>(A  + (size_t)gm * Kp + k) : 0;
            Bs[r][w] = (gn < N && k < Kp) ? *reinterpret_cast<const int*>(Bt + (size_t)gn * Kp + k) : 0;
        }
        __syncthreads();
        #pragma unroll
        for (int w = 0; w < 16; ++w) {
            int a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { a[i] = As[ty + 16 * i][w]; b[i] = Bs[tx + 16 * i][w]; }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = bm + ty + 16 * i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = bn + tx + 16 * j;
            if (n < N) C[(size_t)m * N + n] = requant(acc[i][j], rowA[m], colB[n], q);
        }
    }
}

// ============================================================ host
// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C,
                      int zero_point_A, int zero_point_B, int zero_point_C)
{
    const int Kp = (K + 15) & ~15;
    const bool padA = (Kp != K);

    // Scratch: Bt | Ap (optional) | rowA | colB, grown on demand, reused across calls
    const size_t szBt = (size_t)N * Kp, szAp = padA ? (size_t)M * Kp : 0;
    const size_t offAp = (szBt + 255) & ~(size_t)255;
    const size_t offRA = (offAp + szAp + 255) & ~(size_t)255;
    const size_t offCB = offRA + (((size_t)M * 4 + 255) & ~(size_t)255);
    const size_t need  = offCB + (size_t)N * 4;
    static char*  scratch = nullptr;
    static size_t cap = 0;
    if (need > cap) {
        if (scratch) cudaFree(scratch);
        cudaMalloc(&scratch, need);
        cap = need;
    }
    int8_t* Bt   = reinterpret_cast<int8_t*>(scratch);
    int8_t* Ap   = reinterpret_cast<int8_t*>(scratch + offAp);
    int*    rowA = reinterpret_cast<int*>(scratch + offRA);
    int*    colB = reinterpret_cast<int*>(scratch + offCB);

    transpose_pad_B<<<dim3((N + 31) / 32, (Kp + 31) / 32), dim3(32, 8)>>>(B, Bt, K, N, Kp);
    const int8_t* Aeff = A;
    if (padA) {
        const size_t tot = (size_t)M * Kp;
        pad_A<<<(unsigned)((tot + 255) / 256), 256>>>(A, Ap, M, K, Kp);
        Aeff = Ap;
    }
    row_sums<<<(M + 7) / 8, 256>>>(Aeff, rowA, M, Kp);
    row_sums<<<(N + 7) / 8, 256>>>(Bt, colB, N, Kp);

    Quant q;
    q.scale = (scale_A * scale_B) / scale_C;
    q.zA = zero_point_A; q.zB = zero_point_B; q.zC = zero_point_C;
    q.Kzz = K * zero_point_A * zero_point_B;

    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);

    if (major >= 8) {
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_s8_mma<<<grid, GEMM_THREADS>>>(Aeff, Bt, rowA, colB, C, M, N, Kp, q);
    } else {
        dim3 grid((N + FT - 1) / FT, (M + FT - 1) / FT);
        gemm_s8_dp4a<<<grid, 256>>>(Aeff, Bt, rowA, colB, C, M, N, Kp, q);
    }
    cudaDeviceSynchronize();
}
