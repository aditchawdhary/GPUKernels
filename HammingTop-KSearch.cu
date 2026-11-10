#include <cuda_runtime.h>
#include <stdint.h>

#define THREADS 256
#define IPT 16
#define CH (THREADS * IPT)      // codes per block (4096)
#define NB 257                  // distance bins 0..256

struct State { int T; int needEq; int cnt; };

// ---------------- Kernel 1: distances + histograms ----------------
__global__ void __launch_bounds__(THREADS)
dist_kernel(const int* __restrict__ codes, const int* __restrict__ query, int N,
            unsigned short* __restrict__ dist, int* __restrict__ blockHist,
            int* __restrict__ ghist, int aligned) {
    __shared__ int h[NB];
    for (int x = threadIdx.x; x < NB; x += THREADS) h[x] = 0;

    unsigned q[8];
    #pragma unroll
    for (int w = 0; w < 8; ++w) q[w] = (unsigned)__ldg(query + w);
    __syncthreads();

    const int base = blockIdx.x * CH;
    const int end = min(base + CH, N);
    for (int i = base + threadIdx.x; i < end; i += THREADS) {
        unsigned c[8];
        if (aligned) {
            const int4* p = reinterpret_cast<const int4*>(codes) + 2 * (size_t)i;
            int4 v0 = __ldg(p), v1 = __ldg(p + 1);
            c[0] = v0.x; c[1] = v0.y; c[2] = v0.z; c[3] = v0.w;
            c[4] = v1.x; c[5] = v1.y; c[6] = v1.z; c[7] = v1.w;
        } else {
            #pragma unroll
            for (int w = 0; w < 8; ++w) c[w] = (unsigned)__ldg(codes + 8 * (size_t)i + w);
        }
        int d = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) d += __popc(c[w] ^ q[w]);
        dist[i] = (unsigned short)d;
        atomicAdd(&h[d], 1);
    }
    __syncthreads();
    for (int x = threadIdx.x; x < NB; x += THREADS) {
        int v = h[x];
        blockHist[(size_t)blockIdx.x * NB + x] = v;
        if (v) atomicAdd(&ghist[x], v);
    }
}

// ---------------- Kernel 2: threshold + per-block tie offsets ----------------
__global__ void __launch_bounds__(1024)
select_kernel(const int* __restrict__ ghist, const int* __restrict__ blockHist, int nb, int k,
              int* __restrict__ eqOff, State* st) {
    __shared__ int sT;
    __shared__ int warpSums[32];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;

    if (tid == 0) {
        int cum = 0, T = NB - 1;
        for (int d = 0; d < NB; ++d) {
            int h = ghist[d];
            if (cum + h >= k) { T = d; break; }
            cum += h;
        }
        st->T = T;
        st->needEq = k - cum;
        st->cnt = 0;
        sT = T;
    }
    __syncthreads();
    const int T = sT;

    // Exclusive scan over blocks of blockHist[b][T].
    const int seg = (nb + 1023) / 1024;
    const int s0 = min(tid * seg, nb), s1 = min(s0 + seg, nb);
    int local = 0;
    for (int b = s0; b < s1; ++b) local += blockHist[(size_t)b * NB + T];

    int v = local;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) warpSums[wid] = v;
    __syncthreads();
    if (wid == 0) {
        int ws = warpSums[lane], wv = ws;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int n = __shfl_up_sync(0xffffffffu, wv, o);
            if (lane >= o) wv += n;
        }
        warpSums[lane] = wv - ws;
    }
    __syncthreads();
    int run = warpSums[wid] + v - local;
    for (int b = s0; b < s1; ++b) {
        eqOff[b] = run;
        run += blockHist[(size_t)b * NB + T];
    }
}

// ---------------- Kernel 3: collect exactly k candidates ----------------
__global__ void __launch_bounds__(THREADS)
collect_kernel(const unsigned short* __restrict__ dist, int N,
               const int* __restrict__ blockHist, const int* __restrict__ eqOff,
               State* st, unsigned long long* __restrict__ keys) {
    __shared__ int warpCnt[THREADS / 32];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int b = blockIdx.x;
    const int base = b * CH;
    const int end = min(base + CH, N);
    const unsigned T = (unsigned)st->T;
    const int needEq = st->needEq;
    const int myOff = eqOff[b];
    const bool doEq = blockHist[(size_t)b * NB + T] > 0 && myOff < needEq;

    int running = 0;
    for (int r = base; r < end; r += THREADS) {
        int i = r + tid;
        bool valid = i < end;
        unsigned d = valid ? dist[i] : 0xFFFFu;

        if (valid && d < T) {
            int pos = atomicAdd(&st->cnt, 1);
            keys[pos] = ((unsigned long long)d << 32) | (unsigned)i;
        }
        if (doEq) {
            bool e = valid && d == T;
            unsigned bal = __ballot_sync(0xffffffffu, e);
            if (lane == 0) warpCnt[wid] = __popc(bal);
            __syncthreads();
            int off = 0, total = 0;
            #pragma unroll
            for (int w = 0; w < THREADS / 32; ++w) {
                int c = warpCnt[w];
                off += (w < wid) ? c : 0;
                total += c;
            }
            int rank = running + off + __popc(bal & ((1u << lane) - 1u));
            if (e && myOff + rank < needEq) {
                int pos = atomicAdd(&st->cnt, 1);
                keys[pos] = ((unsigned long long)d << 32) | (unsigned)i;
            }
            running += total;
            __syncthreads();
        }
    }
}

// ---------------- Kernel 4: sort k candidates by (dist, index) ----------------
__global__ void __launch_bounds__(512)
sort_kernel(const unsigned long long* __restrict__ keys, int* __restrict__ out, int k, int P) {
    __shared__ unsigned long long s[1024];
    for (int i = threadIdx.x; i < P; i += blockDim.x) s[i] = (i < k) ? keys[i] : ~0ULL;
    __syncthreads();
    for (int sz = 2; sz <= P; sz <<= 1) {
        for (int j = sz >> 1; j > 0; j >>= 1) {
            for (int t = threadIdx.x; t < (P >> 1); t += blockDim.x) {
                int i = ((t & ~(j - 1)) << 1) | (t & (j - 1));
                int l = i + j;
                unsigned long long a = s[i], c = s[l];
                bool asc = (i & sz) == 0;
                if (asc ? (a > c) : (a < c)) { s[i] = c; s[l] = a; }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) out[i] = (int)(s[i] & 0xFFFFFFFFULL);
}

// codes, query, output are device pointers
extern "C" void solve(const int* codes, const int* query, int* output, int N, int k) {
    if (N <= 0 || k <= 0) return;
    if (k > N) k = N;
    const int nb = (N + CH - 1) / CH;

    // Scratch buffers, grown on demand and reused across calls.
    static char* mem = nullptr;
    static size_t memCap = 0;
    size_t distBytes = (((size_t)N * sizeof(unsigned short)) + 255) & ~(size_t)255;
    size_t bhBytes   = (((size_t)nb * NB * sizeof(int)) + 255) & ~(size_t)255;
    size_t eqBytes   = (((size_t)nb * sizeof(int)) + 255) & ~(size_t)255;
    size_t ghBytes   = 2048;
    size_t keyBytes  = 1024 * sizeof(unsigned long long);
    size_t need = distBytes + bhBytes + eqBytes + ghBytes + keyBytes + 256;
    if (memCap < need) {
        if (mem) cudaFree(mem);
        cudaMalloc(&mem, need);
        memCap = need;
    }
    char* p = mem;
    unsigned short* dist = reinterpret_cast<unsigned short*>(p); p += distBytes;
    int* blockHist = reinterpret_cast<int*>(p);                  p += bhBytes;
    int* eqOff = reinterpret_cast<int*>(p);                      p += eqBytes;
    int* ghist = reinterpret_cast<int*>(p);
    State* st = reinterpret_cast<State*>(p + NB * sizeof(int) + 64);
    p += ghBytes;
    unsigned long long* keys = reinterpret_cast<unsigned long long*>(p);

    int aligned = (((uintptr_t)codes & 15) == 0) ? 1 : 0;

    cudaMemsetAsync(ghist, 0, NB * sizeof(int));
    dist_kernel<<<nb, THREADS>>>(codes, query, N, dist, blockHist, ghist, aligned);
    select_kernel<<<1, 1024>>>(ghist, blockHist, nb, k, eqOff, st);
    collect_kernel<<<nb, THREADS>>>(dist, N, blockHist, eqOff, st, keys);

    int P = 1;
    while (P < k) P <<= 1;
    sort_kernel<<<1, 512>>>(keys, output, k, P);
    cudaDeviceSynchronize();
}
