/*
Multi-Agent Simulation (one flocking step)
agents, agents_next: (4N) float32, agent i = [x, y, vx, vy].
Spec from the reference implementation (leetgpu-challenges #14), tolerance 1e-5:
  neighbors(i) = { j != i : (x_i - x_j)^2 + (y_i - y_j)^2 < 25 }     (strict)
  avg_v  = mean of neighbor velocities, or v_i when there are no neighbors
  new_v  = v_i + 0.05 * (avg_v - v_i)
  new_p  = p_i + new_v
Constraints: N ≤ 100,000; performance at N = 10,000 (values in [-1000, 1000]).
*/

/*
Approach: an exact all-pairs neighbor search (shared-memory tiles), split over the
source agents for parallelism, then a deterministic finalize.

 - Exactness at the r = 5 boundary: the squared distance is formed exactly as the
   reference does it, with round-to-nearest intrinsics so the compiler can't fuse
   anything: dx*dx and dy*dy are each rounded, then added. A pair at distance
   exactly 5 (the example) is therefore excluded, just like in PyTorch. The update
   uses the same float32 operation order: (sum/count - v) * 0.05f + v, then p + new_v.
 - pair_pass: grid (agent blocks of 128, S source splits). Agents are staged
   through shared memory as float4 (x, y, vx, vy). Each thread owns one agent i
   and tests the tile against it. Neighbor velocity sums go into double (only
   true neighbors are summed, which is rare at this density), so the sum is
   essentially exact. j == i is skipped, matching the reference's diagonal fill.
   Each split writes (sum_vx, sum_vy, count) to scratch.
 - finalize: per agent, adds the S partials in a fixed order, rounds the sum to
   float as the reference's float matmul would (up to rounding), divides by the
   count, and applies the update.
At the benchmark: 10^8 pair tests, about 0.1 ms; S is chosen so there are ≥ 512 blocks.
For N = 100,000 the same kernel does 10^10 tests (tens of ms), and stays exact.
*/

#include <cuda_runtime.h>

#define T 128

__global__ void __launch_bounds__(T)
pair_pass(const float4* __restrict__ ag, double2* __restrict__ psum, int* __restrict__ pcnt,
          int N, int jPerSplit)
{
    __shared__ float4 tile[T];
    const int i = blockIdx.x * T + threadIdx.x;
    const bool active = i < N;
    const float4 me = active ? ag[i] : make_float4(0.f, 0.f, 0.f, 0.f);

    const int j0 = blockIdx.y * jPerSplit;
    const int j1 = min(j0 + jPerSplit, N);
    double sx = 0.0, sy = 0.0;
    int cnt = 0;

    for (int base = j0; base < j1; base += T) {
        const int j = base + threadIdx.x;
        if (j < j1) tile[threadIdx.x] = ag[j];
        __syncthreads();
        const int n = min(T, j1 - base);
        #pragma unroll 4
        for (int k = 0; k < n; ++k) {
            const float4 o = tile[k];
            const float dx = __fsub_rn(me.x, o.x);
            const float dy = __fsub_rn(me.y, o.y);
            const float d2 = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
            if (d2 < 25.0f && (base + k) != i) {
                sx += (double)o.z;
                sy += (double)o.w;
                ++cnt;
            }
        }
        __syncthreads();
    }
    if (active) {
        psum[(size_t)blockIdx.y * N + i] = make_double2(sx, sy);
        pcnt[(size_t)blockIdx.y * N + i] = cnt;
    }
}

__global__ void finalize(const float4* __restrict__ ag, const double2* __restrict__ psum,
                         const int* __restrict__ pcnt, float4* __restrict__ out, int N, int S)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double sx = 0.0, sy = 0.0;
    int cnt = 0;
    for (int s = 0; s < S; ++s) {
        const double2 p = psum[(size_t)s * N + i];
        sx += p.x; sy += p.y; cnt += pcnt[(size_t)s * N + i];
    }
    const float4 a = ag[i];
    float ax = a.z, ay = a.w;                     // no neighbors: avg = own velocity
    if (cnt > 0) {
        const float c = (float)cnt;
        ax = __fdiv_rn((float)sx, c);
        ay = __fdiv_rn((float)sy, c);
    }
    const float nvx = __fadd_rn(a.z, __fmul_rn(0.05f, __fsub_rn(ax, a.z)));
    const float nvy = __fadd_rn(a.w, __fmul_rn(0.05f, __fsub_rn(ay, a.w)));
    out[i] = make_float4(__fadd_rn(a.x, nvx), __fadd_rn(a.y, nvy), nvx, nvy);
}

static void*  g_ws = nullptr;
static size_t g_bytes = 0;

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N)
{
    if (N <= 0) return;
    const int iBlocks = (N + T - 1) / T;
    const int jTiles = iBlocks;
    int S = (512 + iBlocks - 1) / iBlocks;
    S = S < 1 ? 1 : (S > jTiles ? jTiles : S);
    const int jPerSplit = ((jTiles + S - 1) / S) * T;
    S = (N + jPerSplit - 1) / jPerSplit;

    const size_t need = (size_t)S * N * (sizeof(double2) + sizeof(int));
    if (need > g_bytes) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, need);
        g_bytes = need;
    }
    double2* psum = reinterpret_cast<double2*>(g_ws);
    int* pcnt = reinterpret_cast<int*>(psum + (size_t)S * N);

    const float4* ag = reinterpret_cast<const float4*>(agents);   // 16 B aligned (cudaMalloc)
    pair_pass<<<dim3(iBlocks, S), T>>>(ag, psum, pcnt, N, jPerSplit);
    finalize<<<(N + 255) / 256, 256>>>(ag, psum, pcnt, reinterpret_cast<float4*>(agents_next), N, S);
    cudaDeviceSynchronize();
}
