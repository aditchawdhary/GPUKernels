/*
BFS Shortest Path on a grid
grid: rows*cols int32 (0 = free, 1 = obstacle). 4-neighbor moves.
result[0] = number of steps on a shortest path from start to end, 0 if start == end,
-1 if the end is unreachable. (Spec: leetgpu-challenges #46. rows, cols ≤ 1000;
performance at 500 x 500 with ~50% random obstacles.)
*/

/*
Approach: level-synchronous frontier BFS in a single persistent 1024-thread block.

 - BFS is a sequence of dependent levels, and a maze path can be thousands of levels
   long. One kernel launch per level would cost several µs each, so the whole
   search runs inside one block, separated by __syncthreads(), which costs
   nanoseconds.
 - State in global scratch (L2-resident, ≤ 8 MB):
     dist[]  : -1 = unvisited (set with one memset), else the BFS level.
     queue[] : every cell is enqueued at most once, so one array of rows*cols ints
               holds the whole BFS in order. Level L is the slice [lo, hi).
 - Per level, threads stride over the frontier slice. For each free, in-bounds
   neighbor they claim it with atomicCAS(dist, -1, L+1). Only the winner appends
   it, through a shared-memory tail counter, so no cell is ever expanded twice.
   Reaching the end cell sets a shared flag, and the search stops after that level.
   That level is the shortest distance, because BFS discovers cells in order of
   distance.
 - Two barriers per level: one after expansion (the new tail is final), one after
   everyone has read the new tail (so the next level's appends can't race the read).
Cost: O(rows*cols) work in total, about 1 ms or less at 500 x 500.
*/

#include <cuda_runtime.h>

#define THREADS 1024

__global__ void __launch_bounds__(THREADS)
bfs_grid(const int* __restrict__ grid, int* __restrict__ dist, int* __restrict__ queue,
         int* __restrict__ result, int rows, int cols, int start, int end)
{
    __shared__ int tail;
    __shared__ int found;

    if (threadIdx.x == 0) {
        queue[0] = start;
        dist[start] = 0;
        tail = 1;
        found = -1;
    }
    __syncthreads();

    int lo = 0, hi = 1, level = 0;
    while (lo < hi && found < 0) {
        for (int i = lo + threadIdx.x; i < hi; i += THREADS) {
            const int cell = queue[i];
            const int r = cell / cols, c = cell - r * cols;
            const int nb[4] = {cell - cols, cell + cols, cell - 1, cell + 1};
            const bool ok[4] = {r > 0, r + 1 < rows, c > 0, c + 1 < cols};
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                if (!ok[d]) continue;
                const int n = nb[d];
                if (__ldg(grid + n) != 0) continue;
                if (atomicCAS(dist + n, -1, level + 1) == -1) {   // first to reach n
                    if (n == end) found = level + 1;
                    queue[atomicAdd(&tail, 1)] = n;
                }
            }
        }
        __syncthreads();                 // all appends for this level are done
        lo = hi;
        hi = tail;
        ++level;
        __syncthreads();                 // everyone has read tail before new appends
    }

    if (threadIdx.x == 0) result[0] = found;
}

static int*   g_ws = nullptr;
static size_t g_cap = 0;

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row,
                      int start_col, int end_row, int end_col)
{
    const int start = start_row * cols + start_col;
    const int end = end_row * cols + end_col;
    if (start == end) {
        cudaMemset(result, 0, sizeof(int));
        cudaDeviceSynchronize();
        return;
    }
    const size_t cells = (size_t)rows * cols;
    if (2 * cells > g_cap) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, 2 * cells * sizeof(int));
        g_cap = 2 * cells;
    }
    int* dist = g_ws;
    int* queue = g_ws + cells;
    cudaMemsetAsync(dist, 0xFF, cells * sizeof(int));          // all -1
    bfs_grid<<<1, THREADS>>>(grid, dist, queue, result, rows, cols, start, end);
    cudaDeviceSynchronize();
}
