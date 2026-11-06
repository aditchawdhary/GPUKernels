#include <cuda_runtime.h>

#define THREADS 256

// One thread per (row, j) pair, where j < D/2.
__global__ void rope_kernel(const float* Q, const float* cos, const float* sin,
                            float* output, int M, int D) {
    int half = D / 2;
    long long idx   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long pairs = (long long)M * half;
    if (idx >= pairs) return;

    long long row = idx / half;
    int j         = idx % half;

    long long a = row * D + j;      // first-half element
    long long b = a + half;         // its partner in the second half

    float q1 = Q[a];
    float q2 = Q[b];

    // Half-split layout: cos[a] == cos[b] and sin[a] == sin[b],
    // so we only need to read them once.
    float c = cos[a];
    float s = sin[a];

    output[a] = q1 * c - q2 * s;
    output[b] = q2 * c + q1 * s;
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    long long pairs = (long long)M * (D / 2);
    int blocks = (int)((pairs + THREADS - 1) / THREADS);
    rope_kernel<<<blocks, THREADS>>>(Q, cos, sin, output, M, D);
}
