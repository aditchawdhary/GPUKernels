#include <cuda_runtime.h>

#define TX 32          // output cols per block
#define TY 8           // output rows per block
#define ZT 4           // output depths per block
#define KMAX 5
#define SW (TX + KMAX - 1)   // 36
#define SH (TY + KMAX - 1)   // 12

__constant__ float c_kernel[KMAX * KMAX * KMAX];

__global__ void conv3d_kernel(const float* __restrict__ input, float* __restrict__ output,
                              int ID, int IR, int IC,
                              int KD, int KR, int KC,
                              int OD, int OR, int OC)
{
    __shared__ float tile[SH][SW + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int oc = blockIdx.x * TX + tx;
    const int orow = blockIdx.y * TY + ty;
    const int z0 = blockIdx.z * ZT;

    const int tileW = TX + KC - 1;
    const int tileH = TY + KR - 1;
    const int colBase = blockIdx.x * TX;
    const int rowBase = blockIdx.y * TY;
    const size_t slice = (size_t)IR * IC;

    float acc[ZT];
    #pragma unroll
    for (int o = 0; o < ZT; ++o) acc[o] = 0.0f;

    // Input slices needed by outputs z0 .. z0+ZT-1
    const int dEnd = min(ID, z0 + ZT + KD - 1);
    for (int d = z0; d < dEnd; ++d) {
        // Cooperative load of this slice's tile + halo
        const float* src = input + (size_t)d * slice;
        for (int i = tid; i < tileH * tileW; i += TX * TY) {
            int r = i / tileW, c = i % tileW;
            int gr = rowBase + r, gc = colBase + c;
            tile[r][c] = (gr < IR && gc < IC) ? __ldg(&src[(size_t)gr * IC + gc]) : 0.0f;
        }
        __syncthreads();

        // This slice contributes to output depth z0+o with kd = d - (z0+o)
        #pragma unroll
        for (int o = 0; o < ZT; ++o) {
            int kd = d - (z0 + o);
            if (kd >= 0 && kd < KD) {
                // For a flipped (true) convolution use:
                // (KD-1-kd), (KR-1-kr), (KC-1-kc) in the weight index below.
                const float* w = c_kernel + kd * KR * KC;
                float s = 0.0f;
                for (int kr = 0; kr < KR; ++kr) {
                    #pragma unroll 5
                    for (int kc = 0; kc < KC; ++kc)
                        s = fmaf(tile[ty + kr][tx + kc], w[kr * KC + kc], s);
                }
                acc[o] += s;
            }
        }
        __syncthreads();
    }

    if (oc < OC && orow < OR) {
        #pragma unroll
        for (int o = 0; o < ZT; ++o) {
            int z = z0 + o;
            if (z < OD)
                output[(size_t)z * OR * OC + (size_t)orow * OC + oc] = acc[o];
        }
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols)
{
    const int OD = input_depth - kernel_depth + 1;
    const int OR = input_rows  - kernel_rows  + 1;
    const int OC = input_cols  - kernel_cols  + 1;
    if (OD <= 0 || OR <= 0 || OC <= 0) return;

    cudaMemcpyToSymbol(c_kernel, kernel,
                       sizeof(float) * kernel_depth * kernel_rows * kernel_cols,
                       0, cudaMemcpyDeviceToDevice);

    dim3 block(TX, TY);
    dim3 grid((OC + TX - 1) / TX, (OR + TY - 1) / TY, (OD + ZT - 1) / ZT);
    conv3d_kernel<<<grid, block>>>(input, output,
                                   input_depth, input_rows, input_cols,
                                   kernel_depth, kernel_rows, kernel_cols,
                                   OD, OR, OC);
    cudaDeviceSynchronize();
}
