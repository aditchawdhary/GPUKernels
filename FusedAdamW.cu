#include <cuda_runtime.h>
#include <math.h>

// One fused AdamW step: each element is read once (params, grad, m, v) and
// written once (params, m, v). Memory-bound, so: float4 accesses + grid-stride loop.

struct AdamWArgs {
    float lr, beta1, beta2, omb1, omb2, eps, decay, bc1, bc2;
};

__device__ __forceinline__ void adamw_elem(float& p, float g, float& mm, float& vv,
                                           const AdamWArgs& a) {
    p *= a.decay;                                   // decoupled weight decay (pre-update p)
    mm = a.beta1 * mm + a.omb1 * g;
    vv = a.beta2 * vv + a.omb2 * g * g;
    const float m_hat = mm / a.bc1;
    const float v_hat = vv / a.bc2;
    p -= a.lr * m_hat / (sqrtf(v_hat) + a.eps);
}

__global__ void adamw_kernel(float* __restrict__ params, const float* __restrict__ grad,
                             float* __restrict__ m, float* __restrict__ v, int N, AdamWArgs a) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Vector body (cudaMalloc pointers are 256-byte aligned).
    const int n4 = N >> 2;
    float4* p4 = reinterpret_cast<float4*>(params);
    const float4* g4 = reinterpret_cast<const float4*>(grad);
    float4* m4 = reinterpret_cast<float4*>(m);
    float4* v4 = reinterpret_cast<float4*>(v);

    for (int i = tid; i < n4; i += stride) {
        float4 p = p4[i], g = __ldg(g4 + i), mm = m4[i], vv = v4[i];
        adamw_elem(p.x, g.x, mm.x, vv.x, a);
        adamw_elem(p.y, g.y, mm.y, vv.y, a);
        adamw_elem(p.z, g.z, mm.z, vv.z, a);
        adamw_elem(p.w, g.w, mm.w, vv.w, a);
        p4[i] = p; m4[i] = mm; v4[i] = vv;
    }

    // Scalar tail (at most 3 elements).
    for (int i = (n4 << 2) + tid; i < N; i += stride) {
        float p = params[i], mm = m[i], vv = v[i];
        adamw_elem(p, __ldg(grad + i), mm, vv, a);
        params[i] = p; m[i] = mm; v[i] = vv;
    }
}

// params, grad, m, v are device pointers
extern "C" void solve(float* params, const float* grad, float* m, float* v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) {
    AdamWArgs a;
    a.lr = lr;
    a.beta1 = beta1;
    a.beta2 = beta2;
    a.omb1 = (float)(1.0 - (double)beta1);
    a.omb2 = (float)(1.0 - (double)beta2);
    a.eps = eps;
    a.decay = (float)(1.0 - (double)lr * (double)weight_decay);
    // Bias corrections in double on the host (beta^t for large t).
    a.bc1 = (float)(1.0 - pow((double)beta1, (double)t));
    a.bc2 = (float)(1.0 - pow((double)beta2, (double)t));

    static int numSMs = 0;
    if (!numSMs) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
        if (numSMs <= 0) numSMs = 80;
    }

    const int threads = 256;
    int blocks = ((N >> 2) + threads - 1) / threads;
    if (blocks > numSMs * 16) blocks = numSMs * 16;
    if (blocks < 1) blocks = 1;

    adamw_kernel<<<blocks, threads>>>(params, grad, m, v, N, a);
    cudaDeviceSynchronize();
}
