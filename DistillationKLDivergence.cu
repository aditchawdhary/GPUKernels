#include <cuda_runtime.h>
#include <math.h>

// Forward KL per row:  KL(P || Q) = sum_i p_i (log p_i - log q_i)
//   p = softmax(teacher / T),  q = softmax(student / T)
//
// Single pass over both rows. With a = t/T, b = s/T, d = a - b:
//   KL = sum_i p_i d_i - LSE(a) + LSE(b)
//      = W / Sa + (mb - ma) + log(Sb) - log(Sa)
// where (online, max-rescaled)
//   ma = max a,  Sa = sum e^(a-ma),  W = sum e^(a-ma) * (a-b)
//   mb = max b,  Sb = sum e^(b-mb)

#define THREADS 512
#define WARPS   (THREADS / 32)

struct KLState {
    float ma, Sa, W;   // teacher max, teacher sum-exp, weighted (a-b) sum
    float mb, Sb;      // student max, student sum-exp
};

__device__ __forceinline__ void merge(KLState& x, const KLState& y)
{
    if (y.ma != -INFINITY) {
        if (x.ma == -INFINITY) { x.ma = y.ma; x.Sa = y.Sa; x.W = y.W; }
        else {
            float m  = fmaxf(x.ma, y.ma);
            float f1 = __expf(x.ma - m), f2 = __expf(y.ma - m);
            x.Sa = x.Sa * f1 + y.Sa * f2;
            x.W  = x.W  * f1 + y.W  * f2;
            x.ma = m;
        }
    }
    if (y.mb != -INFINITY) {
        if (x.mb == -INFINITY) { x.mb = y.mb; x.Sb = y.Sb; }
        else {
            float m = fmaxf(x.mb, y.mb);
            x.Sb = x.Sb * __expf(x.mb - m) + y.Sb * __expf(y.mb - m);
            x.mb = m;
        }
    }
}

// Absorb 4 elements at once: one rescale per batch instead of per element.
__device__ __forceinline__ void absorb4(KLState& st, float4 t, float4 s, float invT)
{
    float a0 = t.x * invT, a1 = t.y * invT, a2 = t.z * invT, a3 = t.w * invT;
    float b0 = s.x * invT, b1 = s.y * invT, b2 = s.z * invT, b3 = s.w * invT;

    float na = fmaxf(st.ma, fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)));
    float nb = fmaxf(st.mb, fmaxf(fmaxf(b0, b1), fmaxf(b2, b3)));

    float ra = __expf(st.ma - na);   // exp(-inf) = 0 on first batch
    float rb = __expf(st.mb - nb);

    float e0 = __expf(a0 - na), e1 = __expf(a1 - na);
    float e2 = __expf(a2 - na), e3 = __expf(a3 - na);

    st.Sa = st.Sa * ra + ((e0 + e1) + (e2 + e3));
    st.W  = st.W  * ra + (e0 * (a0 - b0) + e1 * (a1 - b1))
                       + (e2 * (a2 - b2) + e3 * (a3 - b3));
    st.ma = na;

    st.Sb = st.Sb * rb + ((__expf(b0 - nb) + __expf(b1 - nb))
                        + (__expf(b2 - nb) + __expf(b3 - nb)));
    st.mb = nb;
}

__device__ __forceinline__ void absorb1(KLState& st, float t, float s, float invT)
{
    float a = t * invT, b = s * invT;
    float na = fmaxf(st.ma, a), nb = fmaxf(st.mb, b);
    float ra = __expf(st.ma - na), rb = __expf(st.mb - nb);
    float e  = __expf(a - na);
    st.Sa = st.Sa * ra + e;
    st.W  = st.W  * ra + e * (a - b);
    st.ma = na;
    st.Sb = st.Sb * rb + __expf(b - nb);
    st.mb = nb;
}

__device__ __forceinline__ void warpMerge(KLState& st)
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        KLState o;
        o.ma = __shfl_down_sync(0xffffffff, st.ma, off);
        o.Sa = __shfl_down_sync(0xffffffff, st.Sa, off);
        o.W  = __shfl_down_sync(0xffffffff, st.W,  off);
        o.mb = __shfl_down_sync(0xffffffff, st.mb, off);
        o.Sb = __shfl_down_sync(0xffffffff, st.Sb, off);
        merge(st, o);
    }
}

__global__ void __launch_bounds__(THREADS)
kl_kernel(const float* __restrict__ teacher, const float* __restrict__ student,
          float* __restrict__ kl, int V, float invT)
{
    const int row = blockIdx.x;
    const float* t = teacher + (size_t)row * V;
    const float* s = student + (size_t)row * V;

    KLState st = { -INFINITY, 0.f, 0.f, -INFINITY, 0.f };

    if ((V & 3) == 0) {
        const float4* t4 = reinterpret_cast<const float4*>(t);
        const float4* s4 = reinterpret_cast<const float4*>(s);
        const int V4 = V >> 2;
        int i = threadIdx.x;
        for (; i + THREADS < V4; i += 2 * THREADS) {
            float4 ta = __ldcs(t4 + i), tb = __ldcs(t4 + i + THREADS);
            float4 sa = __ldcs(s4 + i), sb = __ldcs(s4 + i + THREADS);
            absorb4(st, ta, sa, invT);
            absorb4(st, tb, sb, invT);
        }
        if (i < V4) absorb4(st, __ldcs(t4 + i), __ldcs(s4 + i), invT);
    } else {
        for (int i = threadIdx.x; i < V; i += THREADS)
            absorb1(st, __ldcs(t + i), __ldcs(s + i), invT);
    }

    warpMerge(st);

    __shared__ KLState sh[WARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = st;
    __syncthreads();

    if (warp == 0) {
        st = (lane < WARPS) ? sh[lane] : KLState{ -INFINITY, 0.f, 0.f, -INFINITY, 0.f };
        warpMerge(st);
        if (lane == 0)
            kl[row] = st.W / st.Sa + (st.mb - st.ma) + (logf(st.Sb) - logf(st.Sa));
    }
}

// teacher_logits, student_logits, kl are device pointers
extern "C" void solve(const float* teacher_logits, const float* student_logits,
                      float* kl, int N, int V, float temperature)
{
    kl_kernel<<<N, THREADS>>>(teacher_logits, student_logits, kl, V, 1.0f / temperature);
    cudaDeviceSynchronize();
}
