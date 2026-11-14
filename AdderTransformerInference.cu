#include <cuda_runtime.h>
#include <math.h>

// Batched autoregressive inference of the 10-parameter adder transformer.
//
// Only the last position's logits are needed at each decode step, and everything except
// attention is position-wise, while attention is causal. So the last row of the full forward
// pass is exactly: embed(last token) -> unit RMSNorm -> its query attends over keys/values of
// the whole current sequence -> residual -> SwiGLU carry MLP -> residual -> RMSNorm -> tied
// output projection. Keys and values depend only on (token, position), so each step costs
// one 2-D attention over <= 42 keys. One thread runs one sequence through all 11 steps.
//
// All arithmetic is float32 and mirrors the reference's operation order (embedding,
// RMSNorms, RoPE with k = [kn, 0], softmax with max subtraction, silu = x / (1 + e^-x)) so the
// argmax tokens fed back into the sequence match.

#define PROMPT_LEN 31
#define STEPS 11
#define MAXLEN (PROMPT_LEN + STEPS)
#define VOCAB 10
#define RMS_EPS 1e-6f
#define EMBED_CONST 1000.0f

struct Consts {
    float omega;        // 2*pi/19
    float attn_scale;   // (1/sqrt 2) * S^2
};

__device__ __forceinline__ void unit_rms(float x0, float x1, float& y0, float& y1) {
    float r = rsqrtf((x0 * x0 + x1 * x1) * 0.5f + RMS_EPS);
    y0 = x0 * r;
    y1 = x1 * r;
}

__global__ void __launch_bounds__(128)
adder_kernel(const int* __restrict__ prompts, float* __restrict__ output,
             const float* __restrict__ weights, int batch, Consts cst) {
    // per-token tables (shared by the block)
    __shared__ float E0[VOCAB], E1[VOCAB];      // embedding table
    __shared__ float HN0[VOCAB];                // component 0 of the unit-RMSNormed embedding
    __shared__ float KN[VOCAB];                 // normalised key component 0 (key = [kn, 0])
    __shared__ float VV[VOCAB];                 // value component 0
    __shared__ float CS[MAXLEN], SN[MAXLEN];    // RoPE cos / sin by position
    __shared__ float W[10];

    if (threadIdx.x < 10) W[threadIdx.x] = weights[threadIdx.x];
    __syncthreads();
    if (threadIdx.x < VOCAB) {
        int d = threadIdx.x;
        float df = (float)d;
        float e0 = W[0] - (W[1] * df) * df;
        float e1 = -df;
        E0[d] = e0; E1[d] = e1;
        float h0, h1;
        unit_rms(e0, e1, h0, h1);
        HN0[d] = h0;
        float k0, k1;
        unit_rms(h0, 0.0f, k0, k1);
        KN[d] = k0;
        VV[d] = h1 * W[4];
    }
    if (threadIdx.x < MAXLEN) {
        float a = (float)threadIdx.x * cst.omega;
        CS[threadIdx.x] = cosf(a);
        SN[threadIdx.x] = sinf(a);
    }
    __syncthreads();

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    unsigned char seq[MAXLEN];
    const int* pr = prompts + (size_t)b * PROMPT_LEN;
#pragma unroll
    for (int i = 0; i < PROMPT_LEN; i++) seq[i] = (unsigned char)__ldg(&pr[i]);

    const float qw0 = W[2], qw1 = W[3];
    const float ga = W[5], gc = W[6], carry = W[7], nw0 = W[8], nw1 = W[9];
    const float ga2 = ga - gc / EMBED_CONST;
    float* out = output + (size_t)b * STEPS * VOCAB;

    for (int step = 0; step < STEPS; step++) {
        const int L = PROMPT_LEN + step;
        const int last = L - 1;
        const int tok = seq[last];

        // query at the last position
        float hn0 = HN0[tok];
        float q0, q1;
        unit_rms(hn0 * qw0, hn0 * qw1, q0, q1);
        const float c = CS[last], s = SN[last];
        const float qr0 = q0 * c - q1 * s;
        const float qr1 = q0 * s + q1 * c;

        // causal attention over positions 0..last (two passes: max, then exp-sum)
        float mx = -INFINITY;
        for (int j = 0; j < L; j++) {
            float kn = KN[seq[j]];
            float sc = (qr0 * (kn * CS[j]) + qr1 * (kn * SN[j])) * cst.attn_scale;
            mx = fmaxf(mx, sc);
        }
        float sum = 0.0f, acc = 0.0f;
        for (int j = 0; j < L; j++) {
            int t = seq[j];
            float kn = KN[t];
            float sc = (qr0 * (kn * CS[j]) + qr1 * (kn * SN[j])) * cst.attn_scale;
            float e = expf(sc - mx);
            sum += e;
            acc += e * VV[t];
        }
        const float attn0 = acc / sum;

        // residual with O projection [0, attn0]
        float x0 = E0[tok];
        float x1 = E1[tok] + attn0;

        // SwiGLU carry MLP
        float m0, m1;
        unit_rms(x0, x1, m0, m1);
        float g0 = m0 * ga + m1 * gc;
        float g1 = m0 * ga2 + m1 * gc;
        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        x1 += carry * (s1 * m0 - s0 * m0);

        // final RMSNorm with learned weight
        float rms = sqrtf((x0 * x0 + x1 * x1) * 0.5f + RMS_EPS);
        float f0 = (x0 / rms) * nw0;
        float f1 = (x1 / rms) * nw1;

        // tied output projection + argmax (first max wins, like torch.argmax)
        int best = 0;
        float bestv = -INFINITY;
#pragma unroll
        for (int d = 0; d < VOCAB; d++) {
            float lg = f0 * E0[d] + f1 * E1[d];
            out[step * VOCAB + d] = lg;
            if (lg > bestv) { bestv = lg; best = d; }
        }
        seq[L] = (unsigned char)best;
    }
}

// prompts, output, weights are device pointers
extern "C" void solve(const int* prompts, float* output, const float* weights, int batch_size) {
    const double omega = 2.0 * M_PI / 19.0;
    const double amp = log(10.0) / (cos(omega * 0.3) - cos(omega * 0.7));
    const double qk_norm_scale = sqrt(amp / sqrt(2.0));
    Consts c;
    c.omega = (float)omega;
    c.attn_scale = (float)((1.0 / sqrt(2.0)) * qk_norm_scale * qk_norm_scale);

    const int threads = 128;
    adder_kernel<<<(batch_size + threads - 1) / threads, threads>>>(prompts, output, weights,
                                                                    batch_size, c);
}
