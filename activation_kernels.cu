#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

/* namespace vllm: Groups related code together to avoid 
 * naming conflicts, like folders for code organization.
*/ 
namespace vllm {

/* template <typename scalar_t>
 * Creates generic code that works with different data types (float, half-precision, etc.).
 * The actual type is specified when you use the template.
 *
 * scalar_t
 * A placeholder for the actual data type (could be float, double, half, etc.)
*/ 
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
/* __device__
 * Function runs on the GPU and can only be called from GPU code.
 *
 * __forceinline__
 * Tells the compiler to insert the function's code directly where it's 
 * called (avoiding function call overhead for tiny functions).
 *
*/ 
__device__ __forceinline__ scalar_t compute(const scalar_t& x, 
                                            const scalar_t& y) {
    return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
  }

// Activation and gating kernel template.

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
/* __global__
 * Kernel function that runs on GPU but is called from CPU code. This is your entry point for GPU computation.
 *
 * __restrict__
 * Compiler hint that this pointer doesn't alias with others (no overlap in memory), enabling optimizations.
 *
*/ 
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ input,
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel (const T &x) {
  // x * sigmoid(x)
  return (T) ( ((float)x) / (1.0f + expf((float) -x)) );
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T &x) {
  // Equivalent to PyTorch GELU with "None" approximation.
  // Refer to 
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float) x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation
  // Refer to
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float) x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA + x_cube);
  return (T)(0.5f * f * (1.0f + ::tanh(inner)));
}

} // namespace vllm


// Launch activation 
// Use ACT_FIRST (bool) indicating whether to apply the 
// activation fn first
# define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                 \
    int d = input.size(-1) / 2;                                           \
    int64_t num_tokens = input.numel() / input.size(-1);                  \
    dim3 grid(num_tokens);                                                \
    dim3 block(std::min(d, 1024));                                        \
    if (num_tokens == 0) {                                                \
      return;                                                             \
    }                                                                     \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));     \
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();         \
    VLLM_DISPATCH_FLOATING_TYPES(                                         \
        input.scalar_type(), "act_and_mul_kernel", [&] {                  \
          vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST> \
          <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),          \
                                       input.data_ptr<scalar_t>(), d);    \
        });
    
    void silu_and_mul(torch::Tensor& out,   // [..., d]
                      torch::Tensor& input) // [..., 2 * d]
    {
      LAUNCH_ACTIVATION_GATE_KERNEL(vllm:silu_kernel, true);
    }

    void mul_and_silu(torch:: Tensor& out,   // [..., d]
                      torch:: Tensor& input) // [..., 2 * d]
    {
      // The difference between mul_and_silu and silu_and_mul is that 
      // mul_and_silu applies the silu to the latter half of the input.
      LAUNCH_ACTIVATION_GATE_KERNEL(vllm:gelu_kernel, true);
    }

    void gelu_tanh_and_mul(torch::Tensor& out,  // [..., d] 
                           torch::Tensor& input // [..., 2 * d] 
                          ) 
    {
      LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
    }
