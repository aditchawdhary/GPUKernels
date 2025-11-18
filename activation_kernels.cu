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
    scalar_t* __restrict__ out,     // [..., d]
    scalar_t* __restrict__ input,   // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

/** __device__ mean will run on gpu, can be invoked by gpu
  * template <typename T> means will run on generic type T
  *
*/ 
template <typename T>
__device__ __forceinline__ T silu_kernel (const T &x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float) -x)));
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
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanh(inner)));
}

} // namespace vllm


// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the 
// activation fn first

/**
* Creates a macro(copy-paste template), kernel is the activation type, ACT_FIRST is a boolean flag
 
* input, out, and d (well, d is computed from input) are NOT parameters to the macro. 
* They're just... expected to exist in the scope where you use this macro.
* This is a macro footgun. The macro assumes that wherever you paste it, you already have variables named:

* void some_function(torch::Tensor& out, torch::Tensor& input) {
* // Now 'input' and 'out' exist in this scope  
* LAUNCH_ACTIVATION_GATE_KERNEL(SiLU, true);
* // ↑ Macro expands here and uses 'input' and 'out' from surrounding scope
* }

* Why is this terrible?
* Hidden dependencies: The macro silently requires input and out to exist
* No type safety: Compiler won't catch missing variables until after macro expansion
* Confusing AF: Exactly what you just experienced

* Why do they do it anyway?
* Because writing this kernel launch boilerplate manually every time would be:
* Repetitive (same grid/block setup every time)
* Error-prone (easy to mess up the dispatch or stream management)
* Hard to maintain (if you change one, you have to change all of them)
* So they traded clarity for convenience. Classic C++ macro problem.
*/ 
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                \
    // Gets the size of the last dimension of the tensor.
    int d = input.size(-1) / 2;                                         \
    // numel() = number of elements (total count of all values)
    // input shape: [2, 512, 4096]
    // 2 * 512 * 4096 = 4,194,304 total elements
    // num_tokens = 2 * 512 * 4096 / 4096 = 1024
    int64_t num_tokens = input.numel() / input.size(-1);                \
    // `dim3`, `cudaStream_t`, and `<<<>>>` come from CUDA 
    // runtime headers, which are included by:
    // <ATen/cuda/CUDAContext.h>  →  <cuda_runtime.h>  →  
    // defines dim3, cudaStream_t
    dim3 grid(num_tokens);                                              \
    dim3 block(std::min(d, 1024));                                      \
    if (num_tokens == 0) {                                              \
      return;                                                           \
    }                                  
    /*  RAII (Resource Acquisition Is Initialization) pattern
        Scope-Bound Resource Management, 
        https://stackoverflow.com/questions/2321511/what-is-meant-by-resource-acquisition-is-initialization-raii
    
        TL;DR:

        > Active GPU = which GPU will execute operations (context)
        > Data location = which GPU's memory holds the data (physical location)
        > They can be different! (that's the problem)
        > The guard makes them the same (that's the solution)
        > The guard ensures: "Active GPU = GPU where data lives"

        > Setup: 4 GPUs, tensor on GPU 2
        > You have 4 GPUs: GPU 0, GPU 1, GPU 2, GPU 3
        > Currently active GPU: GPU 0; input tensor lives on: GPU 2
        > Switch to GPU 2 (where input lives)
        > Get GPU 2's stream
        > Launch kernel on GPU 2 (processes GPU 2's data)
        > Restore to original GPU (GPU 0)
        > GPUs 0, 1, 3 are not involved at all
        > The guard ensures the kernel runs on the correct GPU in a multi-GPU system.
    
        Line-by-line execution:
        Line 1:
        > const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
        What happens:
        > device_of(input) returns Device(2) (input is on GPU 2)
        > Guard saves current active GPU (GPU 0)
        > Guard switches active GPU to GPU 2 via cudaSetDevice(2)
        > Now all subsequent CUDA operations target GPU 2
        State:
        > Previous GPU: 0, Active GPU: 2 ✓, Kernel will launch on: GPU 2

        Line 2:
        > cppconst cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        What happens:
        > Gets the current CUDA stream for GPU 2
        > PyTorch maintains separate stream contexts per GPU
        > Returns the stream handle for GPU 2 (not GPU 0!)
        State:
        > Active GPU: 2, Stream: GPU 2's current stream ✓

        Line 3-4:
        > VLLM_DISPATCH_FLOATING_TYPES(
        >   input.scalar_type(), "act_and_mul_kernel", [&] {
        What happens:
        > Checks input.scalar_type() (e.g., torch::kFloat32)
        > Macro expands and calls the lambda with scalar_t = float
        > This is just compile-time type dispatch, no GPU operations
        State:
        > Active GPU: 2,            Stream: GPU 2's stream
        > Type resolved: scalar_t = float (or half, bfloat16, etc.)

        Line 5-6:
        > cppvllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>
        >   <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),
        >                               input.data_ptr<scalar_t>(), d);
        What happens:
        > Kernel launch configuration:
        > grid: Number of blocks,   block: Threads per block
        > 0: No shared memory,      stream: GPU 2's stream ✓

        Kernel launches on GPU 2 because:
        > Active device is GPU 2 (set by device_guard)
        > Stream belongs to GPU 2
        > Kernels always launch on the active GPU

        Memory pointers:
        > out.data_ptr<scalar_t>(): Points to memory on GPU 2
        > input.data_ptr<scalar_t>(): Points to memory on GPU 2
        > Kernel reads/writes GPU 2's memory ✓

        Asynchronous execution:
        > Kernel is queued to GPU 2's stream
        > CPU continues immediately (non-blocking)
        > Kernel executes on GPU 2 in the background
        State:
        > Kernel running on: GPU 2 ✓,  GPU 0, 1, 3: Idle (not involved)
        After the scope ends:
        > cpp}  // device_guard destructor runs here
        ```
        What happens:
        1. Guard destructor runs
        2. Restores active GPU to GPU 0
        3. Kernel on GPU 2 **still running asynchronously** (unaffected)

        State:
        - Active GPU: 0 (restored)
        - GPU 2: Kernel still executing in background

        ## Complete timeline with 4 GPUs:
        ```
        Before macro:
          Active GPU: 0
          GPU 0,1,3: [idle]  GPU 2: [has input tensor]
          
        device_guard construction:
          Active GPU: 2 ← switched
          GPU 0,1,3: [idle]  GPU 2: [active] ← now current
          
        Kernel launch:
          Active GPU: 2
          GPU 0, 1, 3: [idle] GPU 2: [kernel executing] ← work happening here

        device_guard destruction:
          Active GPU: 0 ← restored
          GPU 0,1,3: [idle]
          GPU 2: [kernel still running] ← async work continues
          
        What if input was on GPU 1?
        Same logic, just different device:
        cpp// input on GPU 1
        const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
        // Switches to GPU 1
        // Kernel launches on GPU 1
        // Restores to GPU 0 when done

        What if you had inputs on DIFFERENT GPUs?
        cpp// BAD: input on GPU 2, output on GPU 3
        const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
        // Switches to GPU 2
        vllm::act_and_mul_kernel<<<grid, block, 0, stream>>>(
            out.data_ptr(),    // on GPU 3 ← WRONG!
            input.data_ptr(),  // on GPU 2 ← correct
            d);
        // Kernel runs on GPU 2, tries to write to GPU 3's memory → CRASH!
        The code assumes input and out are on the same GPU.
        */ 
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));   \
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();       \
    VLLM_DISPATCH_FLOATING_TYPES(                                       \
      input.scalar_type(), "act_and_mul_kernel", [&] {                  \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST> \
          <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),        \
                                       input.data_ptr<scalar_t>(), d);  \
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
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm:silu_kernel, false);
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
}

void gelu_tanh_and_mul(torch::Tensor& out,  // [..., d] 
                        torch::Tensor& input) // [..., 2 * d] 
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
}

namespace vllm {

template <typename T>
__device__ __forceinline__ T fatrelu_kernel(const T& x, const float threshold) {
    const float f = (float)x;
    return (T)(f > threshold ? f : 0.0f);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&, const float)>
__global__ void act_and_mul_kernel_with_param(
  scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const int d,
  const float param) {
    const int64_t token_idx = blockIdx.x;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
      const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
      out[token_idx * d + idx] = ACT_FN(x, param) * y;
    }
  }

template <typename T>
__device__ __forceinline__ T swigluoai_and_mul(const T& gate, const T& up, 
                                              float alpha, float limit) {
  
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up = 
    up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + expf(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
__global__ void swigluai_and_mul_kernel(
    scalar_t* __restrict__ out,         // [..., d]
    const scalar_t* __restrict__ input, // [..., 2, d]
    const int d, const float alpha, const float limit) {
  const int64_t token_idx = blockIdx.x;
  // TODO: Vectorize loads and stores.
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // gate = x[..., ::2] (even indices)
    const scalar_t gate = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx]);
    // up = x[..., 1::2] (odd indices)
    const scalar_t up = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx + 1]);

    out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
  }
}

} //namespace vllm 

#define LAUNCH_KERNEL_GATE_KERNEL_WITH_PARAM(KERNEL, PARAM)             \
  int d = input.size(-1) / 2;                                           \
  int64_t num_tokens = input.numel() / input.size(-1);                  \
  dim3 grid(num_tokens);                                                \
  dim3 block(std::min(d, 1024));                                        \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));     \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();         \
  VLLM_DISPATCH_FLOATING_TYPES(                                         \
      input.scalar_type(), "act_and_mul_kernel_with_param", [&] {       \
        vllm::act_and_mul_kernel_with_param<scalar_t, KERNEL<scalar_t>> \
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),          \
                                    input.data_ptr<scalar_t>(), d,      \
                                    PARAM);                             \
      });

#define LAUNCH_SIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                          \
      int d = input.size(-1) / 2;                                               \
      int64_t num_tokens = input.numel() / input.size(-1);                      \
      dim3 grid(num_tokens);                                                    \
      dim3 block(std::min(d, 1024));                                            \
      const at::cuda::OptionalCUDAGuard device_guard(device_of(input));         \
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();             \
      VLLM_DISPATCH_FLOATINT_TYPES(                                             \
          input.scalar_type(), "clamp_swiglu_kernel_with_params", [&] {         \
            vllm::swigluai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>           \
            <<<grid, block, 0, stream>>>(out.data<scalar_t>(),                  \
                                         input.data_ptr<scalar_t>(), d, ALPHA,  \
                                        LIMIT);                                 \
          });

void fatrelu_and_mul(torch::Torch& out,     // [..., d]
                     torch::Tensor& input,  // [..., 2 * d]
                     double threshold) {
  LAUNCH_SIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}

namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
  scalar_t* __restrict__ out,
  const scalar_t* 
)

} // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                      \
  int d = input.size(-1);                                                     \
  int64_t num_tokens = input.numel() / d;                                     \
  dim3 grid(num_tokens);                                                      \
  dim3 block(std::min(d, 1024));                                              \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));           \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();               \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&]{ \
    vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                       \
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),                \
                                     input.data_ptr<scalar_t>(), d);          \
  });                                                                         \

namespace vllm {

template <typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t = 
    (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + expf(-1.702f * (float)x)));
}

} // namespace vllm

void gelu_new(torch::Tensor& out,        // [..., d]
              torch::Tensor& input)      // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(torch::Tensor& out,       // [..., d]
               torch::Tensor& input)     // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

void gelu_quick(torch::Tensor& out,      // [..., d]
                torch::Tensor& input)    // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm:gelu_quick_kernel);
}