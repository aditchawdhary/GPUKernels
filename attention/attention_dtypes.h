#pragma once

#include "attention_generic.cuh"
#include "dtype_float16.cuh"
#include "dtype_float32.cuh"
#include "dtype_bfloat16.cuh"
#include "dtype_fp8.cuh"

/* This is a C++ header file that sets up the foundation for a 
CUDA-based attention mechanism implementation (likely for 
transformer models). Let me break down each part:

`#pragma once`
A header guard that ensures this file is only included once 
during compilation, preventing duplicate definitions.

The Include Files

`attention_generic.cuh`
- Contains generic/shared attention mechanism code
- The `.cuh` extension indicates it's a CUDA C++ header file

`dtype_float16.cuh`
- Implements 16-bit floating point (half precision) data type operations
- Used for memory-efficient GPU computations

`dtype_float32.cuh`
- Implements 32-bit floating point (single precision) operations
- Standard precision for many ML workloads

`dtype_bfloat16.cuh`
- Implements Brain Float 16 format (Google's alternative to FP16)
- Maintains FP32's range but with less precision, popular in deep learning

`dtype_fp8.cuh`
- Implements 8-bit floating point operations
- Emerging format for extreme memory/compute efficiency in modern GPUs

## Purpose

This header is setting up a multi-precision attention implementation. 
By including all these data type headers, the code can:
- Support attention operations across different precision levels
- Allow users to trade off between speed, memory usage, and accuracy
- Optimize for different GPU architectures (newer GPUs support FP8, for example)

This pattern is common in high-performance ML libraries like 
FlashAttention, xFormers, or custom transformer implementations.
*/

/*
Memory Usage by Data Type

| Type | Bits | Bytes | Memory vs FP32     |
|------|------|-------|--------------------|
| FP32 | 32   | 4     | 1× (baseline)      |
| FP16 | 16   | 2     | 0.5× (50% savings) |
| BF16 | 16   | 2     | 0.5× (50% savings) |
| FP8  | 8    | 1     | 0.25× (75% savings)|

For a model with 1 billion parameters:
- FP32: 4 GB
- FP16/BF16: 2 GB
- FP8: 1 GB

## Why BFloat16?
BFloat16 vs Float16 trade-off:
FP32:  1 sign | 8 exponent | 23 mantissa (precision)
FP16:  1 sign | 5 exponent | 10 mantissa
BF16:  1 sign | 8 exponent | 7 mantissa  ← Same range as FP32!

Key advantages:
- Same dynamic range as FP32 (can represent same min/max values)
- No overflow issues during training that FP16 often has
- Drop-in replacement - easier to convert FP32 code to BF16
- Training stability - works better for training than FP16 without loss scaling tricks
- Simpler code - just truncate FP32, no complex conversion logic

Popular in: Google TPUs, modern NVIDIA GPUs, training large language models

## Why FP8?
Two common FP8 formats:
- E4M3: 4 exponent bits, 3 mantissa bits (more precision, less range)
- E5M2: 5 exponent bits, 2 mantissa bits (more range, less precision)

Key advantages:
- 4× memory reduction vs FP32, 2× vs FP16/BF16
- Faster matrix multiplications on modern hardware (H100, H200 GPUs)
- Enables larger models or bigger batch sizes in same GPU memory
- Mixed precision: Often use FP8 for forward pass, higher precision for gradients

Trade-offs:
- Requires careful scaling and calibration
- Usually only for inference or selective layers during training
- Needs newer hardware support (NVIDIA Hopper architecture)

## Practical Use Case
A typical transformer attention layer might use:
- FP8: Matrix multiplications (Q, K, V projections) - 75% memory saved
- FP16/BF16: Intermediate calculations - balance speed and accuracy  
- FP32: Critical accumulations (softmax, layer norm) - maintain numerical stability

This mixed-precision approach maximizes both speed and 
memory efficiency while maintaining model quality.
*/