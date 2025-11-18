/*
* Interface for KV (Key-Value) cache operations in vLLM.
* Explanation: Header guard that ensures this file is only included once 
* during compilation, preventing duplicate definitions.
* 
* Explanation: Includes all PyTorch C++ API headers, providing access 
* to torch::Tensor and other PyTorch types.
* 
* Explanation: Standard library includes for std::map and std::vector containers.
*/

#pragma once

#include <torch/all.h>

#include <map>
#include <vector>

/* Explanation: Swaps cache blocks between source and destination tensors. 
* Used for managing memory blocks in the paged attention system. 
* `block_mapping` specifies which blocks to swap.
*/
void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping);

/* Explanation: Copies cache blocks for standard attention. 
* Takes separate vectors for key and value caches (one per layer). 
* The comment clarifies that while the vector itself is const 
* (can't add/remove elements), the tensor contents can be modified. 
* `block_mapping` is a 2D tensor `[num_pairs, 2]` where each row is `[src_block, dst_block]`
*/
// Note: the key_caches and value_caches are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping);

/* Explanation: Specialized version for MLA (Multi-head Latent Attention) 
* architecture where keys and values are stored in a single joint cache per layer,
* rather than separately.
*/ 
void copy_blocks_mla(std::vector<torch::Tensor> const& kv_caches,
                     const torch::Tensor& block_mapping);

/* Explanation: Stores newly computed keys and values into the cache:
* key/value: Input tensors from current computation
* key_cache/value_cache: Destination cache tensors
* slot_mapping: Maps each token to its cache slot [block_id, offset_in_block]
* kv_cache_dtype: String specifying cache data type (e.g., "fp8_e4m3" for quantized caches)
* k_scale/v_scale: Scaling factors for quantized caches
*/
void reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       torch::Tensor& k_scale, torch::Tensor& v_scale);

/* Explanation: FlashAttention-specific version of reshape_and_cache. 
* FlashAttention uses different memory layouts optimized for its algorithm.
* Note the parameter name k_cache(vs key_cache)-likely just naming inconsistency.
*/
void reshape_and_cache_flash(torch::Tensor& key, torch::Tensor& value,
                             torch::Tensor& key_cache, 
                             torch::Tensor& value_cache, 
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             torch::Tensor& k_scale, torch::Tensor& v_scale);

/* Explanation: MLA-specific caching that combines:
* kv_c: Compressed KV representation (latent representation)
* k_pe: Key positional encoding
* These are concatenated and stored in the joint kv_cache. 
* MLA uses a single scale tensor instead of separate k/v scales.
*/
void concat_and_cache_mla(torch::Tensor& kv_c, torch::Tensor& k_pe,
                          torch::Tensor& kv_cache, torch::Tensor& slot_mapping,
                          const std::string& kv_cache_dtype,
                          torch::Tensor& scale);

// Just for unittest
/* Explanation: Test utility function to convert cache from one format to FP8 
* (8-bit floating point). Takes a scaling factor and dtype string.
*  Not used in production code.
*/
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype);

/* Explanation: Retrieves cached values for attention computation:
* src_cache: Paged cache in block format
* dst: Output tensor with gathered cache values
* block_table: Maps each sequence to its cache blocks
* cu_seq_lens: Cumulative sequence lengths (prefix sum) for batch processing
* "maybe_dequant": If cache is quantized (FP8), dequantizes it; otherwise just gathers
* scale: Dequantization scale factors
* seq_starts: Optional tensor for sequence start positions
*/
void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache,     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,           // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,   // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,   // [BATCH+1]
    int64_t batch_size, const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts = std::nullopt);

// TODO(hc): cp_gather_cache need support scaled kvcache in the future.
/* Explanation: "CP" likely means "context parallelism" or "chunked prefill". 
* Similar to gather_and_maybe_dequant_cache but doesn't support quantized caches (no scale parameter). 
*/
void cp_gather_cache(
    torch::Tensor const& src_cache,   // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,         // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table, // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens, // [BATCH+1]
    int64_t batch_size, std::optional<torch::Tensor> seq_starts = std::nullopt);

// Indexer K quantization and cache function
/* Explanation: Specialized quantization for "Indexer" architecture (likely referring to a specific model):
* Quantizes keys (k) before caching
* quant_block_size: Granularity of quantization (e.g., quantize every 64 elements together)
* scale_fmt: Format string for how scales are stored
* cache_stride: Total stride in cache (includes quantized data + metadata)
*/
void indexer_k_quant_and_cache(
    torch::Tensor& k,             // [num_tokens, head_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,           // quantization block size
    const std::string& scale_fmt); 

// Extract function to gather quantized K cache
/* Explanation: Retrieves and dequantizes Indexer-quantized keys:
* Outputs both dst_k (dequantized keys) and dst_scale (the scaling factors used)
*
* Scale shape formula: head_dim / quant_block_size * 4 suggests 
* storing multiple scale components (possibly per-block min/max/scale values)
*
* "CP" prefix again suggests chunked prefill/context parallelism variant
*/
void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,               // [num_tokens, head_dim]
    torch::Tensor& dst_scale,           // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table,   // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens);  // [batch_size + 1]