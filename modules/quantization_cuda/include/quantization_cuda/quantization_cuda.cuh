#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <quantization_cuda/quantization_method.hpp>

static constexpr std::uint64_t QUANTIZATION_CUDA_NF4_BLOCK_SIZE = 64ULL;
static constexpr float QUANTIZATION_CUDA_NF4_DQ_FP8_MAX_NORM_VALUE = 448.0f;
static constexpr std::uint32_t QUANTIZATION_CUDA_PACKED_MAGIC = 0x51544355u;
static constexpr std::uint16_t QUANTIZATION_CUDA_PACKED_VERSION = 1u;

typedef struct {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t reserved0;
    std::int32_t method;
    std::uint32_t reserved1;
    std::uint64_t num_elements;
    std::uint64_t num_blocks;
    std::uint64_t block_size;
    float dq_scale;
    std::uint32_t reserved2;
} quantization_cuda_packed_header_t;

typedef struct {
    quantization_method_t method = quantization_INVALID;
    std::uint64_t num_elements = 0;
    std::uint64_t num_blocks = 0;
    std::uint64_t block_size = QUANTIZATION_CUDA_NF4_BLOCK_SIZE;
    float *d_block_scales_fp32 = nullptr;
    std::uint8_t *d_block_scales_fp8 = nullptr;
    float *d_dq_scale = nullptr;
    std::uint8_t *d_data = nullptr;
} quantization_cuda_array_t;

struct QuantizationCUDADevicePtrs {
    float *d_block_absmax = nullptr;
    float *d_reduce_out = nullptr;
    void *d_reduce_temp_storage = nullptr;
    std::size_t reduce_temp_storage_bytes = 0;

    void free_all() noexcept;
};

struct QuantizationCUDAContext {
    bool initialized = false;
    std::uint64_t num_elements_capacity = 0;
    std::uint64_t num_blocks_capacity = 0;
    cudaStream_t stream = nullptr;
    QuantizationCUDADevicePtrs cuda_ptrs;

    void destroy_all() noexcept;
};

std::uint64_t quantization_cuda_compute_num_blocks(
    std::uint64_t num_elements,
    std::uint64_t block_size = QUANTIZATION_CUDA_NF4_BLOCK_SIZE);

void quantization_cuda_bind_nf4_array(
    quantization_cuda_array_t *quant_array,
    std::uint64_t num_elements,
    float *d_block_scales,
    std::uint8_t *d_data);

void quantization_cuda_bind_nf4_dq_array(
    quantization_cuda_array_t *quant_array,
    std::uint64_t num_elements,
    float *d_dq_scale,
    std::uint8_t *d_block_scales,
    std::uint8_t *d_data);

void quantization_cuda_release(QuantizationCUDAContext *ctx) noexcept;

void quantization_cuda_initialize(
    QuantizationCUDAContext *ctx,
    std::uint64_t num_elements_capacity,
    unsigned long long warmup_seed = 1234ULL);

void quantization_cuda_compress(
    QuantizationCUDAContext *ctx,
    const float *d_float_array,
    quantization_cuda_array_t *quant_array);

void quantization_cuda_decompress(
    QuantizationCUDAContext *ctx,
    const quantization_cuda_array_t *quant_array,
    float *d_float_array);

std::uint64_t quantization_cuda_get_packed_size_for_shape(
    quantization_method_t method,
    std::uint64_t num_elements,
    std::uint64_t block_size = QUANTIZATION_CUDA_NF4_BLOCK_SIZE);

std::uint64_t quantization_cuda_get_packed_size(const quantization_cuda_array_t *quant_array);

int quantization_cuda_validate_packed_buffer(
    const void *buffer,
    std::uint64_t buffer_size,
    quantization_cuda_packed_header_t *header_out) noexcept;

int quantization_cuda_pack_to_buffer(
    const quantization_cuda_array_t *quant_array,
    void *buffer,
    std::uint64_t buffer_size) noexcept;

int quantization_cuda_unpack_from_buffer(
    const void *buffer,
    std::uint64_t buffer_size,
    quantization_cuda_array_t *quant_array) noexcept;
