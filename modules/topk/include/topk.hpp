#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

typedef struct {
    uint16_t num_rows;
    uint16_t num_columns;
    uint16_t num_topk_columns;
    uint16_t *d_topk_indices;
    float *d_values;
} topk_array_t;

typedef struct {
    uint16_t num_rows;
    uint16_t num_columns;
    uint16_t num_topk_columns;
    uint16_t reserved;
} topk_packed_header_t;

struct TopkCUDADevicePtrs {
    float *d_sort_keys_in = nullptr;
    float *d_sort_keys_out = nullptr;
    uint16_t *d_sort_indices_in = nullptr;
    uint16_t *d_sort_indices_out = nullptr;
    int *d_segment_offsets_begin = nullptr;
    int *d_segment_offsets_end = nullptr;
    void *d_cub_temp_storage = nullptr;
    size_t cub_temp_storage_bytes = 0;

    void free_all() noexcept;
};

struct TopkCUDAContext {
    uint16_t num_rows = 0;
    uint16_t num_columns = 0;
    bool initialized = false;

    cudaStream_t stream = nullptr;
    TopkCUDADevicePtrs cuda_ptrs;

    void destroy_all() noexcept;
};

uint16_t topk_compute_num_topk_columns(uint16_t num_columns, float topk_ratio);

uint32_t topk_compute_num_topk_elements(uint16_t num_rows, uint16_t num_topk_columns);

void topk_bind_array(
    topk_array_t *topk_array,
    uint16_t num_rows,
    uint16_t num_columns,
    uint16_t num_topk_columns,
    uint16_t *d_topk_indices,
    float *d_values);

void topk_release(TopkCUDAContext *ctx) noexcept;

void topk_initialize(
    TopkCUDAContext *ctx,
    uint16_t num_rows,
    uint16_t num_columns,
    unsigned long long warmup_seed = 1234ULL);

void topk_extraction(
    TopkCUDAContext *ctx,
    const float *d_float_array,
    topk_array_t *topk_array);

void topk_separation(
    TopkCUDAContext *ctx,
    float *d_float_array,
    topk_array_t *topk_array);

void topk_decompress(
    TopkCUDAContext *ctx,
    const topk_array_t *topk_array,
    float *d_float_array);

void topk_apply(
    TopkCUDAContext *ctx,
    const topk_array_t *topk_array,
    float *d_float_array);

uint64_t topk_get_packed_size_for_shape(
    uint16_t num_rows,
    uint16_t num_columns,
    uint16_t num_topk_columns);

uint64_t topk_get_packed_size(const topk_array_t *topk_array);

int topk_validate_packed_buffer(
    const void *buffer,
    uint64_t buffer_size,
    topk_packed_header_t *header_out) noexcept;

int topk_pack_to_buffer(
    const topk_array_t *topk_array,
    void *buffer,
    uint64_t buffer_size) noexcept;

int topk_unpack_from_buffer(
    const void *buffer,
    uint64_t buffer_size,
    topk_array_t *topk_array) noexcept;
