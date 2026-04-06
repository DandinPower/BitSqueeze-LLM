#pragma once

#include <stdint.h>

#if defined(_WIN32)
#if defined(BITSQZ_LLM_EXPORTS)
#define BITSQZ_LLM_API __declspec(dllexport)
#else
#define BITSQZ_LLM_API __declspec(dllimport)
#endif
#else
#define BITSQZ_LLM_API
#endif

#ifndef BITSQZ_LLM_QUANTIZATION_METHOD_T_DEFINED
#define BITSQZ_LLM_QUANTIZATION_METHOD_T_DEFINED
typedef enum {
    quantization_INVALID = -1,
    Q8_0 = 0,
    Q4_0 = 1,
    Q2_K = 2,
    BF16 = 3,
    FP16 = 4,
    FP8 = 5,
    FP4 = 6,
    MXFP8 = 7,
    MXFP4 = 8,
    NVFP4 = 9,
    NF4_DQ = 10,
    NF4 = 11,
    Q2_K_FAST = 12,
} quantization_method_t;
#endif

typedef enum {
    BITSQZ_SECTION_NONE = 0,
    BITSQZ_SECTION_FP32 = 1,
    BITSQZ_SECTION_QUANT = 2,
    BITSQZ_SECTION_TOPK = 3,
} bitsqz_section_storage_t;

typedef struct {
    uint64_t offset;
    uint64_t size;
    uint8_t storage;
    uint8_t reserved[7];
} bitsqz_section_t;

typedef struct bitsqz_llm_array {
    uint16_t num_rows;
    uint16_t num_columns;
    uint32_t num_elements;

    float outlier_topk_ratio;
    float error_correction_topk_ratio;

    int16_t svd_ranks;
    int16_t svd_niters;
    int16_t effective_rank;
    int16_t reserved0;

    quantization_method_t svd_uv_format;
    quantization_method_t svd_s_format;
    quantization_method_t quantization_only_format;
    int32_t reserved1;

    uint32_t flags;

    bitsqz_section_t section_u;
    bitsqz_section_t section_s;
    bitsqz_section_t section_v;
    bitsqz_section_t section_direct;
    bitsqz_section_t section_outlier_topk;
    bitsqz_section_t section_error_topk;

    void *payload;
} bitsqz_llm_array_t;

typedef struct {
    double topk_separation_latency_ms;
    double svd_lowrank_cuda_latency_ms;
    double quantization_compress_latency_ms;
    double reconsturct_quantization_decompress_latency_ms;
    double reconstruct_svd_latency_ms;
    double error_extraction_latency_ms;
    double other_latency_ms;
} bitsqz_llm_compress_profile_t;

/* Quantized formats accepted by bitsqz_llm are quantization_INVALID, NF4, and NF4_DQ. */
BITSQZ_LLM_API int bitsqz_llm_initialize(
    uint16_t num_rows,
    uint16_t num_columns,
    float outlier_topk_ratio,
    float error_correction_topk_ratio,
    int svd_ranks,
    int svd_niters,
    quantization_method_t svd_uv_format,
    quantization_method_t svd_s_format,
    quantization_method_t quantization_only_format);

BITSQZ_LLM_API void bitsqz_llm_release();

BITSQZ_LLM_API int bitsqz_llm_compress(
    const float *d_row_major_matrix_float_data,
    bitsqz_llm_array_t **out,
    bitsqz_llm_compress_profile_t *profile);

inline int bitsqz_llm_compress(
    const float *d_row_major_matrix_float_data,
    bitsqz_llm_array_t **out) {
    return bitsqz_llm_compress(d_row_major_matrix_float_data, out, nullptr);
}

BITSQZ_LLM_API int bitsqz_llm_decompress(
    const bitsqz_llm_array_t *compressed,
    float *d_dst,
    uint32_t dst_num_elements);

BITSQZ_LLM_API uint64_t bitsqz_llm_get_packed_size(const bitsqz_llm_array_t *compressed);

BITSQZ_LLM_API bitsqz_llm_array_t *load_bitsqz_llm_from_buffer(const void *buffer, uint64_t buffer_size);

BITSQZ_LLM_API void bitsqz_llm_free(bitsqz_llm_array_t *compressed);
