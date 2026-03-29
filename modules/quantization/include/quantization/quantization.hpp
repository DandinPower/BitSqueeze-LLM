#pragma once

#include <stdint.h>
#include <quantization/impl/nf4_dq_impl.hpp>

typedef enum {
    quantization_INVALID = -1,
    Q8_0 = 0,
    Q4_0 = 1,
    Q2_K = 2,
    BF16 = 3,
    FP16 = 4,
    FP8  = 5,
    FP4  = 6,
    MXFP8 = 7,
    MXFP4 = 8,
    NVFP4 = 9,
    NF4_DQ = 10,
    NF4 = 11,
    Q2_K_FAST = 12,
} quantization_method_t;

typedef struct {
    unsigned long long num_elements;
} quantization_shape_t;

typedef struct quantization_buffer {
    quantization_method_t method;
    quantization_shape_t shape;
    void        *payload;
} quantization_buffer_t;

int quantization_compress(const float *src,
                    unsigned long long num_elements,
                    quantization_method_t method,
                    quantization_buffer_t **out);


int quantization_decompress(const quantization_buffer_t *buf,
                   float *dst,
                   unsigned long long dst_num_elements);

long long quantization_get_packed_size(const quantization_buffer_t *buf);

quantization_buffer_t *load_quantization_from_buffer(const void *buffer, long long buffer_size);

void quantization_free(quantization_buffer_t *buf);
