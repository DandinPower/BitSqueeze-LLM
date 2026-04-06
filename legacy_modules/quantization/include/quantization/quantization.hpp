#pragma once

#include <stdint.h>
#include <quantization_cuda/quantization_method.hpp>
#include <quantization/impl/q8_0_impl.hpp>
#include <quantization/impl/q4_0_impl.hpp>
#include <quantization/impl/q2_k_impl.hpp>
#include <quantization/impl/q2_k_fast_impl.hpp>
#include <quantization/impl/bf16_impl.hpp>
#include <quantization/impl/fp16_impl.hpp>
#include <quantization/impl/fp8_impl.hpp>
#include <quantization/impl/fp4_impl.hpp>
#include <quantization/impl/mxfp8_impl.hpp>
#include <quantization/impl/mxfp4_impl.hpp>
#include <quantization/impl/nvfp4_impl.hpp>
#include <quantization/impl/nf4_impl.hpp>
#include <quantization/impl/nf4_dq_impl.hpp>

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
