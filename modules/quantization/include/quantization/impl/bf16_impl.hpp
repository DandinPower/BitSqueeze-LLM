#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <quantization/datatype/bf16.hpp>

typedef struct {
    unsigned long long num_elements;
    uint16_t *data;
} bf16_array_t;

bf16_array_t *allocate_bf16_array(unsigned long long num_elements);
void free_bf16_array(bf16_array_t *bf16_array);
int64_t get_bf16_array_size(const bf16_array_t *bf16_array);
bf16_array_t *load_bf16_array_from_buffer(const void *buffer, int64_t buffer_size);
int bf16_compress(const float *float_array, unsigned long long num_elements, bf16_array_t **bf16_array);
int bf16_decompress(const bf16_array_t *bf16_array, float *float_array);
