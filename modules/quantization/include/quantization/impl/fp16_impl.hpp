#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <quantization/datatype/fp16/fp16.hpp>

typedef struct {
    unsigned long long num_elements;
    uint16_t *data;
} fp16_array_t;

fp16_array_t *allocate_fp16_array(unsigned long long num_elements);
void free_fp16_array(fp16_array_t *fp16_array);
int64_t get_fp16_array_size(const fp16_array_t *fp16_array);
fp16_array_t *load_fp16_array_from_buffer(const void *buffer, int64_t buffer_size);
int fp16_compress(const float *float_array, unsigned long long num_elements, fp16_array_t **fp16_array);
int fp16_decompress(const fp16_array_t *fp16_array, float *float_array);
