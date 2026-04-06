#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FP8_MAX_NORM_VALUE 448.0f

typedef struct {
    unsigned long long num_elements;
    float scale;
    uint8_t *data;
} fp8_array_t;

fp8_array_t *allocate_fp8_array(unsigned long long num_elements);
void free_fp8_array(fp8_array_t *fp8_array);
int64_t get_fp8_array_size(const fp8_array_t *fp8_array);
fp8_array_t *load_fp8_array_from_buffer(const void *buffer, int64_t buffer_size);
int fp8_compress(const float *float_array, unsigned long long num_elements, fp8_array_t **fp8_array);
int fp8_decompress(const fp8_array_t *fp8_array, float *float_array);
