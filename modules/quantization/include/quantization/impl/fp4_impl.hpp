#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FP4_MAX_NORM_VALUE 6.0f

typedef struct {
    unsigned long long num_elements;
    float scale;
    uint8_t *data;
} fp4_array_t;

fp4_array_t *allocate_fp4_array(unsigned long long num_elements);
void free_fp4_array(fp4_array_t *fp4_array);
int64_t get_fp4_array_size(const fp4_array_t *fp4_array);
fp4_array_t *load_fp4_array_from_buffer(const void *buffer, int64_t buffer_size);
int fp4_compress(const float *float_array, unsigned long long num_elements, fp4_array_t **fp4_array);
int fp4_decompress(const fp4_array_t *fp4_array, float *float_array);
