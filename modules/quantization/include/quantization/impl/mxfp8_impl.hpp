#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_MXFP8_BLOCK_SIZE 32
#define MXFP8_MAX_NORM_VALUE 448.0f

typedef struct {
    unsigned long long num_elements;
    unsigned long long num_blocks;
    unsigned long long block_size;
    int8_t *scales;
    uint8_t *data;
} mxfp8_array_t;

mxfp8_array_t *allocate_mxfp8_array(unsigned long long num_elements, unsigned long long block_size);
void free_mxfp8_array(mxfp8_array_t *mxfp8_array);
int64_t get_mxfp8_array_size(const mxfp8_array_t *mxfp8_array);
mxfp8_array_t *load_mxfp8_array_from_buffer(const void *buffer, int64_t buffer_size);
int mxfp8_compress(const float *float_array, unsigned long long num_elements, mxfp8_array_t **mxfp8_array);
int mxfp8_decompress(const mxfp8_array_t *mxfp8_array, float *float_array);
