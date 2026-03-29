#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_MXFP4_BLOCK_SIZE 32
#define MXFP4_MAX_NORM_VALUE 6.0f

typedef struct {
    unsigned long long num_elements;
    unsigned long long num_blocks;
    unsigned long long block_size;
    int8_t *scales;
    uint8_t *data;
} mxfp4_array_t;

mxfp4_array_t *allocate_mxfp4_array(unsigned long long num_elements, unsigned long long block_size);
void free_mxfp4_array(mxfp4_array_t *mxfp4_array);
int64_t get_mxfp4_array_size(const mxfp4_array_t *mxfp4_array);
mxfp4_array_t *load_mxfp4_array_from_buffer(const void *buffer, int64_t buffer_size);
int mxfp4_compress(const float *float_array, unsigned long long num_elements, mxfp4_array_t **mxfp4_array);
int mxfp4_decompress(const mxfp4_array_t *mxfp4_array, float *float_array);
