#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_NF4_BLOCK_SIZE 64

typedef struct {
    unsigned long long num_elements;
    unsigned long long num_blocks;
    unsigned long long block_size;
    float *block_scales;
    uint8_t *data;
} nf4_array_t;

nf4_array_t *allocate_nf4_array(unsigned long long num_elements, unsigned long long block_size);
void free_nf4_array(nf4_array_t *nf4_array);
int64_t get_nf4_array_size(const nf4_array_t *nf4_array);
nf4_array_t *load_nf4_array_from_buffer(const void *buffer, int64_t buffer_size);
int nf4_compress(const float *float_array, unsigned long long num_elements, nf4_array_t **nf4_array);
int nf4_decompress(const nf4_array_t *nf4_array, float *float_array);
