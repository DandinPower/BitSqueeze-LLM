#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEFAULT_Q8_0_BLOCK_SIZE 32

typedef struct {
    unsigned long long num_elements;
    unsigned long long num_blocks;
    unsigned long long block_size;
    float *scales;
    int8_t *data;
} q8_0_array_t;

q8_0_array_t *allocate_q8_0_array(unsigned long long num_elements, unsigned long long block_size);
void free_q8_0_array(q8_0_array_t *q8_0_array);
int64_t get_q8_0_array(const q8_0_array_t *q8_0_array);
q8_0_array_t *load_quantized_array_from_buffer(const void *buffer, int64_t buffer_size);
int q8_0_compress(const float *float_array, unsigned long long num_elements, q8_0_array_t **q8_0_array);
int q8_0_decompress(const q8_0_array_t *q8_0_array, float *float_array);
