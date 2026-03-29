#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEFAULT_Q4_0_BLOCK_SIZE 32
#define DEFAULT_Q4_K_SUPER_BLOCK_SIZE 8

typedef struct {
    unsigned long long num_elements;
    unsigned long long num_blocks;
    unsigned long long block_size;
    float *scales;
    int8_t *data;
} q4_0_array_t;

q4_0_array_t *allocate_q4_0_array(unsigned long long num_elements, unsigned long long block_size);
void free_q4_0_array(q4_0_array_t *q4_0_array);
int64_t get_q4_0_array_size(const q4_0_array_t *q4_0_array);
q4_0_array_t *load_q4_0_array_from_buffer(const void *buffer, int64_t buffer_size);
int q4_0_compress(const float *float_array, unsigned long long num_elements, uint8_t quantized_type, q4_0_array_t **q4_0_array);
int q4_0_decompress(const q4_0_array_t *q4_0_array, float *float_array);
