#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <quantization/datatype/fp16/fp16.hpp>

#define Q2_K_BLOCK_SIZE 16
#define Q2_K_SUPER_BLOCK_SIZE 16
#define WEIGHT_PER_SUPER_BLOCK (Q2_K_BLOCK_SIZE * Q2_K_SUPER_BLOCK_SIZE)

typedef struct {
    uint16_t super_scale;
    uint16_t super_min;
    uint8_t scales[Q2_K_SUPER_BLOCK_SIZE];
    uint8_t data[WEIGHT_PER_SUPER_BLOCK / 4];
} super_block_q2_k;

typedef struct {
    unsigned long long num_elements;
    unsigned long long num_elements_aligned;
    uint32_t num_super_blocks;
    super_block_q2_k *super_blocks;
} q2_k_array_t;

q2_k_array_t *allocate_q2_k_array(unsigned long long num_elements);
void free_q2_k_array(q2_k_array_t *q2_k_array);
int64_t get_q2_k_array_size(const q2_k_array_t *q2_k_array);
q2_k_array_t *load_q2_k_array_from_buffer(const void *buffer, int64_t buffer_size);
int q2_k_compress(const float *float_array, unsigned long long num_elements, q2_k_array_t **q2_k_array);
int q2_k_im_compress(const float *float_array, const float *importance_array, unsigned long long num_elements, q2_k_array_t **q2_k_array);
int q2_k_decompress(const q2_k_array_t *q2_k_array, float *float_array);
