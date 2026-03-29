#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_NVFP4_BLOCK_SIZE 16
#define NVFP4_MAX_NORM_VALUE 6.0f
#define NVFP4_FP8_MAX_NORM 448.0f

typedef struct {
    unsigned long long num_elements;
    unsigned long long num_blocks;
    unsigned long long block_size;
    float tensor_scale;
    uint8_t *block_scales;
    uint8_t *data;
} nvfp4_array_t;

nvfp4_array_t *allocate_nvfp4_array(unsigned long long num_elements, unsigned long long block_size);
void free_nvfp4_array(nvfp4_array_t *nvfp4_array);
int64_t get_nvfp4_array_size(const nvfp4_array_t *nvfp4_array);
nvfp4_array_t *load_nvfp4_array_from_buffer(const void *buffer, int64_t buffer_size);
int nvfp4_compress(const float *float_array, unsigned long long num_elements, nvfp4_array_t **nvfp4_array);
int nvfp4_decompress(const nvfp4_array_t *nvfp4_array, float *float_array);
