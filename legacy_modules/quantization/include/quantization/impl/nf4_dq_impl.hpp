#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_NF4_DQ_BLOCK_SIZE 64
#define NF4_DQ_FP8_MAX_NORM_VALUE 448.0f

/*
 * NF4_DQ (NormalFloat4 with double quantization) with block size 64:
 *  - data: 4-bit NF4_DQ codes packed 2 per byte
 *  - per-block scale (c2) quantized to FP8 E4M3 and stored in block_scales
 *  - a single FP32 scale (c1 = dq_scale) used to dequantize block_scales
 */
typedef struct {
    unsigned long long num_elements;   /* total elements in the original float array */
    unsigned long long num_blocks;     /* number of 64-value blocks */
    unsigned long long block_size;     /* elements per block (default = 64) */
    float    dq_scale;       /* FP32 scale to dequantize block_scales (c1) */
    uint8_t *block_scales;   /* FP8 E4M3 per-block scale codes, length = num_blocks */
    uint8_t *data;           /* packed NF4_DQ codes, length = ceil(num_elements / 2) bytes */
} nf4_dq_array_t;

nf4_dq_array_t *allocate_nf4_dq_array(unsigned long long num_elements,
                                      unsigned long long block_size);

void free_nf4_dq_array(nf4_dq_array_t *nf4_dq_array);

long long get_nf4_dq_array_size(const nf4_dq_array_t *nf4_dq_array);

nf4_dq_array_t *load_nf4_dq_array_from_buffer(const void *buffer, long long buffer_size);

int nf4_dq_compress(const float *float_array,
                    unsigned long long num_elements,
                    nf4_dq_array_t **nf4_dq_array);

int nf4_dq_decompress(const nf4_dq_array_t *nf4_dq_array,
                      float *float_array);
