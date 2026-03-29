#pragma once

#include <stdint.h>

typedef struct {
    uint16_t num_rows;
    uint16_t num_columns;
    uint16_t num_topk_columns;
    uint16_t *topk_indices;
    float *values;
} topk_array_t;

topk_array_t *allocate_topk_array(uint16_t num_rows, uint16_t num_columns, float topk_ratio);

void free_topk_array(topk_array_t *topk_array);

uint64_t get_topk_array_size(const topk_array_t *topk_array);

topk_array_t *load_topk_array_from_buffer(const void *buffer, uint64_t buffer_size);

int topk_extraction(const float *float_array,
                    uint16_t num_rows,
                    uint16_t num_columns,
                    float topk_ratio,
                    topk_array_t **topk_array);

int topk_separation(float *float_array,
                    uint16_t num_rows,
                    uint16_t num_columns,
                    float topk_ratio,
                    topk_array_t **topk_array);

int topk_decompress(const topk_array_t *topk_array, float *float_array);

int topk_apply(const topk_array_t *topk_array, float *float_array);
