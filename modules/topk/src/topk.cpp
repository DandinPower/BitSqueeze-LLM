#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <topk.hpp>

typedef struct {
    float abs_val;
    float val;
    uint16_t idx;
} heap_entry_t;

static inline float _importance_abs(float v) {
    float a = fabsf(v);
    return (a == a) ? a : -INFINITY;
}

static inline int _is_worse(heap_entry_t a, heap_entry_t b) {
    return a.abs_val < b.abs_val;
}

static inline void _sift_down_min(heap_entry_t *heap, uint16_t k, uint16_t p) {
    while (1) {
        uint32_t left = (uint32_t)2 * p + 1;
        if (left >= k) break;

        uint32_t right = left + 1;
        uint16_t c = (uint16_t)left;

        if (right < k && _is_worse(heap[(uint16_t)right], heap[c])) {
            c = (uint16_t)right;
        }

        if (_is_worse(heap[c], heap[p])) {
            heap_entry_t tmp = heap[p];
            heap[p] = heap[c];
            heap[c] = tmp;
            p = c;
        } else {
            break;
        }
    }
}

static inline void _heapify_min(heap_entry_t *heap, uint16_t k) {
    if (k <= 1) return;
    for (int32_t p = (int32_t)(k / 2) - 1; p >= 0; --p) {
        _sift_down_min(heap, k, (uint16_t)p);
    }
}

topk_array_t *allocate_topk_array(uint16_t num_rows, uint16_t num_columns, float topk_ratio) {
    if (num_rows == 0 || num_columns == 0) return NULL;
    if (topk_ratio < 0.0f || topk_ratio > 1.0f) return NULL;

    float raw_topk = (float)num_columns * topk_ratio;
    uint16_t num_topk_columns = (uint16_t)roundf(raw_topk);

    if (num_topk_columns > num_columns) {
        num_topk_columns = num_columns;
    } else if (num_topk_columns == 0 && topk_ratio > 0.0f) {
        num_topk_columns = 1;
    }

    uint32_t topk_elements = (uint32_t)num_rows * num_topk_columns;
    uint64_t total = sizeof(topk_array_t) +
                     (uint64_t)topk_elements * (sizeof(uint16_t) + sizeof(float));

    topk_array_t *topk_array = (topk_array_t *)calloc(1, (size_t)total);
    if (!topk_array) return NULL;

    topk_array->num_rows = num_rows;
    topk_array->num_columns = num_columns;
    topk_array->num_topk_columns = num_topk_columns;
    topk_array->topk_indices = (uint16_t *)(topk_array + 1);
    topk_array->values = (float *)(topk_array->topk_indices + topk_elements);

    return topk_array;
}

void free_topk_array(topk_array_t *topk_array) {
    if (!topk_array) return;
    free(topk_array);
}

uint64_t get_topk_array_size(const topk_array_t *topk_array) {
    if (!topk_array) return 0;

    uint32_t topk_elements = (uint32_t)topk_array->num_rows * topk_array->num_topk_columns;
    return sizeof(topk_array_t) +
           (uint64_t)topk_elements * (sizeof(uint16_t) + sizeof(float));
}

topk_array_t *load_topk_array_from_buffer(const void *buffer, uint64_t buffer_size) {
    if (!buffer || buffer_size < sizeof(topk_array_t)) return NULL;

    topk_array_t *topk_array = (topk_array_t *)calloc(1, (size_t)buffer_size);
    if (!topk_array) return NULL;

    memcpy(topk_array, buffer, (size_t)buffer_size);

    uint64_t expected_size = get_topk_array_size(topk_array);
    if (expected_size == 0 || buffer_size < expected_size) {
        free(topk_array);
        return NULL;
    }

    uint32_t topk_elements = (uint32_t)topk_array->num_rows * topk_array->num_topk_columns;
    topk_array->topk_indices = (uint16_t *)(topk_array + 1);
    topk_array->values = (float *)(topk_array->topk_indices + topk_elements);

    return topk_array;
}

int topk_extraction(const float *float_array,
                    uint16_t num_rows,
                    uint16_t num_columns,
                    float topk_ratio,
                    topk_array_t **topk_array) {
    if (!float_array || !topk_array || *topk_array) return 1;
    if (num_rows == 0 || num_columns == 0) return 1;

    *topk_array = allocate_topk_array(num_rows, num_columns, topk_ratio);
    if (!*topk_array) return 1;

    topk_array_t *out = *topk_array;
    const uint16_t k = out->num_topk_columns;
    const uint16_t num_features = num_columns;
    if (k == 0) return 0;

    int alloc_error = 0;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel
    {
#endif
        heap_entry_t *heap = (heap_entry_t *)malloc((size_t)k * sizeof(heap_entry_t));
        if (!heap) {
#if defined(__linux__) && defined(_OPENMP)
#pragma omp critical
#endif
            { alloc_error = 1; }
        }

#if defined(__linux__) && defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int row = 0; row < (int)num_rows; ++row) {
            if (!heap) continue;

            const uint32_t dense_base = (uint32_t)row * (uint32_t)num_features;
            const uint32_t topk_base = (uint32_t)row * (uint32_t)k;
            const float *x = float_array + dense_base;

            for (uint16_t i = 0; i < k; ++i) {
                float v = x[i];
                heap[i].idx = i;
                heap[i].val = v;
                heap[i].abs_val = _importance_abs(v);
            }

            _heapify_min(heap, k);

            for (uint16_t i = k; i < num_features; ++i) {
                float v = x[i];
                float a = _importance_abs(v);
                if (a > heap[0].abs_val) {
                    heap[0].idx = i;
                    heap[0].val = v;
                    heap[0].abs_val = a;
                    _sift_down_min(heap, k, 0);
                }
            }

            for (uint16_t j = 0; j < k; ++j) {
                out->topk_indices[topk_base + j] = heap[j].idx;
                out->values[topk_base + j] = heap[j].val;
            }
        }

        free(heap);
#if defined(__linux__) && defined(_OPENMP)
    }
#endif

    if (alloc_error) {
        free_topk_array(*topk_array);
        *topk_array = NULL;
        return 1;
    }

    return 0;
}

int topk_separation(float *float_array,
                    uint16_t num_rows,
                    uint16_t num_columns,
                    float topk_ratio,
                    topk_array_t **topk_array) {
    if (!float_array) return 1;

    int rc = topk_extraction(float_array, num_rows, num_columns, topk_ratio, topk_array);
    if (rc != 0) return rc;

    const topk_array_t *out = *topk_array;
    const uint16_t k = out->num_topk_columns;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint16_t row = 0; row < out->num_rows; ++row) {
        uint32_t dense_base = (uint32_t)row * out->num_columns;
        uint32_t topk_base = (uint32_t)row * k;
        for (uint16_t j = 0; j < k; ++j) {
            uint16_t idx = out->topk_indices[topk_base + j];
            float_array[dense_base + idx] = 0.0f;
        }
    }

    return 0;
}

int topk_decompress(const topk_array_t *topk_array, float *float_array) {
    if (!topk_array || !float_array) return 1;

    uint32_t num_elements = (uint32_t)topk_array->num_rows * topk_array->num_columns;
    memset(float_array, 0, (size_t)num_elements * sizeof(float));

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint16_t row = 0; row < topk_array->num_rows; ++row) {
        uint32_t dense_base = (uint32_t)row * topk_array->num_columns;
        uint32_t topk_base = (uint32_t)row * topk_array->num_topk_columns;

        for (uint16_t j = 0; j < topk_array->num_topk_columns; ++j) {
            uint16_t idx = topk_array->topk_indices[topk_base + j];
            float_array[dense_base + idx] = topk_array->values[topk_base + j];
        }
    }

    return 0;
}

int topk_apply(const topk_array_t *topk_array, float *float_array) {
    if (!topk_array || !float_array) return 1;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint16_t row = 0; row < topk_array->num_rows; ++row) {
        uint32_t dense_base = (uint32_t)row * topk_array->num_columns;
        uint32_t topk_base = (uint32_t)row * topk_array->num_topk_columns;

        for (uint16_t j = 0; j < topk_array->num_topk_columns; ++j) {
            uint16_t idx = topk_array->topk_indices[topk_base + j];
            float_array[dense_base + idx] += topk_array->values[topk_base + j];
        }
    }

    return 0;
}
