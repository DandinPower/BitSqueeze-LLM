#include <quantization/impl/bf16_impl.hpp>

static int64_t _get_bf16_array_size(const bf16_array_t *bf16_array) {
    if (!bf16_array) return 0;
    return (int64_t)(sizeof(bf16_array_t) + bf16_array->num_elements * sizeof(uint16_t));
}

bf16_array_t *allocate_bf16_array(unsigned long long num_elements) {
    if (!num_elements) return NULL;

    size_t total = sizeof(bf16_array_t) + num_elements * sizeof(uint16_t);
    bf16_array_t *arr = (bf16_array_t *)calloc(1, total);
    if (!arr) return NULL;

    arr->num_elements = num_elements;
    arr->data = (uint16_t *)(arr + 1);
    return arr;
}

void free_bf16_array(bf16_array_t *bf16_array) {
    if (!bf16_array) return;
    free(bf16_array);
}

int64_t get_bf16_array_size(const bf16_array_t *bf16_array) {
    return _get_bf16_array_size(bf16_array);
}

bf16_array_t *load_bf16_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(bf16_array_t)) return NULL;

    bf16_array_t *arr = (bf16_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;

    memcpy(arr, buffer, buffer_size);
    const int64_t expected = _get_bf16_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }

    arr->data = (uint16_t *)(arr + 1);
    return arr;
}

int bf16_compress(const float *float_array, unsigned long long num_elements, bf16_array_t **bf16_array) {
    if (!float_array || num_elements == 0 || !bf16_array || *bf16_array) return 1;

    bf16_array_t *arr = allocate_bf16_array(num_elements);
    if (!arr) return 1;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long i = 0; i < num_elements; ++i) {
        arr->data[i] = bf16_from_fp32_value(float_array[i]);
    }

    *bf16_array = arr;
    return 0;
}

int bf16_decompress(const bf16_array_t *bf16_array, float *float_array) {
    if (!bf16_array || !float_array) return 1;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long i = 0; i < bf16_array->num_elements; ++i) {
        float_array[i] = fp32_from_bf16_value(bf16_array->data[i]);
    }
    return 0;
}
