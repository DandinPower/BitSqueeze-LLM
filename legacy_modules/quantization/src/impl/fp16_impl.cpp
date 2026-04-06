#include <quantization/impl/fp16_impl.hpp>

static int64_t _get_fp16_array_size(const fp16_array_t *fp16_array) {
    if (!fp16_array) return 0;
    return (int64_t)(sizeof(fp16_array_t) + fp16_array->num_elements * sizeof(uint16_t));
}

fp16_array_t *allocate_fp16_array(unsigned long long num_elements) {
    if (!num_elements) return NULL;

    size_t total = sizeof(fp16_array_t) + num_elements * sizeof(uint16_t);
    fp16_array_t *arr = (fp16_array_t *)calloc(1, total);
    if (!arr) return NULL;

    arr->num_elements = num_elements;
    arr->data = (uint16_t *)(arr + 1);
    return arr;
}

void free_fp16_array(fp16_array_t *fp16_array) {
    if (!fp16_array) return;
    free(fp16_array);
}

int64_t get_fp16_array_size(const fp16_array_t *fp16_array) {
    return _get_fp16_array_size(fp16_array);
}

fp16_array_t *load_fp16_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(fp16_array_t)) return NULL;

    fp16_array_t *arr = (fp16_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;

    memcpy(arr, buffer, buffer_size);
    const int64_t expected = _get_fp16_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }

    arr->data = (uint16_t *)(arr + 1);
    return arr;
}

int fp16_compress(const float *float_array, unsigned long long num_elements, fp16_array_t **fp16_array) {
    if (!float_array || num_elements == 0 || !fp16_array || *fp16_array) return 1;

    fp16_array_t *arr = allocate_fp16_array(num_elements);
    if (!arr) return 1;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long i = 0; i < num_elements; ++i) {
        arr->data[i] = fp16_ieee_from_fp32_value(float_array[i]);
    }

    *fp16_array = arr;
    return 0;
}

int fp16_decompress(const fp16_array_t *fp16_array, float *float_array) {
    if (!fp16_array || !float_array) return 1;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long i = 0; i < fp16_array->num_elements; ++i) {
        float_array[i] = fp16_ieee_to_fp32_value(fp16_array->data[i]);
    }
    return 0;
}
