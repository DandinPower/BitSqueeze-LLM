#include <quantization/impl/q8_0_impl.hpp>

static int64_t _get_q8_0_array_size(const q8_0_array_t *q8_0_array) {
    if (!q8_0_array) return 0;
    return sizeof(q8_0_array_t) + q8_0_array->num_blocks * sizeof(float) + q8_0_array->num_elements * sizeof(int8_t);
}

q8_0_array_t *allocate_q8_0_array(unsigned long long num_elements, unsigned long long block_size) {
    if (!num_elements || !block_size) return NULL;

    unsigned long long num_blocks = (num_elements + block_size - 1) / block_size;
    size_t total = sizeof(q8_0_array_t) + num_blocks * sizeof(float) + num_elements * sizeof(int8_t);

    q8_0_array_t *qa = (q8_0_array_t *)calloc(1, total);
    if (!qa) return NULL;

    qa->num_elements = num_elements;
    qa->num_blocks = num_blocks;
    qa->block_size = block_size;
    qa->scales = (float *)(qa + 1);
    qa->data = (int8_t *)(qa->scales + num_blocks);
    return qa;
}

void free_q8_0_array(q8_0_array_t *q8_0_array) {
    if (!q8_0_array) return;
    free(q8_0_array);
}

int64_t get_q8_0_array(const q8_0_array_t *q8_0_array) {
    return _get_q8_0_array_size(q8_0_array);
}

q8_0_array_t *load_quantized_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(q8_0_array_t)) return NULL;

    q8_0_array_t *q8_0_array = (q8_0_array_t *)calloc(1, buffer_size);
    if (!q8_0_array) return NULL;

    memcpy(q8_0_array, buffer, buffer_size);
    const int64_t expected = _get_q8_0_array_size(q8_0_array);
    if (buffer_size < expected) {
        free(q8_0_array);
        return NULL;
    }

    q8_0_array->scales = (float *)(q8_0_array + 1);
    q8_0_array->data = (int8_t *)(q8_0_array->scales + q8_0_array->num_blocks);
    return q8_0_array;
}

static int _quantize_q8_0(const float *float_array, q8_0_array_t *q8_0_array) {
    if (!float_array || !q8_0_array) return 1;

    const unsigned long long block_size = q8_0_array->block_size;
    const unsigned long long num_blocks = q8_0_array->num_blocks;
    const unsigned long long num_elements = q8_0_array->num_elements;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements) ? block_size : (num_elements - start);

        float abs_max = 0.0f;
        for (unsigned long long i = 0; i < remain; ++i) {
            float v = fabsf(float_array[start + i]);
            if (v > abs_max) abs_max = v;
        }

        float scale = (abs_max > 0.0f) ? (abs_max / 127.0f) : 0.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        q8_0_array->scales[b] = scale;

        for (unsigned long long i = 0; i < remain; ++i) {
            float val = float_array[start + i] * inv_scale;
            long qi = lrintf(val);
            if (qi < -127) qi = -127;
            if (qi > 127) qi = 127;
            q8_0_array->data[start + i] = (int8_t)qi;
        }
    }
    return 0;
}

int q8_0_compress(const float *float_array, unsigned long long num_elements, q8_0_array_t **q8_0_array) {
    if (!float_array || num_elements == 0 || !q8_0_array || *q8_0_array) return 1;

    *q8_0_array = allocate_q8_0_array(num_elements, DEFAULT_Q8_0_BLOCK_SIZE);
    if (!*q8_0_array) return 1;

    return _quantize_q8_0(float_array, *q8_0_array);
}

int q8_0_decompress(const q8_0_array_t *q8_0_array, float *float_array) {
    if (!q8_0_array || !float_array) return 1;

    const unsigned long long block_size = q8_0_array->block_size;
    const unsigned long long num_blocks = q8_0_array->num_blocks;
    const unsigned long long num_elements = q8_0_array->num_elements;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements) ? block_size : (num_elements - start);
        const float scale = q8_0_array->scales[b];

        for (unsigned long long i = 0; i < remain; ++i) {
            float_array[start + i] = scale * (float)q8_0_array->data[start + i];
        }
    }
    return 0;
}
