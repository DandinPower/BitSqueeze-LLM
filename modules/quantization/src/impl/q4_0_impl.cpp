#include <quantization/impl/q4_0_impl.hpp>

static int64_t _get_q4_0_array_size(const q4_0_array_t *q4_0_array) {
    if (!q4_0_array) return 0;

    const unsigned long long num_elements_for_data = (q4_0_array->num_elements + 1) / 2;
    return sizeof(q4_0_array_t) + q4_0_array->num_blocks * sizeof(float) + num_elements_for_data * sizeof(int8_t);
}

q4_0_array_t *allocate_q4_0_array(unsigned long long num_elements, unsigned long long block_size) {
    if (!num_elements || !block_size) return NULL;

    unsigned long long num_blocks = (num_elements + block_size - 1) / block_size;
    unsigned long long num_elements_for_data = (num_elements + 1) / 2;

    size_t total = sizeof(q4_0_array_t) + num_blocks * sizeof(float) + num_elements_for_data * sizeof(int8_t);

    q4_0_array_t *qa = (q4_0_array_t *)calloc(1, total);
    if (!qa) return NULL;

    qa->num_elements = num_elements;
    qa->num_blocks = num_blocks;
    qa->block_size = block_size;
    qa->scales = (float *)(qa + 1);
    qa->data = (int8_t *)(qa->scales + num_blocks);
    return qa;
}

void free_q4_0_array(q4_0_array_t *q4_0_array) {
    if (!q4_0_array) return;
    free(q4_0_array);
}

int64_t get_q4_0_array_size(const q4_0_array_t *q4_0_array) {
    return _get_q4_0_array_size(q4_0_array);
}

q4_0_array_t *load_q4_0_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(q4_0_array_t)) return NULL;

    q4_0_array_t *q4_0_array = (q4_0_array_t *)calloc(1, buffer_size);
    if (!q4_0_array) return NULL;

    memcpy(q4_0_array, buffer, buffer_size);
    const int64_t expected = _get_q4_0_array_size(q4_0_array);
    if (buffer_size < expected) {
        free(q4_0_array);
        return NULL;
    }

    q4_0_array->scales = (float *)(q4_0_array + 1);
    q4_0_array->data = (int8_t *)(q4_0_array->scales + q4_0_array->num_blocks);
    return q4_0_array;
}

static int _quantize_q4_0(const float *float_array, q4_0_array_t *q4_0_array) {
    if (!float_array || !q4_0_array) return 1;

    const unsigned long long block_size = q4_0_array->block_size;
    const unsigned long long num_blocks = q4_0_array->num_blocks;
    const unsigned long long num_elements = q4_0_array->num_elements;
    uint8_t *data = (uint8_t *)q4_0_array->data;

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

        float scale = (abs_max > 0.0f) ? (abs_max / 7.0f) : 0.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        q4_0_array->scales[b] = scale;

        for (unsigned long long i = 0; i < remain; ++i) {
            float val = float_array[start + i] * inv_scale;
            long qi = lrintf(val);
            if (qi < -7) qi = -7;
            if (qi > 7) qi = 7;

            const uint8_t four_bit_qi = ((uint8_t)qi) & 0x0F;
            const unsigned long long data_index = (start + i) / 2;
            if (((start + i) % 2) == 0) {
                data[data_index] = (uint8_t)(four_bit_qi << 4);
            } else {
                data[data_index] = (uint8_t)(data[data_index] | four_bit_qi);
            }
        }
    }
    return 0;
}

int q4_0_compress(const float *float_array, unsigned long long num_elements, uint8_t quantized_type, q4_0_array_t **q4_0_array) {
    if (!float_array || num_elements == 0 || !q4_0_array || *q4_0_array) return 1;
    if (quantized_type != 0) return 1;

    *q4_0_array = allocate_q4_0_array(num_elements, DEFAULT_Q4_0_BLOCK_SIZE);
    if (!*q4_0_array) return 1;

    return _quantize_q4_0(float_array, *q4_0_array);
}

int q4_0_decompress(const q4_0_array_t *q4_0_array, float *float_array) {
    if (!q4_0_array || !float_array) return 1;

    const unsigned long long block_size = q4_0_array->block_size;
    const unsigned long long num_blocks = q4_0_array->num_blocks;
    const unsigned long long num_elements = q4_0_array->num_elements;
    const uint8_t *src_data = (const uint8_t *)q4_0_array->data;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements) ? block_size : (num_elements - start);
        const float scale = q4_0_array->scales[b];

        for (unsigned long long i = 0; i < remain; ++i) {
            const unsigned long long data_index = (start + i) / 2;
            const uint8_t packed_qi = src_data[data_index];

            uint8_t qi = ((start + i) % 2 == 0) ? (packed_qi >> 4) : (packed_qi & 0x0F);
            const int8_t signed_qi = (int8_t)(qi << 4) >> 4;
            float_array[start + i] = scale * (float)(signed_qi);
        }
    }
    return 0;
}
