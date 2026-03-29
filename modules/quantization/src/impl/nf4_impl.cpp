#include <quantization/impl/nf4_impl.hpp>

static const float NF4_LEVELS[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

static int64_t _get_nf4_array_size(const nf4_array_t *nf4_array) {
    if (!nf4_array) return 0;
    const unsigned long long packed_elems = (nf4_array->num_elements + 1) / 2;
    return sizeof(nf4_array_t) + nf4_array->num_blocks * sizeof(float) + packed_elems * sizeof(uint8_t);
}

nf4_array_t *allocate_nf4_array(unsigned long long num_elements, unsigned long long block_size) {
    if (!num_elements || !block_size) return NULL;

    unsigned long long num_blocks = (num_elements + block_size - 1) / block_size;
    unsigned long long packed_elems = (num_elements + 1) / 2;
    size_t total = sizeof(nf4_array_t) + num_blocks * sizeof(float) + packed_elems * sizeof(uint8_t);

    nf4_array_t *arr = (nf4_array_t *)calloc(1, total);
    if (!arr) return NULL;

    arr->num_elements = num_elements;
    arr->num_blocks = num_blocks;
    arr->block_size = block_size;
    arr->block_scales = (float *)(arr + 1);
    arr->data = (uint8_t *)(arr->block_scales + num_blocks);
    return arr;
}

void free_nf4_array(nf4_array_t *nf4_array) {
    if (!nf4_array) return;
    free(nf4_array);
}

int64_t get_nf4_array_size(const nf4_array_t *nf4_array) {
    return _get_nf4_array_size(nf4_array);
}

nf4_array_t *load_nf4_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(nf4_array_t)) return NULL;

    nf4_array_t *arr = (nf4_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;

    memcpy(arr, buffer, buffer_size);
    const int64_t expected = _get_nf4_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }

    arr->block_scales = (float *)(arr + 1);
    arr->data = (uint8_t *)(arr->block_scales + arr->num_blocks);
    return arr;
}

static uint8_t float_to_nf4_code(float x) {
    if (!isfinite(x)) x = 0.0f;

    int best_idx = 0;
    float best_err = fabsf(x - NF4_LEVELS[0]);
    for (int i = 1; i < 16; ++i) {
        float err = fabsf(x - NF4_LEVELS[i]);
        if (err < best_err) {
            best_err = err;
            best_idx = i;
        }
    }
    return (uint8_t)best_idx;
}

static float nf4_code_to_fp32(uint8_t code) {
    return NF4_LEVELS[code & 0xF];
}

int nf4_compress(const float *float_array, unsigned long long num_elements, nf4_array_t **nf4_array) {
    if (!float_array || num_elements == 0 || !nf4_array || *nf4_array) return 1;

    nf4_array_t *arr = allocate_nf4_array(num_elements, DEFAULT_NF4_BLOCK_SIZE);
    if (!arr) return 1;

    const unsigned long long block_size = arr->block_size;
    const unsigned long long num_blocks = arr->num_blocks;
    const unsigned long long total = arr->num_elements;
    uint8_t *dst = arr->data;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= total) ? block_size : (total - start);

        float abs_max = 0.0f;
        for (unsigned long long i = 0; i < remain; ++i) {
            float v = float_array[start + i];
            if (!isfinite(v)) v = 0.0f;
            float av = fabsf(v);
            if (av > abs_max) abs_max = av;
        }
        float block_scale = (abs_max > 0.0f) ? abs_max : 1.0f;
        arr->block_scales[b] = block_scale;
        float inv_block_scale = 1.0f / block_scale;

        for (unsigned long long i = 0; i < remain; ++i) {
            float v = float_array[start + i] * inv_block_scale;
            uint8_t code = float_to_nf4_code(v) & 0xF;
            const unsigned long long packed_idx = (start + i) / 2;
            if (((start + i) % 2) == 0) {
                dst[packed_idx] = (uint8_t)(code << 4);
            } else {
                dst[packed_idx] |= code;
            }
        }
    }

    *nf4_array = arr;
    return 0;
}

int nf4_decompress(const nf4_array_t *nf4_array, float *float_array) {
    if (!nf4_array || !float_array) return 1;

    const unsigned long long block_size = nf4_array->block_size;
    const unsigned long long num_blocks = nf4_array->num_blocks;
    const unsigned long long num_elements = nf4_array->num_elements;
    const uint8_t *src = nf4_array->data;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements) ? block_size : (num_elements - start);

        float block_scale = nf4_array->block_scales[b];
        if (block_scale == 0.0f || !isfinite(block_scale)) block_scale = 1.0f;

        for (unsigned long long i = 0; i < remain; ++i) {
            const unsigned long long packed_idx = (start + i) / 2;
            uint8_t packed = src[packed_idx];
            uint8_t code = ((start + i) % 2 == 0) ? (packed >> 4) : (packed & 0xF);
            float val = nf4_code_to_fp32(code);
            float_array[start + i] = block_scale * val;
        }
    }
    return 0;
}
