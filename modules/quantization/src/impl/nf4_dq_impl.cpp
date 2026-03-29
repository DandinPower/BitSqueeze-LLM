#include <quantization/impl/nf4_dq_impl.hpp>

#define FP8_EXPONENT_BIAS 7
#define FP8_EXP_BITS      4
#define FP8_MANT_BITS     3

static const float NF4_DQ_LEVELS[16] = {
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

static long long _get_nf4_dq_array_size(const nf4_dq_array_t *nf4_dq_array) {
    if (!nf4_dq_array) return 0;
    const unsigned long long packed_elems = (nf4_dq_array->num_elements + 1) / 2;
    return sizeof(nf4_dq_array_t)
         + nf4_dq_array->num_blocks * sizeof(uint8_t)
         + packed_elems * sizeof(uint8_t);
}

nf4_dq_array_t *allocate_nf4_dq_array(unsigned long long num_elements,
                                      unsigned long long block_size) {
    if (!num_elements || !block_size) return NULL;

    unsigned long long num_blocks = (num_elements + block_size - 1) / block_size;
    unsigned long long packed_elems = (num_elements + 1) / 2;
    size_t total = sizeof(nf4_dq_array_t)
                 + num_blocks * sizeof(uint8_t)
                 + packed_elems * sizeof(uint8_t);

    nf4_dq_array_t *arr = (nf4_dq_array_t *)calloc(1, total);
    if (!arr) return NULL;

    arr->num_elements = num_elements;
    arr->num_blocks   = num_blocks;
    arr->block_size   = block_size;
    arr->dq_scale     = 1.0f;
    arr->block_scales = (uint8_t *)(arr + 1);
    arr->data         = (uint8_t *)(arr->block_scales + num_blocks);
    return arr;
}

void free_nf4_dq_array(nf4_dq_array_t *nf4_dq_array) {
    if (!nf4_dq_array) return;
    free(nf4_dq_array);
}

long long get_nf4_dq_array_size(const nf4_dq_array_t *nf4_dq_array) {
    return _get_nf4_dq_array_size(nf4_dq_array);
}

nf4_dq_array_t *load_nf4_dq_array_from_buffer(const void *buffer, long long buffer_size) {
    if (!buffer || buffer_size < (long long)sizeof(nf4_dq_array_t)) return NULL;

    nf4_dq_array_t *arr = (nf4_dq_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;

    memcpy(arr, buffer, buffer_size);
    const long long expected = _get_nf4_dq_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }

    arr->block_scales = (uint8_t *)(arr + 1);
    arr->data         = (uint8_t *)(arr->block_scales + arr->num_blocks);
    return arr;
}

static uint8_t fp32_to_e4m3(float x) {
    if (!isfinite(x)) x = (x < 0.0f) ? -NF4_DQ_FP8_MAX_NORM_VALUE : NF4_DQ_FP8_MAX_NORM_VALUE;
    const int sign = signbit(x) ? 1 : 0;
    float ax = fabsf(x);
    if (ax == 0.0f) return (uint8_t)(sign << 7);
    if (ax > NF4_DQ_FP8_MAX_NORM_VALUE) ax = NF4_DQ_FP8_MAX_NORM_VALUE;

    int exp2;
    float mant = frexpf(ax, &exp2);
    int exponent_field = exp2 - 1 + FP8_EXPONENT_BIAS;

    if (exponent_field <= 0) {
        int mant_field = (int)lrintf(ax * 512.0f);
        if (mant_field > 7) mant_field = 7;
        return (uint8_t)((sign << 7) | (mant_field & 0x7));
    }

    int mant_field = (int)lrintf(((mant * 2.0f) - 1.0f) * 8.0f);
    if (mant_field > 7) {
        mant_field = 0;
        exponent_field += 1;
    }

    if (exponent_field >= (1 << FP8_EXP_BITS)) {
        exponent_field = (1 << FP8_EXP_BITS) - 1;
        mant_field = 7;
    }

    return (uint8_t)((sign << 7) | ((exponent_field & 0xF) << 3) | (mant_field & 0x7));
}

static float e4m3_to_fp32(uint8_t v) {
    const int sign = (v >> 7) & 0x1;
    const int exponent_field = (v >> 3) & 0xF;
    const int mant_field = v & 0x7;

    float result;
    if (exponent_field == 0) {
        float mant = (float)mant_field / 8.0f;
        result = mant * ldexpf(1.0f, 1 - FP8_EXPONENT_BIAS);
    } else {
        int exponent = exponent_field - FP8_EXPONENT_BIAS;
        float mant = 1.0f + (float)mant_field / 8.0f;
        result = mant * ldexpf(1.0f, exponent);
    }
    return sign ? -result : result;
}

static uint8_t float_to_nf4_dq_code(float x) {
    if (!isfinite(x)) x = 0.0f;

    int best_idx = 0;
    float best_err = fabsf(x - NF4_DQ_LEVELS[0]);
    for (int i = 1; i < 16; ++i) {
        float err = fabsf(x - NF4_DQ_LEVELS[i]);
        if (err < best_err) {
            best_err = err;
            best_idx = i;
        }
    }
    return (uint8_t)best_idx;
}

static float nf4_dq_code_to_fp32(uint8_t code) {
    return NF4_DQ_LEVELS[code & 0xF];
}

static float choose_dq_scale(const float *block_scales, unsigned long long num_blocks) {
    float abs_max = 0.0f;
    for (unsigned long long i = 0; i < num_blocks; ++i) {
        float v = block_scales[i];
        if (!isfinite(v)) continue;
        float av = fabsf(v);
        if (av > abs_max) abs_max = av;
    }
    if (abs_max == 0.0f) return 1.0f;
    return abs_max / NF4_DQ_FP8_MAX_NORM_VALUE;
}

static int _quantize_nf4_dq(const float *float_array, nf4_dq_array_t *arr) {
    if (!float_array || !arr) return 1;

    const unsigned long long block_size   = arr->block_size;
    const unsigned long long num_blocks   = arr->num_blocks;
    const unsigned long long num_elements = arr->num_elements;
    uint8_t *dst = arr->data;

    float *block_scales = (float *)malloc(num_blocks * sizeof(float));
    if (!block_scales) return 1;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements)
                                  ? block_size
                                  : (num_elements - start);

        float abs_max = 0.0f;
        for (unsigned long long i = 0; i < remain; ++i) {
            float v = float_array[start + i];
            if (!isfinite(v)) v = 0.0f;
            float av = fabsf(v);
            if (av > abs_max) abs_max = av;
        }
        block_scales[b] = (abs_max > 0.0f) ? abs_max : 1.0f;
    }

    float dq_scale = choose_dq_scale(block_scales, num_blocks);
    if (dq_scale == 0.0f) dq_scale = 1.0f;
    arr->dq_scale = dq_scale;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements)
                                  ? block_size
                                  : (num_elements - start);

        uint8_t block_scale_code = fp32_to_e4m3(block_scales[b] / dq_scale);
        arr->block_scales[b] = block_scale_code;
        float block_scale = dq_scale * e4m3_to_fp32(block_scale_code);
        if (block_scale == 0.0f || !isfinite(block_scale)) block_scale = 1.0f;
        float inv_block_scale = 1.0f / block_scale;

        for (unsigned long long i = 0; i < remain; ++i) {
            float v = float_array[start + i] * inv_block_scale;
            uint8_t code = float_to_nf4_dq_code(v) & 0xF;
            const unsigned long long packed_idx = (start + i) / 2;
            if (((start + i) % 2) == 0) {
                dst[packed_idx] = (uint8_t)(code << 4);
            } else {
                dst[packed_idx] |= code;
            }
        }
    }

    free(block_scales);
    return 0;
}

int nf4_dq_compress(const float *float_array,
                    unsigned long long num_elements,
                    nf4_dq_array_t **nf4_dq_array) {
    if (!float_array || num_elements == 0 || !nf4_dq_array || *nf4_dq_array) return 1;

    nf4_dq_array_t *arr = allocate_nf4_dq_array(num_elements, DEFAULT_NF4_DQ_BLOCK_SIZE);
    if (!arr) return 1;

    if (_quantize_nf4_dq(float_array, arr)) {
        free_nf4_dq_array(arr);
        return 1;
    }

    *nf4_dq_array = arr;
    return 0;
}

int nf4_dq_decompress(const nf4_dq_array_t *nf4_dq_array,
                      float *float_array) {
    if (!nf4_dq_array || !float_array) return 1;

    const unsigned long long block_size   = nf4_dq_array->block_size;
    const unsigned long long num_blocks   = nf4_dq_array->num_blocks;
    const unsigned long long num_elements = nf4_dq_array->num_elements;
    const uint8_t *src = nf4_dq_array->data;
    const float dq_scale = nf4_dq_array->dq_scale;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements)
                                  ? block_size
                                  : (num_elements - start);

        float block_scale = dq_scale * e4m3_to_fp32(nf4_dq_array->block_scales[b]);
        if (block_scale == 0.0f || !isfinite(block_scale)) block_scale = 1.0f;

        for (unsigned long long i = 0; i < remain; ++i) {
            const unsigned long long packed_idx = (start + i) / 2;
            uint8_t packed = src[packed_idx];
            uint8_t code = ((start + i) % 2 == 0) ? (packed >> 4) : (packed & 0xF);
            float val = nf4_dq_code_to_fp32(code);
            float_array[start + i] = block_scale * val;
        }
    }
    return 0;
}
