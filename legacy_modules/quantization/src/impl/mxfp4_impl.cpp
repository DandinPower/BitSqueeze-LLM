#include <quantization/impl/mxfp4_impl.hpp>

#define FP4_EXPONENT_BIAS 1
#define FP4_EXP_BITS 2
#define FP4_MANT_BITS 1

static int64_t _get_mxfp4_array_size(const mxfp4_array_t *mxfp4_array) {
    if (!mxfp4_array) return 0;
    const unsigned long long packed_elems = (mxfp4_array->num_elements + 1) / 2;
    return sizeof(mxfp4_array_t) + mxfp4_array->num_blocks * sizeof(int8_t) + packed_elems * sizeof(uint8_t);
}

mxfp4_array_t *allocate_mxfp4_array(unsigned long long num_elements, unsigned long long block_size) {
    if (!num_elements || !block_size) return NULL;

    unsigned long long num_blocks = (num_elements + block_size - 1) / block_size;
    unsigned long long packed_elems = (num_elements + 1) / 2;
    size_t total = sizeof(mxfp4_array_t) + num_blocks * sizeof(int8_t) + packed_elems * sizeof(uint8_t);

    mxfp4_array_t *arr = (mxfp4_array_t *)calloc(1, total);
    if (!arr) return NULL;

    arr->num_elements = num_elements;
    arr->num_blocks = num_blocks;
    arr->block_size = block_size;
    arr->scales = (int8_t *)(arr + 1);
    arr->data = (uint8_t *)(arr->scales + num_blocks);
    return arr;
}

void free_mxfp4_array(mxfp4_array_t *mxfp4_array) {
    if (!mxfp4_array) return;
    free(mxfp4_array);
}

int64_t get_mxfp4_array_size(const mxfp4_array_t *mxfp4_array) {
    return _get_mxfp4_array_size(mxfp4_array);
}

mxfp4_array_t *load_mxfp4_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(mxfp4_array_t)) return NULL;

    mxfp4_array_t *arr = (mxfp4_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;

    memcpy(arr, buffer, buffer_size);
    const int64_t expected = _get_mxfp4_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }

    arr->scales = (int8_t *)(arr + 1);
    arr->data = (uint8_t *)(arr->scales + arr->num_blocks);
    return arr;
}

static void _build_fp4_levels(float levels[8]) {
    for (int exp_field = 0; exp_field < (1 << FP4_EXP_BITS); ++exp_field) {
        for (int mant_field = 0; mant_field < (1 << FP4_MANT_BITS); ++mant_field) {
            const int idx = (exp_field << FP4_MANT_BITS) | mant_field;
            float val;
            if (exp_field == 0) {
                val = ((float)mant_field / 2.0f) * ldexpf(1.0f, 1 - FP4_EXPONENT_BIAS);
            } else {
                float mant = 1.0f + ((float)mant_field / 2.0f);
                int exponent = exp_field - FP4_EXPONENT_BIAS;
                val = mant * ldexpf(1.0f, exponent);
            }
            levels[idx] = val;
        }
    }
    levels[0] = 0.0f;
}

static uint8_t fp32_to_e2m1(float x) {
    static float levels[8];
    static int initialized = 0;
    if (!initialized) {
        _build_fp4_levels(levels);
        initialized = 1;
    }

    const int sign = signbit(x) ? 1 : 0;
    float ax = fabsf(x);
    if (ax == 0.0f) return (uint8_t)(sign << 3);
    if (!isfinite(ax)) ax = MXFP4_MAX_NORM_VALUE;
    if (ax > MXFP4_MAX_NORM_VALUE) ax = MXFP4_MAX_NORM_VALUE;

    int best_idx = 0;
    float best_err = fabsf(levels[0] - ax);
    for (int idx = 1; idx < 8; ++idx) {
        float err = fabsf(levels[idx] - ax);
        if (err < best_err) {
            best_err = err;
            best_idx = idx;
        }
    }

    return (uint8_t)((sign << 3) | (best_idx & 0x7));
}

static float e2m1_to_fp32(uint8_t v) {
    const int sign = (v >> 3) & 0x1;
    const int exp_field = (v >> 1) & 0x3;
    const int mant_field = v & 0x1;

    float result;
    if (exp_field == 0) {
        result = ((float)mant_field / 2.0f) * ldexpf(1.0f, 1 - FP4_EXPONENT_BIAS);
    } else {
        float mant = 1.0f + ((float)mant_field / 2.0f);
        int exponent = exp_field - FP4_EXPONENT_BIAS;
        result = mant * ldexpf(1.0f, exponent);
    }
    return sign ? -result : result;
}

static int8_t choose_scale_exponent(float abs_max) {
    if (abs_max <= 0.0f) return 0;
    float target = abs_max / MXFP4_MAX_NORM_VALUE;
    if (target <= 0.0f) return 0;
    return (int8_t)ceilf(log2f(target));
}

static int _quantize_mxfp4(const float *float_array, mxfp4_array_t *arr) {
    if (!float_array || !arr) return 1;

    const unsigned long long block_size = arr->block_size;
    const unsigned long long num_blocks = arr->num_blocks;
    const unsigned long long num_elements = arr->num_elements;
    uint8_t *dst = arr->data;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements) ? block_size : (num_elements - start);

        float abs_max = 0.0f;
        for (unsigned long long i = 0; i < remain; ++i) {
            float v = float_array[start + i];
            if (!isfinite(v)) v = 0.0f;
            float av = fabsf(v);
            if (av > abs_max) abs_max = av;
        }

        int8_t scale_exp = choose_scale_exponent(abs_max);
        arr->scales[b] = scale_exp;
        float inv_scale = ldexpf(1.0f, -scale_exp);

        for (unsigned long long i = 0; i < remain; ++i) {
            float v = float_array[start + i] * inv_scale;
            uint8_t code = fp32_to_e2m1(v) & 0xF;
            const unsigned long long packed_idx = (start + i) / 2;
            if (((start + i) % 2) == 0) {
                dst[packed_idx] = (uint8_t)(code << 4);
            } else {
                dst[packed_idx] |= code;
            }
        }
    }
    return 0;
}

int mxfp4_compress(const float *float_array, unsigned long long num_elements, mxfp4_array_t **mxfp4_array) {
    if (!float_array || num_elements == 0 || !mxfp4_array || *mxfp4_array) return 1;

    mxfp4_array_t *arr = allocate_mxfp4_array(num_elements, DEFAULT_MXFP4_BLOCK_SIZE);
    if (!arr) return 1;

    if (_quantize_mxfp4(float_array, arr)) {
        free_mxfp4_array(arr);
        return 1;
    }

    *mxfp4_array = arr;
    return 0;
}

int mxfp4_decompress(const mxfp4_array_t *mxfp4_array, float *float_array) {
    if (!mxfp4_array || !float_array) return 1;

    const unsigned long long block_size = mxfp4_array->block_size;
    const unsigned long long num_blocks = mxfp4_array->num_blocks;
    const unsigned long long num_elements = mxfp4_array->num_elements;
    const uint8_t *src = mxfp4_array->data;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements) ? block_size : (num_elements - start);
        float scale = ldexpf(1.0f, mxfp4_array->scales[b]);

        for (unsigned long long i = 0; i < remain; ++i) {
            const unsigned long long packed_idx = (start + i) / 2;
            uint8_t packed = src[packed_idx];
            uint8_t code = ((start + i) % 2 == 0) ? (packed >> 4) : (packed & 0xF);
            float val = e2m1_to_fp32(code);
            float_array[start + i] = scale * val;
        }
    }
    return 0;
}
