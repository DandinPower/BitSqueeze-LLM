#include <quantization/impl/mxfp8_impl.hpp>

#define FP8_EXPONENT_BIAS 7
#define FP8_EXP_BITS 4
#define FP8_MANT_BITS 3

static int64_t _get_mxfp8_array_size(const mxfp8_array_t *mxfp8_array) {
    if (!mxfp8_array) return 0;
    return sizeof(mxfp8_array_t) + mxfp8_array->num_blocks * sizeof(int8_t) + mxfp8_array->num_elements * sizeof(uint8_t);
}

mxfp8_array_t *allocate_mxfp8_array(unsigned long long num_elements, unsigned long long block_size) {
    if (!num_elements || !block_size) return NULL;

    unsigned long long num_blocks = (num_elements + block_size - 1) / block_size;
    size_t total = sizeof(mxfp8_array_t) + num_blocks * sizeof(int8_t) + num_elements * sizeof(uint8_t);

    mxfp8_array_t *arr = (mxfp8_array_t *)calloc(1, total);
    if (!arr) return NULL;

    arr->num_elements = num_elements;
    arr->num_blocks = num_blocks;
    arr->block_size = block_size;
    arr->scales = (int8_t *)(arr + 1);
    arr->data = (uint8_t *)(arr->scales + num_blocks);
    return arr;
}

void free_mxfp8_array(mxfp8_array_t *mxfp8_array) {
    if (!mxfp8_array) return;
    free(mxfp8_array);
}

int64_t get_mxfp8_array_size(const mxfp8_array_t *mxfp8_array) {
    return _get_mxfp8_array_size(mxfp8_array);
}

mxfp8_array_t *load_mxfp8_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(mxfp8_array_t)) return NULL;

    mxfp8_array_t *arr = (mxfp8_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;

    memcpy(arr, buffer, buffer_size);
    const int64_t expected = _get_mxfp8_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }

    arr->scales = (int8_t *)(arr + 1);
    arr->data = (uint8_t *)(arr->scales + arr->num_blocks);
    return arr;
}

static uint8_t fp32_to_e4m3(float x) {
    if (!isfinite(x)) x = (x < 0.0f) ? -MXFP8_MAX_NORM_VALUE : MXFP8_MAX_NORM_VALUE;
    const int sign = signbit(x) ? 1 : 0;
    float ax = fabsf(x);
    if (ax == 0.0f) return (uint8_t)(sign << 7);
    if (ax > MXFP8_MAX_NORM_VALUE) ax = MXFP8_MAX_NORM_VALUE;

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

static int8_t choose_scale_exponent(float abs_max) {
    if (abs_max <= 0.0f) return 0;
    float target = abs_max / MXFP8_MAX_NORM_VALUE;
    if (target <= 0.0f) return 0;
    int8_t exp2 = (int8_t)ceilf(log2f(target));
    return exp2;
}

static int _quantize_mxfp8(const float *float_array, mxfp8_array_t *arr) {
    if (!float_array || !arr) return 1;

    const unsigned long long block_size = arr->block_size;
    const unsigned long long num_blocks = arr->num_blocks;
    const unsigned long long num_elements = arr->num_elements;

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
        float scale = ldexpf(1.0f, scale_exp);

        for (unsigned long long i = 0; i < remain; ++i) {
            float v = float_array[start + i] / scale;
            arr->data[start + i] = fp32_to_e4m3(v);
        }
    }
    return 0;
}

int mxfp8_compress(const float *float_array, unsigned long long num_elements, mxfp8_array_t **mxfp8_array) {
    if (!float_array || num_elements == 0 || !mxfp8_array || *mxfp8_array) return 1;

    mxfp8_array_t *arr = allocate_mxfp8_array(num_elements, DEFAULT_MXFP8_BLOCK_SIZE);
    if (!arr) return 1;

    if (_quantize_mxfp8(float_array, arr)) {
        free_mxfp8_array(arr);
        return 1;
    }

    *mxfp8_array = arr;
    return 0;
}

int mxfp8_decompress(const mxfp8_array_t *mxfp8_array, float *float_array) {
    if (!mxfp8_array || !float_array) return 1;

    const unsigned long long block_size = mxfp8_array->block_size;
    const unsigned long long num_blocks = mxfp8_array->num_blocks;
    const unsigned long long num_elements = mxfp8_array->num_elements;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements) ? block_size : (num_elements - start);
        float scale = ldexpf(1.0f, mxfp8_array->scales[b]);

        for (unsigned long long i = 0; i < remain; ++i) {
            float val = e4m3_to_fp32(mxfp8_array->data[start + i]);
            float_array[start + i] = scale * val;
        }
    }
    return 0;
}
