#include <quantization/impl/fp8_impl.hpp>

#define FP8_EXPONENT_BIAS 7
#define FP8_EXP_BITS 4
#define FP8_MANT_BITS 3

static int64_t _get_fp8_array_size(const fp8_array_t *fp8_array) {
    if (!fp8_array) return 0;
    return sizeof(fp8_array_t) + fp8_array->num_elements * sizeof(uint8_t);
}

fp8_array_t *allocate_fp8_array(unsigned long long num_elements) {
    if (!num_elements) return NULL;

    size_t total = sizeof(fp8_array_t) + num_elements * sizeof(uint8_t);
    fp8_array_t *arr = (fp8_array_t *)calloc(1, total);
    if (!arr) return NULL;

    arr->num_elements = num_elements;
    arr->data = (uint8_t *)(arr + 1);
    arr->scale = 1.0f;
    return arr;
}

void free_fp8_array(fp8_array_t *fp8_array) {
    if (!fp8_array) return;
    free(fp8_array);
}

int64_t get_fp8_array_size(const fp8_array_t *fp8_array) {
    return _get_fp8_array_size(fp8_array);
}

fp8_array_t *load_fp8_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(fp8_array_t)) return NULL;

    fp8_array_t *arr = (fp8_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;

    memcpy(arr, buffer, buffer_size);
    const int64_t expected = _get_fp8_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }

    arr->data = (uint8_t *)(arr + 1);
    return arr;
}

static uint8_t fp32_to_e4m3(float x) {
    if (!isfinite(x)) x = (x < 0.0f) ? -FP8_MAX_NORM_VALUE : FP8_MAX_NORM_VALUE;
    const int sign = signbit(x) ? 1 : 0;
    float ax = fabsf(x);
    if (ax == 0.0f) return (uint8_t)(sign << 7);
    if (ax > FP8_MAX_NORM_VALUE) ax = FP8_MAX_NORM_VALUE;

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

static float choose_scale(const float *arr, unsigned long long n) {
    float abs_max = 0.0f;
    for (unsigned long long i = 0; i < n; ++i) {
        float v = arr[i];
        if (!isfinite(v)) continue;
        float av = fabsf(v);
        if (av > abs_max) abs_max = av;
    }
    if (abs_max == 0.0f) return 1.0f;
    return abs_max / FP8_MAX_NORM_VALUE;
}

int fp8_compress(const float *float_array, unsigned long long num_elements, fp8_array_t **fp8_array) {
    if (!float_array || num_elements == 0 || !fp8_array || *fp8_array) return 1;

    fp8_array_t *arr = allocate_fp8_array(num_elements);
    if (!arr) return 1;

    float scale = choose_scale(float_array, num_elements);
    if (scale == 0.0f) scale = 1.0f;
    arr->scale = scale;
    float inv_scale = 1.0f / scale;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long i = 0; i < num_elements; ++i) {
        float v = float_array[i] * inv_scale;
        arr->data[i] = fp32_to_e4m3(v);
    }

    *fp8_array = arr;
    return 0;
}

int fp8_decompress(const fp8_array_t *fp8_array, float *float_array) {
    if (!fp8_array || !float_array) return 1;

    const float scale = fp8_array->scale;
#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long i = 0; i < fp8_array->num_elements; ++i) {
        float v = e4m3_to_fp32(fp8_array->data[i]);
        float_array[i] = scale * v;
    }
    return 0;
}
