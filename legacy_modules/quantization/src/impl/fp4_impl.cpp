#include <quantization/impl/fp4_impl.hpp>

#define FP4_EXPONENT_BIAS 1
#define FP4_EXP_BITS 2
#define FP4_MANT_BITS 1

static int64_t _get_fp4_array_size(const fp4_array_t *fp4_array) {
    if (!fp4_array) return 0;
    const unsigned long long packed_elems = (fp4_array->num_elements + 1) / 2;
    return sizeof(fp4_array_t) + packed_elems * sizeof(uint8_t);
}

fp4_array_t *allocate_fp4_array(unsigned long long num_elements) {
    if (!num_elements) return NULL;

    unsigned long long packed_elems = (num_elements + 1) / 2;
    size_t total = sizeof(fp4_array_t) + packed_elems * sizeof(uint8_t);

    fp4_array_t *arr = (fp4_array_t *)calloc(1, total);
    if (!arr) return NULL;

    arr->num_elements = num_elements;
    arr->data = (uint8_t *)(arr + 1);
    arr->scale = 1.0f;
    return arr;
}

void free_fp4_array(fp4_array_t *fp4_array) {
    if (!fp4_array) return;
    free(fp4_array);
}

int64_t get_fp4_array_size(const fp4_array_t *fp4_array) {
    return _get_fp4_array_size(fp4_array);
}

fp4_array_t *load_fp4_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(fp4_array_t)) return NULL;

    fp4_array_t *arr = (fp4_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;

    memcpy(arr, buffer, buffer_size);
    const int64_t expected = _get_fp4_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }

    arr->data = (uint8_t *)(arr + 1);
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
    if (!isfinite(ax)) ax = FP4_MAX_NORM_VALUE;
    if (ax > FP4_MAX_NORM_VALUE) ax = FP4_MAX_NORM_VALUE;

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

static float choose_scale(const float *arr, unsigned long long n) {
    float abs_max = 0.0f;
    for (unsigned long long i = 0; i < n; ++i) {
        float v = arr[i];
        if (!isfinite(v)) continue;
        float av = fabsf(v);
        if (av > abs_max) abs_max = av;
    }
    if (abs_max == 0.0f) return 1.0f;
    return abs_max / FP4_MAX_NORM_VALUE;
}

int fp4_compress(const float *float_array, unsigned long long num_elements, fp4_array_t **fp4_array) {
    if (!float_array || num_elements == 0 || !fp4_array || *fp4_array) return 1;

    fp4_array_t *arr = allocate_fp4_array(num_elements);
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
        uint8_t code = fp32_to_e2m1(v) & 0xF;
        const unsigned long long packed_idx = i / 2;
        if ((i % 2) == 0) {
            arr->data[packed_idx] = (uint8_t)(code << 4);
        } else {
            arr->data[packed_idx] |= code;
        }
    }

    *fp4_array = arr;
    return 0;
}

int fp4_decompress(const fp4_array_t *fp4_array, float *float_array) {
    if (!fp4_array || !float_array) return 1;

    const float scale = fp4_array->scale;
    const uint8_t *src = fp4_array->data;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long i = 0; i < fp4_array->num_elements; ++i) {
        const unsigned long long packed_idx = i / 2;
        uint8_t packed = src[packed_idx];
        uint8_t code = (i % 2 == 0) ? (packed >> 4) : (packed & 0xF);
        float v = e2m1_to_fp32(code);
        float_array[i] = scale * v;
    }
    return 0;
}
