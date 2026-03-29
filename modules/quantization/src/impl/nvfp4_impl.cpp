#include <quantization/impl/nvfp4_impl.hpp>

#define FP8_EXPONENT_BIAS 7
#define FP8_EXP_BITS 4
#define FP8_MANT_BITS 3

#define FP4_EXPONENT_BIAS 1
#define FP4_EXP_BITS 2
#define FP4_MANT_BITS 1

static int64_t _get_nvfp4_array_size(const nvfp4_array_t *nvfp4_array) {
    if (!nvfp4_array) return 0;
    const unsigned long long packed_elems = (nvfp4_array->num_elements + 1) / 2;
    return sizeof(nvfp4_array_t) + nvfp4_array->num_blocks * sizeof(uint8_t) + packed_elems * sizeof(uint8_t);
}

nvfp4_array_t *allocate_nvfp4_array(unsigned long long num_elements, unsigned long long block_size) {
    if (!num_elements || !block_size) return NULL;

    unsigned long long num_blocks = (num_elements + block_size - 1) / block_size;
    unsigned long long packed_elems = (num_elements + 1) / 2;
    size_t total = sizeof(nvfp4_array_t) + num_blocks * sizeof(uint8_t) + packed_elems * sizeof(uint8_t);

    nvfp4_array_t *arr = (nvfp4_array_t *)calloc(1, total);
    if (!arr) return NULL;

    arr->num_elements = num_elements;
    arr->num_blocks = num_blocks;
    arr->block_size = block_size;
    arr->block_scales = (uint8_t *)(arr + 1);
    arr->data = (uint8_t *)(arr->block_scales + num_blocks);
    arr->tensor_scale = 1.0f;
    return arr;
}

void free_nvfp4_array(nvfp4_array_t *nvfp4_array) {
    if (!nvfp4_array) return;
    free(nvfp4_array);
}

int64_t get_nvfp4_array_size(const nvfp4_array_t *nvfp4_array) {
    return _get_nvfp4_array_size(nvfp4_array);
}

nvfp4_array_t *load_nvfp4_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(nvfp4_array_t)) return NULL;

    nvfp4_array_t *arr = (nvfp4_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;

    memcpy(arr, buffer, buffer_size);
    const int64_t expected = _get_nvfp4_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }

    arr->block_scales = (uint8_t *)(arr + 1);
    arr->data = (uint8_t *)(arr->block_scales + arr->num_blocks);
    return arr;
}

static uint8_t fp32_to_e4m3(float x) {
    if (!isfinite(x)) x = (x < 0.0f) ? -NVFP4_FP8_MAX_NORM : NVFP4_FP8_MAX_NORM;
    const int sign = signbit(x) ? 1 : 0;
    float ax = fabsf(x);
    if (ax == 0.0f) return (uint8_t)(sign << 7);
    if (ax > NVFP4_FP8_MAX_NORM) ax = NVFP4_FP8_MAX_NORM;

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
    if (!isfinite(ax)) ax = NVFP4_MAX_NORM_VALUE;
    if (ax > NVFP4_MAX_NORM_VALUE) ax = NVFP4_MAX_NORM_VALUE;

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

static float choose_tensor_scale(const float *arr, unsigned long long n) {
    float abs_max = 0.0f;
    for (unsigned long long i = 0; i < n; ++i) {
        float v = arr[i];
        if (!isfinite(v)) continue;
        float av = fabsf(v);
        if (av > abs_max) abs_max = av;
    }
    if (abs_max == 0.0f) return 1.0f;
    return abs_max / NVFP4_MAX_NORM_VALUE;
}

static uint8_t choose_block_scale_fp8(const float *arr, unsigned long long start, unsigned long long len, float tensor_scale) {
    float abs_max = 0.0f;
    for (unsigned long long i = 0; i < len; ++i) {
        float v = arr[start + i] / tensor_scale;
        if (!isfinite(v)) v = 0.0f;
        float av = fabsf(v);
        if (av > abs_max) abs_max = av;
    }
    float target = abs_max / NVFP4_MAX_NORM_VALUE;
    float scale = (target > 0.0f) ? target : 1.0f;
    return fp32_to_e4m3(scale);
}

static int _quantize_nvfp4(const float *float_array, nvfp4_array_t *arr) {
    if (!float_array || !arr) return 1;

    const unsigned long long block_size = arr->block_size;
    const unsigned long long num_blocks = arr->num_blocks;
    const unsigned long long num_elements = arr->num_elements;
    uint8_t *dst = arr->data;

    arr->tensor_scale = choose_tensor_scale(float_array, num_elements);
    float inv_tensor_scale = 1.0f / arr->tensor_scale;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements) ? block_size : (num_elements - start);

        uint8_t block_scale_code = choose_block_scale_fp8(float_array, start, remain, arr->tensor_scale);
        arr->block_scales[b] = block_scale_code;
        float block_scale = e4m3_to_fp32(block_scale_code);
        float inv_block_scale = 1.0f / block_scale;

        for (unsigned long long i = 0; i < remain; ++i) {
            float v = float_array[start + i] * inv_tensor_scale * inv_block_scale;
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

int nvfp4_compress(const float *float_array, unsigned long long num_elements, nvfp4_array_t **nvfp4_array) {
    if (!float_array || num_elements == 0 || !nvfp4_array || *nvfp4_array) return 1;

    nvfp4_array_t *arr = allocate_nvfp4_array(num_elements, DEFAULT_NVFP4_BLOCK_SIZE);
    if (!arr) return 1;

    if (_quantize_nvfp4(float_array, arr)) {
        free_nvfp4_array(arr);
        return 1;
    }

    *nvfp4_array = arr;
    return 0;
}

int nvfp4_decompress(const nvfp4_array_t *nvfp4_array, float *float_array) {
    if (!nvfp4_array || !float_array) return 1;

    const unsigned long long block_size = nvfp4_array->block_size;
    const unsigned long long num_blocks = nvfp4_array->num_blocks;
    const unsigned long long num_elements = nvfp4_array->num_elements;
    const uint8_t *src = nvfp4_array->data;
    const float tensor_scale = nvfp4_array->tensor_scale;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (unsigned long long b = 0; b < num_blocks; ++b) {
        const unsigned long long start = b * block_size;
        const unsigned long long remain = (start + block_size <= num_elements) ? block_size : (num_elements - start);
        float block_scale = e4m3_to_fp32(nvfp4_array->block_scales[b]);
        float scale = tensor_scale * block_scale;

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
