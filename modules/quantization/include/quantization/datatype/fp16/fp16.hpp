#pragma once
#ifndef FP16_FP16_H
#define FP16_FP16_H

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#include <cstdint>
#include <cmath>
#elif !defined(__OPENCL_VERSION__)
#include <stdint.h>
#include <math.h>
#endif

#include <quantization/datatype/fp16/bitcasts.hpp>
#include <quantization/datatype/fp16/macros.hpp>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(__F16C__) && FP16_USE_NATIVE_CONVERSION && !FP16_USE_FLOAT16_TYPE && !FP16_USE_FP16_TYPE
#include <immintrin.h>
#endif
#if (defined(__aarch64__) || defined(_M_ARM64)) && FP16_USE_NATIVE_CONVERSION && !FP16_USE_FLOAT16_TYPE && !FP16_USE_FP16_TYPE
#include <arm_neon.h>
#endif

static inline uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
#ifdef _MSC_VER
    unsigned long nonsign_bsr;
    _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
    uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
    uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
    renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
    const int32_t inf_nan_mask = ((int32_t)(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
    const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
    return sign | ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) | inf_nan_mask) & ~zero_mask);
}

static inline float fp16_ieee_to_fp32_value(uint16_t h) {
#if FP16_USE_NATIVE_CONVERSION
#if FP16_USE_FLOAT16_TYPE
    union {
        uint16_t as_bits;
        _Float16 as_value;
    } fp16 = { h };
    return (float)fp16.as_value;
#elif FP16_USE_FP16_TYPE
    union {
        uint16_t as_bits;
        __fp16 as_value;
    } fp16 = { h };
    return (float)fp16.as_value;
#else
#if (defined(__INTEL_COMPILER) || defined(__GNUC__)) && defined(__F16C__)
    return _cvtsh_ss((unsigned short)h);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)) && defined(__AVX2__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)(unsigned int)h)));
#elif defined(_M_ARM64) || defined(__aarch64__)
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
#else
#error "Archtecture- or compiler-specific implementation required"
#endif
#endif
#else
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
#endif
}

static inline uint16_t fp16_ieee_from_fp32_value(float f) {
#if FP16_USE_NATIVE_CONVERSION
#if FP16_USE_FLOAT16_TYPE
    union {
        _Float16 as_value;
        uint16_t as_bits;
    } fp16 = { (_Float16)f };
    return fp16.as_bits;
#elif FP16_USE_FP16_TYPE
    union {
        __fp16 as_value;
        uint16_t as_bits;
    } fp16 = { (__fp16)f };
    return fp16.as_bits;
#else
#if (defined(__INTEL_COMPILER) || defined(__GNUC__)) && defined(__F16C__)
    return _cvtss_sh(f, _MM_FROUND_CUR_DIRECTION);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)) && defined(__AVX2__)
    return (uint16_t)_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_CUR_DIRECTION));
#elif defined(_M_ARM64) || defined(__aarch64__)
    return vget_lane_u16(vcvt_f16_f32(vdupq_n_f32(f)), 0);
#else
#error "Archtecture- or compiler-specific implementation required"
#endif
#endif
#else
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
#if defined(_MSC_VER) && defined(_M_IX86_FP) && (_M_IX86_FP == 0) || defined(__GNUC__) && defined(__FLT_EVAL_METHOD__) && (__FLT_EVAL_METHOD__ != 0)
    const volatile float saturated_f = fabsf(f) * scale_to_inf;
#else
    const float saturated_f = fabsf(f) * scale_to_inf;
#endif
    float base = saturated_f * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
#endif
}

static inline uint32_t fp16_alt_to_fp32_bits(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
#ifdef _MSC_VER
    unsigned long nonsign_bsr;
    _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
    uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
    uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
    renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
    const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
    return sign | (((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) & ~zero_mask);
}

static inline float fp16_alt_to_fp32_value(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0x70) << 23;
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset);

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline uint16_t fp16_alt_from_fp32_value(float f) {
    const uint32_t w = fp32_to_bits(f);
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t shl1_w = w + w;

    const uint32_t shl1_max_fp16_fp32 = UINT32_C(0x8FFFC000);
    const uint32_t shl1_base = shl1_w > shl1_max_fp16_fp32 ? shl1_max_fp16_fp32 : shl1_w;
    uint32_t shl1_bias = shl1_base & UINT32_C(0xFF000000);
    const uint32_t exp_difference = 23 - 10;
    const uint32_t shl1_bias_min = (127 - 1 - exp_difference) << 24;
    if (shl1_bias < shl1_bias_min) {
        shl1_bias = shl1_bias_min;
    }

    const float bias = fp32_from_bits((shl1_bias >> 1) + ((exp_difference + 2) << 23));
    const float base = fp32_from_bits((shl1_base >> 1) + (2 << 23)) + bias;

    const uint32_t exp_f = fp32_to_bits(base) >> 13;
    return (sign >> 16) | ((exp_f & UINT32_C(0x00007C00)) + (fp32_to_bits(base) & UINT32_C(0x00000FFF)));
}

#endif
