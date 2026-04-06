#ifndef BF16_H_
#define BF16_H_

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
#include <cstdint>
#include <cstring>
#include <cmath>
#endif

#if defined(__cplusplus) && __cplusplus >= 202002L
#include <bit>
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline float fp32_from_bf16_value(uint16_t b) {
    uint32_t u_b = (uint32_t)b << 16;
    float f;

#if defined(__cplusplus) && __cplusplus >= 202002L
    f = std::bit_cast<float>(u_b);
#else
    memcpy(&f, &u_b, sizeof(f));
#endif

    return f;
}

static inline uint16_t bf16_from_fp32_value(float f) {
    uint32_t u_f;

#if defined(__cplusplus) && __cplusplus >= 202002L
    u_f = std::bit_cast<uint32_t>(f);
#else
    memcpy(&u_f, &f, sizeof(u_f));
#endif

    if ((u_f & 0x7F800000) == 0x7F800000 && (u_f & 0x007FFFFF) != 0) {
        return (uint16_t)(u_f >> 16) | 0x0040;
    }

    uint32_t remainder = u_f & 0xFFFF;
    uint32_t lsb = (u_f >> 16) & 1;

    if (remainder > 0x8000 || (remainder == 0x8000 && lsb == 1)) {
        u_f += 0x10000;
    }

    return (uint16_t)(u_f >> 16);
}

#ifdef __cplusplus
}
#endif

#endif
