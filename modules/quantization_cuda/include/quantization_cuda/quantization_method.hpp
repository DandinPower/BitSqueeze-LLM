#pragma once

#ifndef BITSQZ_LLM_QUANTIZATION_METHOD_T_DEFINED
#define BITSQZ_LLM_QUANTIZATION_METHOD_T_DEFINED
typedef enum {
    quantization_INVALID = -1,
    Q8_0 = 0,
    Q4_0 = 1,
    Q2_K = 2,
    BF16 = 3,
    FP16 = 4,
    FP8 = 5,
    FP4 = 6,
    MXFP8 = 7,
    MXFP4 = 8,
    NVFP4 = 9,
    NF4_DQ = 10,
    NF4 = 11,
    Q2_K_FAST = 12,
} quantization_method_t;
#endif
