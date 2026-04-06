#pragma once

#include <quantization/impl/q2_k_impl.hpp>

int q2_k_fast_compress(const float *float_array, unsigned long long num_elements, q2_k_array_t **q2_k_array);
int q2_k_fast_decompress(const q2_k_array_t *q2_k_array, float *float_array);
