#include <quantization/impl/q2_k_impl.hpp>

#define MAX_VAL(a, b) ((a) > (b) ? (a) : (b))
#define MIN_VAL(a, b) ((a) < (b) ? (a) : (b))

q2_k_array_t *allocate_q2_k_array(unsigned long long num_elements) {
    if (!num_elements) return NULL;

    unsigned long long num_elements_aligned = (num_elements % WEIGHT_PER_SUPER_BLOCK == 0)
        ? num_elements
        : num_elements + (WEIGHT_PER_SUPER_BLOCK - (num_elements % WEIGHT_PER_SUPER_BLOCK));

    unsigned long long num_super_blocks = num_elements_aligned / WEIGHT_PER_SUPER_BLOCK;

    size_t total = sizeof(q2_k_array_t) + num_super_blocks * sizeof(super_block_q2_k);
    q2_k_array_t *qa = (q2_k_array_t *)calloc(1, total);
    if (!qa) return NULL;

    qa->num_elements = num_elements;
    qa->num_elements_aligned = num_elements_aligned;
    qa->num_super_blocks = (uint32_t)num_super_blocks;
    qa->super_blocks = (super_block_q2_k *)(qa + 1);

    return qa;
}

void free_q2_k_array(q2_k_array_t *q2_k_array) {
    if (!q2_k_array) return;
    free(q2_k_array);
}

int64_t get_q2_k_array_size(const q2_k_array_t *q2_k_array) {
    if (!q2_k_array) return 0;
    return sizeof(q2_k_array_t) + q2_k_array->num_super_blocks * sizeof(super_block_q2_k);
}

q2_k_array_t *load_q2_k_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(q2_k_array_t)) return NULL;

    q2_k_array_t *q2_k_array = (q2_k_array_t *)calloc(1, buffer_size);
    if (!q2_k_array) return NULL;

    memcpy(q2_k_array, buffer, buffer_size);
    const int64_t expected = get_q2_k_array_size(q2_k_array);
    if (buffer_size < expected) {
        free(q2_k_array);
        return NULL;
    }

    q2_k_array->super_blocks = (super_block_q2_k *)(q2_k_array + 1);
    return q2_k_array;
}

static void find_optimal_scale_and_min(const float *weights, const float *abs_weights, float *scale, float *min_val) {
    uint8_t L[Q2_K_BLOCK_SIZE];
    uint8_t Laux[Q2_K_BLOCK_SIZE];

    float min = weights[0];
    float max = weights[0];
    float sum_w = abs_weights[0];
    float sum_x = sum_w * weights[0];

    for (int i = 1; i < Q2_K_BLOCK_SIZE; ++i) {
        if (weights[i] < min) min = weights[i];
        if (weights[i] > max) max = weights[i];
        sum_w += abs_weights[i];
        sum_x += abs_weights[i] * weights[i];
    }

    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < Q2_K_BLOCK_SIZE; ++i) L[i] = 0;
        *min_val = min;
        *scale = 0.f;
        return;
    }

    float iscale = 3.f / (max - min);
    float best_scale = 1.f / iscale;
    float best_error = 0.f;
    for (int i = 0; i < Q2_K_BLOCK_SIZE; ++i) {
        int l = (int)lrintf(iscale * (weights[i] - min));
        L[i] = MAX_VAL(0, MIN_VAL(3, l));
        float diff = best_scale * L[i] + min - weights[i];
        diff = fabsf(diff);
        float w = abs_weights[i];
        best_error += w * diff;
    }

    const float rmin = -0.5f;
    const float rdelta = 0.1f;
    const int nstep = 15;

    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + 3.f) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < Q2_K_BLOCK_SIZE; ++i) {
            int l = (int)lrintf(iscale * (weights[i] - min));
            l = MAX_VAL(0, MIN_VAL(3, l));
            Laux[i] = (uint8_t)l;
            float w = abs_weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * weights[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < Q2_K_BLOCK_SIZE; ++i) {
                float diff = this_scale * Laux[i] + this_min - weights[i];
                diff = fabsf(diff);
                float w = abs_weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < Q2_K_BLOCK_SIZE; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                best_scale = this_scale;
                min = this_min;
            }
        }
    }
    *scale = best_scale;
    *min_val = min;
}

int q2_k_compress(const float *float_array, unsigned long long num_elements, q2_k_array_t **q2_k_array) {
    const float q4_scale = 15.f;

    if (!float_array || num_elements == 0 || !q2_k_array || *q2_k_array) {
        return 1;
    }

    *q2_k_array = allocate_q2_k_array(num_elements);
    if (!*q2_k_array) {
        return 1;
    }
    q2_k_array_t *qa = *q2_k_array;

    float *float_array_aligned = (float *)calloc(qa->num_elements_aligned, sizeof(float));
    if (!float_array_aligned) {
        free_q2_k_array(qa);
        *q2_k_array = NULL;
        return 1;
    }
    memcpy(float_array_aligned, float_array, qa->num_elements * sizeof(float));

    const uint32_t num_super_blocks = qa->num_super_blocks;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint32_t curr_super_block_index = 0; curr_super_block_index < num_super_blocks; curr_super_block_index++) {
        uint8_t L[WEIGHT_PER_SUPER_BLOCK];
        float weights[Q2_K_BLOCK_SIZE];
        float abs_weights[Q2_K_BLOCK_SIZE];
        float mins[Q2_K_SUPER_BLOCK_SIZE];
        float scales[Q2_K_SUPER_BLOCK_SIZE];

        super_block_q2_k *curr_super_block = &qa->super_blocks[curr_super_block_index];
        const float *sb_base = float_array_aligned + (unsigned long long)curr_super_block_index * WEIGHT_PER_SUPER_BLOCK;

        float max_scale = -INFINITY;
        float max_abs_min = 0.f;

        for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
            memcpy(weights, sb_base + j * Q2_K_BLOCK_SIZE, Q2_K_BLOCK_SIZE * sizeof(float));
            for (int i = 0; i < Q2_K_BLOCK_SIZE; ++i) {
                abs_weights[i] = fabsf(weights[i]);
            }
            find_optimal_scale_and_min(weights, abs_weights, &scales[j], &mins[j]);
            if (scales[j] > max_scale) {
                max_scale = scales[j];
            }
            if (fabsf(mins[j]) > max_abs_min) {
                max_abs_min = fabsf(mins[j]);
            }
        }

        if (max_scale > 0) {
            float iscale = q4_scale / max_scale;
            for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
                int l = (int)lrintf(iscale * scales[j]);
                curr_super_block->scales[j] = l;
            }
            curr_super_block->super_scale = fp16_ieee_from_fp32_value(max_scale / q4_scale);
        } else {
            memset(curr_super_block->scales, 0, sizeof(curr_super_block->scales));
            curr_super_block->super_scale = fp16_ieee_from_fp32_value(0.f);
        }

        if (max_abs_min > 0) {
            const float iscale = 7.f / max_abs_min;
            for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
                int l = (int)lrintf(iscale * mins[j]);
                l = MAX_VAL(-8, MIN_VAL(7, l));
                curr_super_block->scales[j] |= ((l & 0xF) << 4);
            }
            curr_super_block->super_min = fp16_ieee_from_fp32_value(max_abs_min / 7.f);
        } else {
            curr_super_block->super_min = fp16_ieee_from_fp32_value(0.f);
        }

        for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
            const float temp_scale = fp16_ieee_to_fp32_value(curr_super_block->super_scale) * (curr_super_block->scales[j] & 0xF);
            const float m = fp16_ieee_to_fp32_value(curr_super_block->super_min);
            const int8_t min_q = (curr_super_block->scales[j] >> 4);
            const float temp_min = m * ((int8_t)(min_q << 4) >> 4);

            for (int ii = 0; ii < Q2_K_BLOCK_SIZE; ii++) {
                float val = (temp_scale > 0.f) ? (sb_base[j * Q2_K_BLOCK_SIZE + ii] - temp_min) / temp_scale : 0.f;
                int l = (int)lrintf(val);
                l = MAX_VAL(0, MIN_VAL(3, l));
                L[j * Q2_K_BLOCK_SIZE + ii] = (uint8_t)l;
            }
        }

        uint32_t packed_run = WEIGHT_PER_SUPER_BLOCK / 2;
        for (int j = 0; j < WEIGHT_PER_SUPER_BLOCK; j += packed_run) {
            for (int l = 0; l < Q2_K_BLOCK_SIZE * 2; l++) {
                uint8_t b0 = L[j + l + 0];
                uint8_t b1 = L[j + l + 32];
                uint8_t b2 = L[j + l + 64];
                uint8_t b3 = L[j + l + 96];
                curr_super_block->data[j / 4 + l] = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6);
            }
        }
    }

    free(float_array_aligned);
    return 0;
}

int q2_k_im_compress(const float *float_array, const float *importance_array, unsigned long long num_elements, q2_k_array_t **q2_k_array) {
    const float q4_scale = 15.f;

    if (!float_array || !importance_array || num_elements == 0 || !q2_k_array || *q2_k_array) {
        return 1;
    }

    *q2_k_array = allocate_q2_k_array(num_elements);
    if (!*q2_k_array) {
        return 1;
    }
    q2_k_array_t *qa = *q2_k_array;

    float *float_array_aligned = (float *)calloc(qa->num_elements_aligned, sizeof(float));
    if (!float_array_aligned) {
        free_q2_k_array(qa);
        *q2_k_array = NULL;
        return 1;
    }
    float *importance_array_aligned = (float *)calloc(qa->num_elements_aligned, sizeof(float));
    if (!importance_array_aligned) {
        free(float_array_aligned);
        free_q2_k_array(qa);
        *q2_k_array = NULL;
        return 1;
    }

    memcpy(float_array_aligned, float_array, qa->num_elements * sizeof(float));
    memcpy(importance_array_aligned, importance_array, qa->num_elements * sizeof(float));

    const uint32_t num_super_blocks = qa->num_super_blocks;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint32_t curr_super_block_index = 0; curr_super_block_index < num_super_blocks; curr_super_block_index++) {
        uint8_t L[WEIGHT_PER_SUPER_BLOCK];
        float weights[Q2_K_BLOCK_SIZE];
        float abs_weights[Q2_K_BLOCK_SIZE];
        float mins[Q2_K_SUPER_BLOCK_SIZE];
        float scales[Q2_K_SUPER_BLOCK_SIZE];

        super_block_q2_k *curr_super_block = &qa->super_blocks[curr_super_block_index];
        const float *sb_base = float_array_aligned + (unsigned long long)curr_super_block_index * WEIGHT_PER_SUPER_BLOCK;
        const float *im_sb_base = importance_array_aligned + (unsigned long long)curr_super_block_index * WEIGHT_PER_SUPER_BLOCK;

        float max_scale = -INFINITY;
        float max_abs_min = 0.f;

        for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
            memcpy(weights, sb_base + j * Q2_K_BLOCK_SIZE, Q2_K_BLOCK_SIZE * sizeof(float));
            memcpy(abs_weights, im_sb_base + j * Q2_K_BLOCK_SIZE, Q2_K_BLOCK_SIZE * sizeof(float));

            find_optimal_scale_and_min(weights, abs_weights, &scales[j], &mins[j]);
            if (scales[j] > max_scale) {
                max_scale = scales[j];
            }
            if (fabsf(mins[j]) > max_abs_min) {
                max_abs_min = fabsf(mins[j]);
            }
        }

        if (max_scale > 0) {
            float iscale = q4_scale / max_scale;
            for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
                int l = (int)lrintf(iscale * scales[j]);
                curr_super_block->scales[j] = l;
            }
            curr_super_block->super_scale = fp16_ieee_from_fp32_value(max_scale / q4_scale);
        } else {
            memset(curr_super_block->scales, 0, sizeof(curr_super_block->scales));
            curr_super_block->super_scale = fp16_ieee_from_fp32_value(0.f);
        }

        if (max_abs_min > 0) {
            const float iscale = 7.f / max_abs_min;
            for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
                int l = (int)lrintf(iscale * mins[j]);
                l = MAX_VAL(-8, MIN_VAL(7, l));
                curr_super_block->scales[j] |= ((l & 0xF) << 4);
            }
            curr_super_block->super_min = fp16_ieee_from_fp32_value(max_abs_min / 7.f);
        } else {
            curr_super_block->super_min = fp16_ieee_from_fp32_value(0.f);
        }

        for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
            const float temp_scale = fp16_ieee_to_fp32_value(curr_super_block->super_scale) * (curr_super_block->scales[j] & 0xF);
            const float m = fp16_ieee_to_fp32_value(curr_super_block->super_min);
            const int8_t min_q = (curr_super_block->scales[j] >> 4);
            const float temp_min = m * ((int8_t)(min_q << 4) >> 4);

            for (int ii = 0; ii < Q2_K_BLOCK_SIZE; ii++) {
                float val = (temp_scale > 0.f) ? (sb_base[j * Q2_K_BLOCK_SIZE + ii] - temp_min) / temp_scale : 0.f;
                int l = (int)lrintf(val);
                l = MAX_VAL(0, MIN_VAL(3, l));
                L[j * Q2_K_BLOCK_SIZE + ii] = (uint8_t)l;
            }
        }

        uint32_t packed_run = WEIGHT_PER_SUPER_BLOCK / 2;
        for (int j = 0; j < WEIGHT_PER_SUPER_BLOCK; j += packed_run) {
            for (int l = 0; l < Q2_K_BLOCK_SIZE * 2; l++) {
                uint8_t b0 = L[j + l + 0];
                uint8_t b1 = L[j + l + 32];
                uint8_t b2 = L[j + l + 64];
                uint8_t b3 = L[j + l + 96];
                curr_super_block->data[j / 4 + l] = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6);
            }
        }
    }

    free(float_array_aligned);
    free(importance_array_aligned);
    return 0;
}

int q2_k_decompress(const q2_k_array_t *q2_k_array, float *float_array) {
    if (!q2_k_array || !float_array || q2_k_array->num_super_blocks == 0) {
        return 1;
    }

    const unsigned long long total_elements = q2_k_array->num_elements;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint32_t s = 0; s < q2_k_array->num_super_blocks; ++s) {
        const super_block_q2_k *curr_super_block = &q2_k_array->super_blocks[s];
        const float super_scale = fp16_ieee_to_fp32_value(curr_super_block->super_scale);
        const float super_min = fp16_ieee_to_fp32_value(curr_super_block->super_min);

        float scales[Q2_K_SUPER_BLOCK_SIZE];
        float mins[Q2_K_SUPER_BLOCK_SIZE];

        for (int i = 0; i < Q2_K_SUPER_BLOCK_SIZE; ++i) {
            uint8_t packed_val = curr_super_block->scales[i];
            scales[i] = super_scale * (packed_val & 0x0F);

            int8_t min_q = (packed_val >> 4);
            mins[i] = super_min * ((int8_t)(min_q << 4) >> 4);
        }

        const uint8_t *q = curr_super_block->data;
        const unsigned long long base_idx = (unsigned long long)s * WEIGHT_PER_SUPER_BLOCK;

        for (int l = 0; l < 32; ++l) {
            uint8_t packed_byte = q[l];

            unsigned long long idx0 = base_idx + (unsigned long long)l;
            unsigned long long idx1 = base_idx + (unsigned long long)l + 32;
            unsigned long long idx2 = base_idx + (unsigned long long)l + 64;
            unsigned long long idx3 = base_idx + (unsigned long long)l + 96;

            const int local0 = l;
            const int local1 = l + 32;
            const int local2 = l + 64;
            const int local3 = l + 96;

            if (idx0 < total_elements) float_array[idx0] = mins[local0 / 16] + scales[local0 / 16] * ((packed_byte >> 0) & 3);
            if (idx1 < total_elements) float_array[idx1] = mins[local1 / 16] + scales[local1 / 16] * ((packed_byte >> 2) & 3);
            if (idx2 < total_elements) float_array[idx2] = mins[local2 / 16] + scales[local2 / 16] * ((packed_byte >> 4) & 3);
            if (idx3 < total_elements) float_array[idx3] = mins[local3 / 16] + scales[local3 / 16] * ((packed_byte >> 6) & 3);
        }

        for (int l = 0; l < 32; ++l) {
            uint8_t packed_byte = q[32 + l];

            unsigned long long idx0 = base_idx + 128 + (unsigned long long)l;
            unsigned long long idx1 = base_idx + 160 + (unsigned long long)l;
            unsigned long long idx2 = base_idx + 192 + (unsigned long long)l;
            unsigned long long idx3 = base_idx + 224 + (unsigned long long)l;

            if (idx0 < total_elements) float_array[idx0] = mins[(idx0 - base_idx) / 16] + scales[(idx0 - base_idx) / 16] * ((packed_byte >> 0) & 3);
            if (idx1 < total_elements) float_array[idx1] = mins[(idx1 - base_idx) / 16] + scales[(idx1 - base_idx) / 16] * ((packed_byte >> 2) & 3);
            if (idx2 < total_elements) float_array[idx2] = mins[(idx2 - base_idx) / 16] + scales[(idx2 - base_idx) / 16] * ((packed_byte >> 4) & 3);
            if (idx3 < total_elements) float_array[idx3] = mins[(idx3 - base_idx) / 16] + scales[(idx3 - base_idx) / 16] * ((packed_byte >> 6) & 3);
        }
    }
    return 0;
}
