#include <quantization/impl/q2_k_fast_impl.hpp>

#define MAX_VAL(a, b) ((a) > (b) ? (a) : (b))
#define MIN_VAL(a, b) ((a) < (b) ? (a) : (b))

static void find_fast_scale_and_min(const float *weights, float *scale, float *min_val) {
    const float q2_scale = 3.f;
    float local_min = INFINITY;
    float local_max = -INFINITY;

    for (int l = 0; l < Q2_K_BLOCK_SIZE; l++) {
        if (weights[l] < local_min) local_min = weights[l];
    }

    for (int l = 0; l < Q2_K_BLOCK_SIZE; l++) {
        float shifted = weights[l] - local_min;
        if (shifted > local_max) local_max = shifted;
    }

    *scale = local_max / q2_scale;
    *min_val = local_min;
}

int q2_k_fast_compress(const float *float_array, unsigned long long num_elements, q2_k_array_t **q2_k_array) {
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
        float mins[Q2_K_SUPER_BLOCK_SIZE];
        float scales[Q2_K_SUPER_BLOCK_SIZE];

        super_block_q2_k *curr_super_block = &qa->super_blocks[curr_super_block_index];
        const float *sb_base = float_array_aligned + (unsigned long long)curr_super_block_index * WEIGHT_PER_SUPER_BLOCK;

        float max_scale = -INFINITY;
        float max_abs_min = 0.f;

        for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
            memcpy(weights, sb_base + j * Q2_K_BLOCK_SIZE, Q2_K_BLOCK_SIZE * sizeof(float));
            find_fast_scale_and_min(weights, &scales[j], &mins[j]);
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

int q2_k_fast_decompress(const q2_k_array_t *q2_k_array, float *float_array) {
    return q2_k_decompress(q2_k_array, float_array);
}
