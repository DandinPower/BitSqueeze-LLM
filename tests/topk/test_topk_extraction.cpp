#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

#include <topk.hpp>

static bool almost_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

static std::set<uint16_t> expected_topk_indices(const float *row, uint16_t num_columns, uint16_t k) {
    std::vector<uint16_t> idx(num_columns);
    for (uint16_t i = 0; i < num_columns; ++i) idx[i] = i;

    std::sort(idx.begin(), idx.end(), [row](uint16_t a, uint16_t b) {
        return std::fabs(row[a]) > std::fabs(row[b]);
    });

    std::set<uint16_t> out;
    for (uint16_t i = 0; i < k; ++i) out.insert(idx[i]);
    return out;
}

int main(void) {
    const uint16_t rows = 3;
    const uint16_t cols = 6;
    const float ratio = 0.5f;

    std::vector<float> input = {
        0.4f, -9.3f, 2.1f, -5.2f, 7.8f, -1.0f,
        -6.6f, 1.3f, 8.4f, -2.5f, 0.7f, 3.9f,
        4.2f, -0.8f, 5.1f, -7.7f, 2.6f, 9.9f,
    };
    std::vector<float> original = input;

    topk_array_t *topk_array = NULL;
    if (topk_extraction(input.data(), rows, cols, ratio, &topk_array) != 0 || !topk_array) {
        std::fprintf(stderr, "topk_extraction failed\n");
        return EXIT_FAILURE;
    }

    if (topk_array->num_rows != rows || topk_array->num_columns != cols || topk_array->num_topk_columns != 3) {
        std::fprintf(stderr, "unexpected topk_array shape\n");
        free_topk_array(topk_array);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < input.size(); ++i) {
        if (!almost_equal(input[i], original[i])) {
            std::fprintf(stderr, "input mutated by topk_extraction at %zu\n", i);
            free_topk_array(topk_array);
            return EXIT_FAILURE;
        }
    }

    for (uint16_t r = 0; r < rows; ++r) {
        std::set<uint16_t> actual_indices;
        const uint32_t base = (uint32_t)r * topk_array->num_topk_columns;
        for (uint16_t j = 0; j < topk_array->num_topk_columns; ++j) {
            uint16_t c = topk_array->topk_indices[base + j];
            if (c >= cols) {
                std::fprintf(stderr, "invalid topk index\n");
                free_topk_array(topk_array);
                return EXIT_FAILURE;
            }
            if (!actual_indices.insert(c).second) {
                std::fprintf(stderr, "duplicate topk index in row\n");
                free_topk_array(topk_array);
                return EXIT_FAILURE;
            }

            float expected_val = original[(uint32_t)r * cols + c];
            float actual_val = topk_array->values[base + j];
            if (!almost_equal(expected_val, actual_val)) {
                std::fprintf(stderr, "topk value mismatch\n");
                free_topk_array(topk_array);
                return EXIT_FAILURE;
            }
        }

        std::set<uint16_t> expected = expected_topk_indices(&original[(uint32_t)r * cols], cols, topk_array->num_topk_columns);
        if (actual_indices != expected) {
            std::fprintf(stderr, "topk indices mismatch on row %u\n", (unsigned)r);
            free_topk_array(topk_array);
            return EXIT_FAILURE;
        }
    }

    std::vector<float> decompressed(rows * cols, -123.0f);
    if (topk_decompress(topk_array, decompressed.data()) != 0) {
        std::fprintf(stderr, "topk_decompress failed\n");
        free_topk_array(topk_array);
        return EXIT_FAILURE;
    }

    for (uint16_t r = 0; r < rows; ++r) {
        std::set<uint16_t> selected;
        const uint32_t base = (uint32_t)r * topk_array->num_topk_columns;
        for (uint16_t j = 0; j < topk_array->num_topk_columns; ++j) {
            selected.insert(topk_array->topk_indices[base + j]);
        }

        for (uint16_t c = 0; c < cols; ++c) {
            float v = decompressed[(uint32_t)r * cols + c];
            if (selected.count(c)) {
                if (!almost_equal(v, original[(uint32_t)r * cols + c])) {
                    std::fprintf(stderr, "decompressed selected value mismatch\n");
                    free_topk_array(topk_array);
                    return EXIT_FAILURE;
                }
            } else {
                if (!almost_equal(v, 0.0f)) {
                    std::fprintf(stderr, "decompressed omitted value not zero\n");
                    free_topk_array(topk_array);
                    return EXIT_FAILURE;
                }
            }
        }
    }

    std::vector<float> zero_matrix(rows * cols, 0.0f);
    if (topk_apply(topk_array, zero_matrix.data()) != 0) {
        std::fprintf(stderr, "topk_apply failed\n");
        free_topk_array(topk_array);
        return EXIT_FAILURE;
    }

    for (uint16_t r = 0; r < rows; ++r) {
        std::set<uint16_t> selected;
        const uint32_t base = (uint32_t)r * topk_array->num_topk_columns;
        for (uint16_t j = 0; j < topk_array->num_topk_columns; ++j) {
            uint16_t c = topk_array->topk_indices[base + j];
            selected.insert(c);
            float v = zero_matrix[(uint32_t)r * cols + c];
            if (!almost_equal(v, topk_array->values[base + j])) {
                std::fprintf(stderr, "topk_apply value mismatch\n");
                free_topk_array(topk_array);
                return EXIT_FAILURE;
            }
        }

        for (uint16_t c = 0; c < cols; ++c) {
            if (!selected.count(c) && !almost_equal(zero_matrix[(uint32_t)r * cols + c], 0.0f)) {
                std::fprintf(stderr, "topk_apply wrote non-selected value\n");
                free_topk_array(topk_array);
                return EXIT_FAILURE;
            }
        }
    }

    free_topk_array(topk_array);
    std::printf("topk_extraction and topk_apply test passed\n");
    return EXIT_SUCCESS;
}
