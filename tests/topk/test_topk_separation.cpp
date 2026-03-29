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

static int run_case(float ratio, uint16_t rows, uint16_t cols, const std::vector<float> &source) {
    std::vector<float> residual = source;
    topk_array_t *topk_array = NULL;

    if (topk_separation(residual.data(), rows, cols, ratio, &topk_array) != 0 || !topk_array) {
        std::fprintf(stderr, "topk_separation failed\n");
        return 1;
    }

    if (topk_array->num_rows != rows || topk_array->num_columns != cols) {
        std::fprintf(stderr, "shape mismatch in topk_array\n");
        free_topk_array(topk_array);
        return 1;
    }

    for (uint16_t r = 0; r < rows; ++r) {
        std::set<uint16_t> selected;
        uint32_t base = (uint32_t)r * topk_array->num_topk_columns;

        for (uint16_t j = 0; j < topk_array->num_topk_columns; ++j) {
            uint16_t c = topk_array->topk_indices[base + j];
            selected.insert(c);
            float v = residual[(uint32_t)r * cols + c];
            if (!almost_equal(v, 0.0f)) {
                std::fprintf(stderr, "selected position not zeroed by separation\n");
                free_topk_array(topk_array);
                return 1;
            }
        }

        for (uint16_t c = 0; c < cols; ++c) {
            if (!selected.count(c)) {
                float expected = source[(uint32_t)r * cols + c];
                float actual = residual[(uint32_t)r * cols + c];
                if (!almost_equal(expected, actual)) {
                    std::fprintf(stderr, "non-selected position changed by separation\n");
                    free_topk_array(topk_array);
                    return 1;
                }
            }
        }
    }

    if (topk_apply(topk_array, residual.data()) != 0) {
        std::fprintf(stderr, "topk_apply failed\n");
        free_topk_array(topk_array);
        return 1;
    }

    for (size_t i = 0; i < source.size(); ++i) {
        if (!almost_equal(residual[i], source[i])) {
            std::fprintf(stderr, "reconstruction mismatch at %zu\n", i);
            free_topk_array(topk_array);
            return 1;
        }
    }

    free_topk_array(topk_array);
    return 0;
}

int main(void) {
    const uint16_t rows = 2;
    const uint16_t cols = 8;
    const std::vector<float> source = {
        1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f,
        -1.1f, 2.2f, -3.3f, 4.4f, -5.5f, 6.6f, -7.7f, 8.8f,
    };

    if (run_case(0.0f, rows, cols, source) != 0) return EXIT_FAILURE;
    if (run_case(0.01f, rows, cols, source) != 0) return EXIT_FAILURE;
    if (run_case(1.0f, rows, cols, source) != 0) return EXIT_FAILURE;

    std::printf("topk_separation test passed\n");
    return EXIT_SUCCESS;
}
