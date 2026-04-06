#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <topk_cuda.cuh>

namespace {

bool almost_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceBuffers {
    float *d_residual = nullptr;
    uint16_t *d_topk_indices = nullptr;
    float *d_topk_values = nullptr;

    ~DeviceBuffers() {
        cudaFree(d_residual);
        cudaFree(d_topk_indices);
        cudaFree(d_topk_values);
    }
};

int run_case(float ratio, uint16_t rows, uint16_t cols, const std::vector<float> &source) {
    TopkCUDAContext ctx;
    DeviceBuffers buffers;

    try {
        const uint16_t k = topk_compute_num_topk_columns(cols, ratio);
        const uint32_t topk_elements = topk_compute_num_topk_elements(rows, k);

        topk_initialize(&ctx, rows, cols, 20260405ULL);
        check_cuda(cudaMalloc(&buffers.d_residual, source.size() * sizeof(float)), "cudaMalloc d_residual failed");
        if (topk_elements > 0) {
            check_cuda(cudaMalloc(&buffers.d_topk_indices, topk_elements * sizeof(uint16_t)),
                       "cudaMalloc d_topk_indices failed");
            check_cuda(cudaMalloc(&buffers.d_topk_values, topk_elements * sizeof(float)),
                       "cudaMalloc d_topk_values failed");
        }
        check_cuda(cudaMemcpyAsync(
                       buffers.d_residual,
                       source.data(),
                       source.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpyAsync residual H2D failed");

        topk_array_t topk_array{};
        topk_bind_array(&topk_array, rows, cols, k, buffers.d_topk_indices, buffers.d_topk_values);
        topk_separation(&ctx, buffers.d_residual, &topk_array);

        std::vector<float> residual(source.size(), 0.0f);
        check_cuda(cudaMemcpyAsync(
                       residual.data(),
                       buffers.d_residual,
                       residual.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync residual D2H failed");

        std::vector<uint16_t> topk_indices(topk_elements, 0);
        if (topk_elements > 0) {
            check_cuda(cudaMemcpyAsync(
                           topk_indices.data(),
                           buffers.d_topk_indices,
                           topk_indices.size() * sizeof(uint16_t),
                           cudaMemcpyDeviceToHost),
                       "cudaMemcpyAsync topk indices D2H failed");
        }

        for (uint16_t r = 0; r < rows; ++r) {
            std::set<uint16_t> selected;
            const uint32_t base = static_cast<uint32_t>(r) * k;

            for (uint16_t j = 0; j < k; ++j) {
                const uint16_t c = topk_indices[base + j];
                selected.insert(c);
                if (!almost_equal(residual[static_cast<uint32_t>(r) * cols + c], 0.0f)) {
                    std::fprintf(stderr, "selected position not zeroed by separation\n");
                    topk_release(&ctx);
                    return 1;
                }
            }

            for (uint16_t c = 0; c < cols; ++c) {
                if (!selected.count(c)) {
                    const float expected = source[static_cast<uint32_t>(r) * cols + c];
                    const float actual = residual[static_cast<uint32_t>(r) * cols + c];
                    if (!almost_equal(expected, actual)) {
                        std::fprintf(stderr, "non-selected position changed by separation\n");
                        topk_release(&ctx);
                        return 1;
                    }
                }
            }
        }

        topk_apply(&ctx, &topk_array, buffers.d_residual);
        std::vector<float> restored(source.size(), 0.0f);
        check_cuda(cudaMemcpyAsync(
                       restored.data(),
                       buffers.d_residual,
                       restored.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync restored D2H failed");

        for (size_t i = 0; i < source.size(); ++i) {
            if (!almost_equal(restored[i], source[i])) {
                std::fprintf(stderr, "reconstruction mismatch at %zu\n", i);
                topk_release(&ctx);
                return 1;
            }
        }

        topk_release(&ctx);
        return 0;
    } catch (const std::exception &e) {
        topk_release(&ctx);
        std::fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    }
}

} // namespace

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
