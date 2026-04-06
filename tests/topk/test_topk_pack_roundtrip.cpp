#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <topk.cuh>

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
    float *d_input = nullptr;
    float *d_dense = nullptr;
    uint16_t *d_indices = nullptr;
    float *d_values = nullptr;
    uint16_t *d_indices_unpacked = nullptr;
    float *d_values_unpacked = nullptr;

    ~DeviceBuffers() {
        cudaFree(d_input);
        cudaFree(d_dense);
        cudaFree(d_indices);
        cudaFree(d_values);
        cudaFree(d_indices_unpacked);
        cudaFree(d_values_unpacked);
    }
};

} // namespace

int main(void) {
    const uint16_t rows = 4;
    const uint16_t cols = 5;
    const float ratio = 0.4f;
    const uint16_t k = topk_compute_num_topk_columns(cols, ratio);
    const uint32_t topk_elements = topk_compute_num_topk_elements(rows, k);
    const std::vector<float> input = {
        1.0f, -5.0f, 0.2f, 7.0f, -3.0f,
        2.0f, 4.0f, -8.0f, 0.5f, 1.5f,
        -9.0f, 0.1f, 3.0f, 2.0f, -4.0f,
        6.0f, -7.0f, 8.0f, -1.0f, 0.3f,
    };

    TopkCUDAContext ctx;
    DeviceBuffers buffers;

    try {
        topk_initialize(&ctx, rows, cols, 20260405ULL);
        check_cuda(cudaMalloc(&buffers.d_input, input.size() * sizeof(float)), "cudaMalloc d_input failed");
        check_cuda(cudaMalloc(&buffers.d_dense, input.size() * sizeof(float)), "cudaMalloc d_dense failed");
        check_cuda(cudaMalloc(&buffers.d_indices, topk_elements * sizeof(uint16_t)), "cudaMalloc d_indices failed");
        check_cuda(cudaMalloc(&buffers.d_values, topk_elements * sizeof(float)), "cudaMalloc d_values failed");
        check_cuda(cudaMalloc(&buffers.d_indices_unpacked, topk_elements * sizeof(uint16_t)), "cudaMalloc d_indices_unpacked failed");
        check_cuda(cudaMalloc(&buffers.d_values_unpacked, topk_elements * sizeof(float)), "cudaMalloc d_values_unpacked failed");
        check_cuda(cudaMemcpy(buffers.d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy input H2D failed");

        topk_array_t topk_array{};
        topk_bind_array(&topk_array, rows, cols, k, buffers.d_indices, buffers.d_values);
        topk_extraction(&ctx, buffers.d_input, &topk_array);

        std::vector<uint8_t> packed(topk_get_packed_size(&topk_array), 0);
        if (topk_pack_to_buffer(&topk_array, packed.data(), packed.size()) != 0) {
            std::fprintf(stderr, "topk_pack_to_buffer failed\n");
            topk_release(&ctx);
            return EXIT_FAILURE;
        }

        topk_packed_header_t header{};
        if (topk_validate_packed_buffer(packed.data(), packed.size(), &header) != 0) {
            std::fprintf(stderr, "topk_validate_packed_buffer failed\n");
            topk_release(&ctx);
            return EXIT_FAILURE;
        }
        if (header.num_rows != rows || header.num_columns != cols || header.num_topk_columns != k) {
            std::fprintf(stderr, "packed header mismatch\n");
            topk_release(&ctx);
            return EXIT_FAILURE;
        }

        topk_array_t unpacked{};
        topk_bind_array(&unpacked, rows, cols, k, buffers.d_indices_unpacked, buffers.d_values_unpacked);
        if (topk_unpack_from_buffer(packed.data(), packed.size(), &unpacked) != 0) {
            std::fprintf(stderr, "topk_unpack_from_buffer failed\n");
            topk_release(&ctx);
            return EXIT_FAILURE;
        }

        topk_decompress(&ctx, &unpacked, buffers.d_dense);
        std::vector<float> dense(input.size(), 0.0f);
        check_cuda(cudaMemcpy(dense.data(), buffers.d_dense, dense.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy dense D2H failed");

        for (uint16_t r = 0; r < rows; ++r) {
            int nonzero_count = 0;
            for (uint16_t c = 0; c < cols; ++c) {
                const float value = dense[static_cast<uint32_t>(r) * cols + c];
                if (!almost_equal(value, 0.0f)) {
                    ++nonzero_count;
                    if (!almost_equal(value, input[static_cast<uint32_t>(r) * cols + c])) {
                        std::fprintf(stderr, "roundtrip value mismatch\n");
                        topk_release(&ctx);
                        return EXIT_FAILURE;
                    }
                }
            }
            if (nonzero_count != static_cast<int>(k)) {
                std::fprintf(stderr, "unexpected nonzero count after roundtrip\n");
                topk_release(&ctx);
                return EXIT_FAILURE;
            }
        }

        topk_release(&ctx);
        std::printf("topk pack roundtrip test passed\n");
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        topk_release(&ctx);
        std::fprintf(stderr, "Fatal error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
