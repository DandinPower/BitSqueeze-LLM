#include <algorithm>
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

#include <topk.cuh>

namespace {

bool almost_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

std::set<uint16_t> expected_topk_indices(const float *row, uint16_t num_columns, uint16_t k) {
    std::vector<uint16_t> idx(num_columns);
    for (uint16_t i = 0; i < num_columns; ++i) idx[i] = i;

    std::sort(idx.begin(), idx.end(), [row](uint16_t a, uint16_t b) {
        return std::fabs(row[a]) > std::fabs(row[b]);
    });

    std::set<uint16_t> out;
    for (uint16_t i = 0; i < k; ++i) out.insert(idx[i]);
    return out;
}

void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceBuffers {
    float *d_input = nullptr;
    float *d_decompressed = nullptr;
    float *d_zero_matrix = nullptr;
    uint16_t *d_topk_indices = nullptr;
    float *d_topk_values = nullptr;
    uint16_t *d_roundtrip_indices = nullptr;
    float *d_roundtrip_values = nullptr;

    ~DeviceBuffers() {
        cudaFree(d_input);
        cudaFree(d_decompressed);
        cudaFree(d_zero_matrix);
        cudaFree(d_topk_indices);
        cudaFree(d_topk_values);
        cudaFree(d_roundtrip_indices);
        cudaFree(d_roundtrip_values);
    }
};

} // namespace

int main(void) {
    const uint16_t rows = 3;
    const uint16_t cols = 6;
    const float ratio = 0.5f;
    const uint16_t k = topk_compute_num_topk_columns(cols, ratio);
    const uint32_t topk_elements = topk_compute_num_topk_elements(rows, k);

    std::vector<float> input = {
        0.4f, -9.3f, 2.1f, -5.2f, 7.8f, -1.0f,
        -6.6f, 1.3f, 8.4f, -2.5f, 0.7f, 3.9f,
        4.2f, -0.8f, 5.1f, -7.7f, 2.6f, 9.9f,
    };
    const std::vector<float> original = input;

    TopkCUDAContext ctx;
    DeviceBuffers buffers;

    try {
        topk_initialize(&ctx, rows, cols, 20260405ULL);

        check_cuda(cudaMalloc(&buffers.d_input, input.size() * sizeof(float)), "cudaMalloc d_input failed");
        check_cuda(cudaMalloc(&buffers.d_decompressed, input.size() * sizeof(float)), "cudaMalloc d_decompressed failed");
        check_cuda(cudaMalloc(&buffers.d_zero_matrix, input.size() * sizeof(float)), "cudaMalloc d_zero_matrix failed");
        check_cuda(cudaMalloc(&buffers.d_topk_indices, topk_elements * sizeof(uint16_t)), "cudaMalloc d_topk_indices failed");
        check_cuda(cudaMalloc(&buffers.d_topk_values, topk_elements * sizeof(float)), "cudaMalloc d_topk_values failed");
        check_cuda(cudaMalloc(&buffers.d_roundtrip_indices, topk_elements * sizeof(uint16_t)), "cudaMalloc d_roundtrip_indices failed");
        check_cuda(cudaMalloc(&buffers.d_roundtrip_values, topk_elements * sizeof(float)), "cudaMalloc d_roundtrip_values failed");

        check_cuda(cudaMemcpy(
                       buffers.d_input,
                       input.data(),
                       input.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpy input H2D failed");

        topk_array_t topk_array{};
        topk_bind_array(&topk_array, rows, cols, k, buffers.d_topk_indices, buffers.d_topk_values);
        topk_extraction(&ctx, buffers.d_input, &topk_array);

        std::vector<float> input_after(input.size(), 0.0f);
        check_cuda(cudaMemcpy(
                       input_after.data(),
                       buffers.d_input,
                       input_after.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy input D2H failed");
        for (size_t i = 0; i < input_after.size(); ++i) {
            if (!almost_equal(input_after[i], original[i])) {
                std::fprintf(stderr, "input mutated by topk_extraction at %zu\n", i);
                topk_release(&ctx);
                return EXIT_FAILURE;
            }
        }

        std::vector<uint16_t> topk_indices(topk_elements, 0);
        std::vector<float> topk_values(topk_elements, 0.0f);
        check_cuda(cudaMemcpy(
                       topk_indices.data(),
                       buffers.d_topk_indices,
                       topk_indices.size() * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy topk indices D2H failed");
        check_cuda(cudaMemcpy(
                       topk_values.data(),
                       buffers.d_topk_values,
                       topk_values.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy topk values D2H failed");

        for (uint16_t r = 0; r < rows; ++r) {
            std::set<uint16_t> actual_indices;
            const uint32_t base = static_cast<uint32_t>(r) * k;
            for (uint16_t j = 0; j < k; ++j) {
                const uint16_t c = topk_indices[base + j];
                if (c >= cols) {
                    std::fprintf(stderr, "invalid topk index\n");
                    topk_release(&ctx);
                    return EXIT_FAILURE;
                }
                if (!actual_indices.insert(c).second) {
                    std::fprintf(stderr, "duplicate topk index in row\n");
                    topk_release(&ctx);
                    return EXIT_FAILURE;
                }

                const float expected_val = original[static_cast<uint32_t>(r) * cols + c];
                if (!almost_equal(expected_val, topk_values[base + j])) {
                    std::fprintf(stderr, "topk value mismatch\n");
                    topk_release(&ctx);
                    return EXIT_FAILURE;
                }
            }

            const std::set<uint16_t> expected = expected_topk_indices(&original[static_cast<uint32_t>(r) * cols], cols, k);
            if (actual_indices != expected) {
                std::fprintf(stderr, "topk indices mismatch on row %u\n", static_cast<unsigned>(r));
                topk_release(&ctx);
                return EXIT_FAILURE;
            }
        }

        topk_decompress(&ctx, &topk_array, buffers.d_decompressed);
        std::vector<float> decompressed(input.size(), -123.0f);
        check_cuda(cudaMemcpy(
                       decompressed.data(),
                       buffers.d_decompressed,
                       decompressed.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy decompressed D2H failed");

        for (uint16_t r = 0; r < rows; ++r) {
            std::set<uint16_t> selected;
            const uint32_t base = static_cast<uint32_t>(r) * k;
            for (uint16_t j = 0; j < k; ++j) {
                selected.insert(topk_indices[base + j]);
            }

            for (uint16_t c = 0; c < cols; ++c) {
                const float v = decompressed[static_cast<uint32_t>(r) * cols + c];
                if (selected.count(c)) {
                    if (!almost_equal(v, original[static_cast<uint32_t>(r) * cols + c])) {
                        std::fprintf(stderr, "decompressed selected value mismatch\n");
                        topk_release(&ctx);
                        return EXIT_FAILURE;
                    }
                } else if (!almost_equal(v, 0.0f)) {
                    std::fprintf(stderr, "decompressed omitted value not zero\n");
                    topk_release(&ctx);
                    return EXIT_FAILURE;
                }
            }
        }

        check_cuda(cudaMemset(buffers.d_zero_matrix, 0, input.size() * sizeof(float)), "cudaMemset d_zero_matrix failed");
        topk_apply(&ctx, &topk_array, buffers.d_zero_matrix);
        std::vector<float> zero_matrix(input.size(), 0.0f);
        check_cuda(cudaMemcpy(
                       zero_matrix.data(),
                       buffers.d_zero_matrix,
                       zero_matrix.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy zero_matrix D2H failed");

        for (uint16_t r = 0; r < rows; ++r) {
            std::set<uint16_t> selected;
            const uint32_t base = static_cast<uint32_t>(r) * k;
            for (uint16_t j = 0; j < k; ++j) {
                const uint16_t c = topk_indices[base + j];
                selected.insert(c);
                if (!almost_equal(zero_matrix[static_cast<uint32_t>(r) * cols + c], topk_values[base + j])) {
                    std::fprintf(stderr, "topk_apply value mismatch\n");
                    topk_release(&ctx);
                    return EXIT_FAILURE;
                }
            }

            for (uint16_t c = 0; c < cols; ++c) {
                if (!selected.count(c) && !almost_equal(zero_matrix[static_cast<uint32_t>(r) * cols + c], 0.0f)) {
                    std::fprintf(stderr, "topk_apply wrote non-selected value\n");
                    topk_release(&ctx);
                    return EXIT_FAILURE;
                }
            }
        }

        std::vector<uint8_t> packed(topk_get_packed_size(&topk_array), 0);
        if (topk_pack_to_buffer(&topk_array, packed.data(), packed.size()) != 0) {
            std::fprintf(stderr, "topk_pack_to_buffer failed\n");
            topk_release(&ctx);
            return EXIT_FAILURE;
        }

        topk_array_t roundtrip_array{};
        topk_bind_array(&roundtrip_array, rows, cols, k, buffers.d_roundtrip_indices, buffers.d_roundtrip_values);
        if (topk_unpack_from_buffer(packed.data(), packed.size(), &roundtrip_array) != 0) {
            std::fprintf(stderr, "topk_unpack_from_buffer failed\n");
            topk_release(&ctx);
            return EXIT_FAILURE;
        }

        topk_decompress(&ctx, &roundtrip_array, buffers.d_decompressed);
        std::vector<float> roundtrip_decompressed(input.size(), 0.0f);
        check_cuda(cudaMemcpy(
                       roundtrip_decompressed.data(),
                       buffers.d_decompressed,
                       roundtrip_decompressed.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy roundtrip D2H failed");
        for (size_t i = 0; i < decompressed.size(); ++i) {
            if (!almost_equal(roundtrip_decompressed[i], decompressed[i])) {
                std::fprintf(stderr, "roundtrip decompress mismatch at %zu\n", i);
                topk_release(&ctx);
                return EXIT_FAILURE;
            }
        }

        topk_release(&ctx);
        std::printf("topk_extraction, topk_decompress, topk_apply, and pack roundtrip test passed\n");
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        topk_release(&ctx);
        std::fprintf(stderr, "Fatal error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
