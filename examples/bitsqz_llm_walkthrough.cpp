#include <bitsqz_llm.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

double mean_absolute_error(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    double total = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        total += std::fabs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]));
    }
    return total / static_cast<double>(lhs.size());
}

void release_runtimes() {
    bitsqz_llm_compress_release();
    bitsqz_llm_decompress_release();
}

} // namespace

int main() {
    constexpr uint16_t rows = 8;
    constexpr uint16_t cols = 8;

    std::vector<float> host_matrix(rows * cols);
    for (std::size_t i = 0; i < host_matrix.size(); ++i) {
        host_matrix[i] = static_cast<float>((i % cols) * 0.25f + (i / cols) * 0.5f);
    }

    float *d_restored = nullptr;
    bitsqz_llm_array_t *compressed = nullptr;

    try {
        check_cuda(cudaMalloc(&d_restored, host_matrix.size() * sizeof(float)), "cudaMalloc d_restored failed");

        if (bitsqz_llm_compress_initialize(
                rows,
                cols,
                0.0f,
                0.0f,
                4,
                2,
                NF4,
                NF4,
                quantization_NONE,
                BITSQZ_LLM_INPUT_HOST) != 0) {
            std::fprintf(stderr, "bitsqz_llm_compress_initialize failed\n");
            return 1;
        }

        if (bitsqz_llm_decompress_initialize(
                rows,
                cols,
                0.0f,
                0.0f,
                // Decompressor SVD size must exactly match the packed SVD rank.
                4,
                NF4,
                NF4,
                quantization_NONE) != 0) {
            std::fprintf(stderr, "bitsqz_llm_decompress_initialize failed\n");
            bitsqz_llm_compress_release();
            return 1;
        }

        bitsqz_llm_compress_profile_t profile{};
        if (bitsqz_llm_compress(host_matrix.data(), &compressed, &profile) != 0 || compressed == nullptr) {
            std::fprintf(stderr, "bitsqz_llm_compress failed\n");
            release_runtimes();
            return 1;
        }

        if (bitsqz_llm_decompress(compressed, d_restored, static_cast<uint32_t>(host_matrix.size())) != 0) {
            std::fprintf(stderr, "bitsqz_llm_decompress failed\n");
            bitsqz_llm_free(compressed);
            release_runtimes();
            return 1;
        }

        std::vector<float> restored(host_matrix.size(), 0.0f);
        check_cuda(
            cudaMemcpy(
                restored.data(),
                d_restored,
                restored.size() * sizeof(float),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy device->host failed");

        std::printf("rows=%u cols=%u\n", rows, cols);
        std::printf("compress_input=host\n");
        std::printf("packed_size=%llu bytes\n", static_cast<unsigned long long>(bitsqz_llm_get_packed_size(compressed)));
        std::printf("effective_rank=%d\n", static_cast<int>(compressed->effective_rank));
        std::printf("mae=%f\n", mean_absolute_error(host_matrix, restored));
        std::printf("compress_ms=%f\n", profile.quantization_compress_latency_ms + profile.reconstruct_svd_latency_ms);

        bitsqz_llm_free(compressed);
        release_runtimes();
        cudaFree(d_restored);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "example failed: %s\n", e.what());
        bitsqz_llm_free(compressed);
        release_runtimes();
        cudaFree(d_restored);
        return 1;
    }

    return 0;
}
