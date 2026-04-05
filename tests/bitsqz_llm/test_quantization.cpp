#include <bitsqz_llm.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceBuffers {
    float *d_source = nullptr;
    float *d_restored = nullptr;

    ~DeviceBuffers() {
        cudaFree(d_source);
        cudaFree(d_restored);
    }
};

bool almost_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

void measure_metrics(const std::vector<float> &orig,
                     const std::vector<float> &deq,
                     double *mae,
                     double *mse) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < orig.size(); ++i) {
        const double diff = static_cast<double>(deq[i]) - static_cast<double>(orig[i]);
        sum_abs += std::fabs(diff);
        sum_sq += diff * diff;
    }
    *mae = sum_abs / static_cast<double>(orig.size());
    *mse = sum_sq / static_cast<double>(orig.size());
}

int run_case(const char *name,
             const std::vector<float> &source,
             uint16_t rows,
             uint16_t cols,
             float outlier_ratio,
             float error_ratio) {
    using Clock = std::chrono::steady_clock;
    auto elapsed_ms = [](const Clock::time_point &begin, const Clock::time_point &end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    DeviceBuffers buffers;
    try {
        check_cuda(cudaMalloc(&buffers.d_source, source.size() * sizeof(float)), "cudaMalloc d_source failed");
        check_cuda(cudaMalloc(&buffers.d_restored, source.size() * sizeof(float)), "cudaMalloc d_restored failed");
        check_cuda(cudaMemcpy(
                       buffers.d_source,
                       source.data(),
                       source.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpy source H2D failed");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s: device setup failed: %s\n", name, e.what());
        return 1;
    }

    const auto init_begin = Clock::now();
    if (bitsqz_llm_initialize(rows,
                              cols,
                              outlier_ratio,
                              error_ratio,
                              -1,
                              2,
                              quantization_INVALID,
                              quantization_INVALID,
                              Q8_0) != 0) {
        std::fprintf(stderr, "%s: initialize failed\n", name);
        return 1;
    }
    const double initialize_latency_ms = elapsed_ms(init_begin, Clock::now());

    bitsqz_llm_array_t *compressed = nullptr;
    bitsqz_llm_compress_profile_t compress_profile{};
    const auto compress_begin = Clock::now();
    if (bitsqz_llm_compress(buffers.d_source, &compressed, &compress_profile) != 0 || !compressed) {
        std::fprintf(stderr, "%s: compress failed\n", name);
        bitsqz_llm_release();
        return 1;
    }
    const double compress_latency_ms = elapsed_ms(compress_begin, Clock::now());

    bitsqz_llm_release();
    std::vector<float> restored(source.size(), 0.0f);
    const auto decompress_begin = Clock::now();
    if (bitsqz_llm_decompress(compressed, buffers.d_restored, static_cast<uint32_t>(restored.size())) == 0) {
        std::fprintf(stderr, "%s: decompress should fail without initialize\n", name);
        bitsqz_llm_free(compressed);
        return 1;
    }
    if (bitsqz_llm_initialize(rows,
                              cols,
                              outlier_ratio,
                              error_ratio,
                              -1,
                              2,
                              quantization_INVALID,
                              quantization_INVALID,
                              Q8_0) != 0) {
        std::fprintf(stderr, "%s: re-initialize before decompress failed\n", name);
        bitsqz_llm_free(compressed);
        return 1;
    }
    if (bitsqz_llm_decompress(compressed, buffers.d_restored, static_cast<uint32_t>(restored.size())) != 0) {
        std::fprintf(stderr, "%s: decompress failed\n", name);
        bitsqz_llm_free(compressed);
        bitsqz_llm_release();
        return 1;
    }
    try {
        check_cuda(cudaMemcpy(
                       restored.data(),
                       buffers.d_restored,
                       restored.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy restored D2H failed");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s: restore copy failed: %s\n", name, e.what());
        bitsqz_llm_free(compressed);
        bitsqz_llm_release();
        return 1;
    }
    const double decompress_latency_ms = elapsed_ms(decompress_begin, Clock::now());

    if (compressed->num_rows != rows || compressed->num_columns != cols || compressed->num_elements != source.size()) {
        std::fprintf(stderr, "%s: shape metadata mismatch\n", name);
        bitsqz_llm_free(compressed);
        bitsqz_llm_release();
        return 1;
    }

    double mae = 0.0;
    double mse = 0.0;
    measure_metrics(source, restored, &mae, &mse);

    const uint64_t packed = bitsqz_llm_get_packed_size(compressed);
    if (packed <= sizeof(bitsqz_llm_array_t)) {
        std::fprintf(stderr, "%s: packed size is invalid\n", name);
        bitsqz_llm_free(compressed);
        bitsqz_llm_release();
        return 1;
    }
    const uint64_t baseline =
        static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) * sizeof(float);
    const double normalized_bits_per_value =
        static_cast<double>(packed) * 32.0 / static_cast<double>(baseline);

    bitsqz_llm_array_t *loaded = load_bitsqz_llm_from_buffer(compressed, packed);
    if (!loaded) {
        std::fprintf(stderr, "%s: load_bitsqz_llm_from_buffer failed\n", name);
        bitsqz_llm_free(compressed);
        bitsqz_llm_release();
        return 1;
    }

    bitsqz_llm_release();
    std::vector<float> restored_after_load(source.size(), 0.0f);
    const auto decompress_reload_begin = Clock::now();
    if (bitsqz_llm_initialize(rows,
                              cols,
                              outlier_ratio,
                              error_ratio,
                              -1,
                              2,
                              quantization_INVALID,
                              quantization_INVALID,
                              Q8_0) != 0) {
        std::fprintf(stderr, "%s: re-initialize before reload decompress failed\n", name);
        bitsqz_llm_free(loaded);
        bitsqz_llm_free(compressed);
        return 1;
    }
    if (bitsqz_llm_decompress(loaded, buffers.d_restored, static_cast<uint32_t>(restored_after_load.size())) != 0) {
        std::fprintf(stderr, "%s: decompress after load failed\n", name);
        bitsqz_llm_free(loaded);
        bitsqz_llm_free(compressed);
        bitsqz_llm_release();
        return 1;
    }
    try {
        check_cuda(cudaMemcpy(
                       restored_after_load.data(),
                       buffers.d_restored,
                       restored_after_load.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy restored_after_load D2H failed");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s: reload restore copy failed: %s\n", name, e.what());
        bitsqz_llm_free(loaded);
        bitsqz_llm_free(compressed);
        bitsqz_llm_release();
        return 1;
    }
    const double decompress_after_load_latency_ms = elapsed_ms(decompress_reload_begin, Clock::now());

    for (size_t i = 0; i < restored.size(); ++i) {
        if (!almost_equal(restored[i], restored_after_load[i], 1e-5f)) {
            std::fprintf(stderr, "%s: reload mismatch at %zu\n", name, i);
            bitsqz_llm_free(loaded);
            bitsqz_llm_free(compressed);
            bitsqz_llm_release();
            return 1;
        }
    }

    std::printf(
        "[%s] packed=%llu bytes, baseline=%llu bytes, norm=%.6f bits/value, MAE=%.6f, MSE=%.6f, init=%.3f ms, compress=%.3f ms, decompress=%.3f ms, decompress_reload=%.3f ms\n"
        "  compress_profile(ms): topk=%.3f, svd=%.3f, q_compress=%.3f, reconstruct_q_decompress=%.3f, reconstruct_svd=%.3f, error_extract=%.3f, other=%.3f\n",
                name,
                static_cast<unsigned long long>(packed),
                static_cast<unsigned long long>(baseline),
                normalized_bits_per_value,
                mae,
                mse,
                initialize_latency_ms,
                compress_latency_ms,
                decompress_latency_ms,
                decompress_after_load_latency_ms,
                compress_profile.topk_separation_latency_ms,
                compress_profile.svd_lowrank_cuda_latency_ms,
                compress_profile.quantization_compress_latency_ms,
                compress_profile.reconsturct_quantization_decompress_latency_ms,
                compress_profile.reconstruct_svd_latency_ms,
                compress_profile.error_extraction_latency_ms,
                compress_profile.other_latency_ms);

    bitsqz_llm_free(loaded);
    bitsqz_llm_free(compressed);
    bitsqz_llm_release();
    return 0;
}

} // namespace

int main(void) {
    const uint16_t rows = 512;
    const uint16_t cols = 8192;
    std::vector<float> source(static_cast<size_t>(rows) * cols, 0.0f);

    for (uint16_t r = 0; r < rows; ++r) {
        for (uint16_t c = 0; c < cols; ++c) {
            const size_t idx = static_cast<size_t>(r) * cols + c;
            source[idx] = std::sin(0.11f * static_cast<float>(idx)) + std::cos(0.07f * static_cast<float>(r + c));
        }
    }

    if (run_case("quantization_only", source, rows, cols, 0.0f, 0.0f) != 0) return EXIT_FAILURE;
    if (run_case("quantization_plus_error", source, rows, cols, 0.0f, 0.10f) != 0) return EXIT_FAILURE;
    if (run_case("outlier_plus_quantization", source, rows, cols, 0.05f, 0.0f) != 0) return EXIT_FAILURE;
    if (run_case("outlier_quantization_error", source, rows, cols, 0.05f, 0.10f) != 0) return EXIT_FAILURE;

    std::printf("bitsqz_llm no-svd test passed\n");
    return EXIT_SUCCESS;
}
