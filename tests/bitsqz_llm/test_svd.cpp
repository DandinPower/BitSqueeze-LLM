#include <bitsqz_llm.hpp>

#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

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
             float error_ratio,
             quantization_method_t uv_format,
             quantization_method_t s_format,
             int rank,
             int niters
            ) {
    using Clock = std::chrono::steady_clock;
    auto elapsed_ms = [](const Clock::time_point &begin, const Clock::time_point &end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    const auto init_begin = Clock::now();
    if (bitsqz_llm_initialize(rows,
                              cols,
                              outlier_ratio,
                              error_ratio,
                              rank,
                              niters,
                              uv_format,
                              s_format,
                              quantization_INVALID) != 0) {
        std::fprintf(stderr, "%s: initialize failed\n", name);
        return 1;
    }
    const double initialize_latency_ms = elapsed_ms(init_begin, Clock::now());

    bitsqz_llm_array_t *compressed = nullptr;
    bitsqz_llm_compress_profile_t compress_profile{};
    const auto compress_begin = Clock::now();
    if (bitsqz_llm_compress(source.data(), &compressed, &compress_profile) != 0 ||
        !compressed) {
        std::fprintf(stderr, "%s: compress failed\n", name);
        bitsqz_llm_release();
        return 1;
    }
    const double compress_latency_ms = elapsed_ms(compress_begin, Clock::now());

    std::vector<float> restored(source.size(), 0.0f);
    const auto decompress_begin = Clock::now();
    if (bitsqz_llm_decompress(compressed, restored.data(), static_cast<uint32_t>(restored.size())) != 0) {
        std::fprintf(stderr, "%s: decompress failed\n", name);
        bitsqz_llm_free(compressed);
        bitsqz_llm_release();
        return 1;
    }
    const double decompress_latency_ms = elapsed_ms(decompress_begin, Clock::now());

    if (compressed->effective_rank <= 0) {
        std::fprintf(stderr, "%s: invalid effective rank\n", name);
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

    std::vector<float> restored_after_load(source.size(), 0.0f);
    const auto decompress_reload_begin = Clock::now();
    if (bitsqz_llm_decompress(loaded, restored_after_load.data(), static_cast<uint32_t>(restored_after_load.size())) != 0) {
        std::fprintf(stderr, "%s: decompress after load failed\n", name);
        bitsqz_llm_free(loaded);
        bitsqz_llm_free(compressed);
        bitsqz_llm_release();
        return 1;
    }
    const double decompress_after_load_latency_ms = elapsed_ms(decompress_reload_begin, Clock::now());

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
            source[idx] =
                std::sin(0.013f * static_cast<float>(idx)) +
                std::cos(0.051f * static_cast<float>(r + c)) +
                0.05f * std::sin(0.17f * static_cast<float>(r * c + 1));
        }
    }

    if (run_case("svd_only", source, rows, cols, 0.0f, 0.0f, quantization_INVALID, quantization_INVALID, 128, 2) != 0) {
        return EXIT_FAILURE;
    }
    if (run_case("svd_quantization", source, rows, cols, 0.0f, 0.0f, Q8_0, Q8_0, 128, 2) != 0) return EXIT_FAILURE;
    if (run_case("svd_error", source, rows, cols, 0.0f, 0.10f, quantization_INVALID, quantization_INVALID, 128, 2) != 0) {
        return EXIT_FAILURE;
    }
    if (run_case("svd_quantization_error", source, rows, cols, 0.0f, 0.10f, Q8_0, Q8_0, 128, 2) != 0) return EXIT_FAILURE;
    if (run_case("outlier_svd", source, rows, cols, 0.05f, 0.0f, quantization_INVALID, quantization_INVALID, 128, 2) != 0) {
        return EXIT_FAILURE;
    }
    if (run_case("outlier_svd_error", source, rows, cols, 0.05f, 0.10f, quantization_INVALID, quantization_INVALID, 128, 2) != 0) {
        return EXIT_FAILURE;
    }
    if (run_case("outlier_svd_quantization_error", source, rows, cols, 0.05f, 0.10f, Q8_0, Q8_0, 128, 2) != 0) {
        return EXIT_FAILURE;
    }
    if (run_case("outlier_svd_quantization_error_best", source, rows, cols, 0.001f, 0.015f, NF4_DQ, quantization_INVALID, 128, 2) != 0) {
        return EXIT_FAILURE;
    }
    if (run_case("outlier_svd_quantization_error_best", source, rows, cols, 0.001f, 0.015f, NF4_DQ, quantization_INVALID, 128, 6) != 0) {
        return EXIT_FAILURE;
    }

    std::printf("bitsqz_llm svd test passed\n");
    return EXIT_SUCCESS;
}
