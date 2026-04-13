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
#include <topk_cuda.cuh>

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

void release_runtimes() {
    bitsqz_llm_compress_release();
    bitsqz_llm_decompress_release();
}

bool almost_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

float smaller_topk_ratio(uint16_t cols, float ratio) {
    if (ratio <= 0.0f) {
        return 0.0f;
    }

    const uint16_t current_topk_columns = topk_compute_num_topk_columns(cols, ratio);
    if (current_topk_columns <= 1) {
        return 0.0f;
    }

    const float numerator = static_cast<float>(current_topk_columns - 1) - 0.49f;
    return numerator > 0.0f ? numerator / static_cast<float>(cols) : 0.0f;
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

const float *select_compress_input(const std::vector<float> &source,
                                   const DeviceBuffers &buffers,
                                   bitsqz_llm_input_location_t input_location) {
    return input_location == BITSQZ_LLM_INPUT_HOST ? source.data() : buffers.d_source;
}

int initialize_compressor(uint16_t rows,
                          uint16_t cols,
                          float outlier_ratio,
                          float error_ratio,
                          quantization_method_t direct_format,
                          bitsqz_llm_input_location_t input_location) {
    return bitsqz_llm_compress_initialize(
        rows,
        cols,
        outlier_ratio,
        error_ratio,
        -1,
        2,
        quantization_NONE,
        quantization_NONE,
        direct_format,
        input_location);
}

int initialize_decompressor(uint16_t rows,
                            uint16_t cols,
                            float outlier_ratio,
                            float error_ratio,
                            quantization_method_t direct_format) {
    return bitsqz_llm_decompress_initialize(
        rows,
        cols,
        outlier_ratio,
        error_ratio,
        0,
        quantization_NONE,
        quantization_NONE,
        direct_format);
}

int initialize_both(uint16_t rows,
                    uint16_t cols,
                    float outlier_ratio,
                    float error_ratio,
                    quantization_method_t direct_format,
                    bitsqz_llm_input_location_t input_location) {
    if (initialize_compressor(rows, cols, outlier_ratio, error_ratio, direct_format, input_location) != 0) {
        return 1;
    }
    if (initialize_decompressor(rows, cols, outlier_ratio, error_ratio, direct_format) != 0) {
        bitsqz_llm_compress_release();
        return 1;
    }
    return 0;
}

int run_lifecycle_checks(const char *name,
                         const std::vector<float> &source,
                         uint16_t rows,
                         uint16_t cols,
                         float outlier_ratio,
                         float error_ratio,
                         quantization_method_t direct_format,
                         bitsqz_llm_input_location_t input_location) {
    release_runtimes();

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

    const float *compress_input = select_compress_input(source, buffers, input_location);
    bitsqz_llm_array_t dummy{};
    dummy.payload = reinterpret_cast<void *>(1);

    bitsqz_llm_array_t *compressed = nullptr;
    if (bitsqz_llm_compress(compress_input, &compressed, nullptr) == 0) {
        std::fprintf(stderr, "%s: compress should fail before compressor init\n", name);
        bitsqz_llm_free(compressed);
        return 1;
    }
    if (bitsqz_llm_decompress(&dummy, buffers.d_restored, static_cast<uint32_t>(source.size())) == 0) {
        std::fprintf(stderr, "%s: decompress should fail before decompressor init\n", name);
        return 1;
    }

    if (initialize_both(rows, cols, outlier_ratio, error_ratio, direct_format, input_location) != 0) {
        std::fprintf(stderr, "%s: initialize_both failed\n", name);
        return 1;
    }

    if (bitsqz_llm_compress(compress_input, &compressed, nullptr) != 0 || compressed == nullptr) {
        std::fprintf(stderr, "%s: compress failed after init\n", name);
        release_runtimes();
        return 1;
    }
    if (bitsqz_llm_decompress(compressed, buffers.d_restored, static_cast<uint32_t>(source.size())) != 0) {
        std::fprintf(stderr, "%s: decompress failed after init\n", name);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }

    bitsqz_llm_compress_release();
    bitsqz_llm_array_t *compressed_after_decompress_only = nullptr;
    if (bitsqz_llm_compress(compress_input, &compressed_after_decompress_only, nullptr) == 0) {
        std::fprintf(stderr, "%s: compress should fail after compressor release\n", name);
        bitsqz_llm_free(compressed_after_decompress_only);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }
    if (bitsqz_llm_decompress(compressed, buffers.d_restored, static_cast<uint32_t>(source.size())) != 0) {
        std::fprintf(stderr, "%s: decompressor should survive compressor release\n", name);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }

    if (initialize_compressor(rows, cols, outlier_ratio, error_ratio, direct_format, input_location) != 0) {
        std::fprintf(stderr, "%s: compressor reinitialize failed\n", name);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }

    bitsqz_llm_decompress_release();
    if (bitsqz_llm_decompress(compressed, buffers.d_restored, static_cast<uint32_t>(source.size())) == 0) {
        std::fprintf(stderr, "%s: decompress should fail after decompressor release\n", name);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }

    bitsqz_llm_array_t *compressed_after_compressor_only = nullptr;
    if (bitsqz_llm_compress(compress_input, &compressed_after_compressor_only, nullptr) != 0 ||
        compressed_after_compressor_only == nullptr) {
        std::fprintf(stderr, "%s: compressor should survive decompressor release\n", name);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }

    bitsqz_llm_free(compressed_after_compressor_only);
    bitsqz_llm_free(compressed);
    release_runtimes();
    return 0;
}

int run_case(const char *name,
             const std::vector<float> &source,
             uint16_t rows,
             uint16_t cols,
             float outlier_ratio,
             float error_ratio,
             quantization_method_t direct_format,
             bitsqz_llm_input_location_t input_location = BITSQZ_LLM_INPUT_DEVICE) {
    using Clock = std::chrono::steady_clock;
    auto elapsed_ms = [](const Clock::time_point &begin, const Clock::time_point &end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    release_runtimes();

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

    const float *compress_input = select_compress_input(source, buffers, input_location);

    const auto init_begin = Clock::now();
    if (initialize_both(rows, cols, outlier_ratio, error_ratio, direct_format, input_location) != 0) {
        std::fprintf(stderr, "%s: initialize_both failed\n", name);
        return 1;
    }
    const double initialize_latency_ms = elapsed_ms(init_begin, Clock::now());

    bitsqz_llm_array_t *compressed = nullptr;
    bitsqz_llm_compress_profile_t compress_profile{};
    const auto compress_begin = Clock::now();
    if (bitsqz_llm_compress(compress_input, &compressed, &compress_profile) != 0 || compressed == nullptr) {
        std::fprintf(stderr, "%s: compress failed\n", name);
        release_runtimes();
        return 1;
    }
    const double compress_latency_ms = elapsed_ms(compress_begin, Clock::now());

    std::vector<float> restored(source.size(), 0.0f);
    const auto decompress_begin = Clock::now();
    if (bitsqz_llm_decompress(compressed, buffers.d_restored, static_cast<uint32_t>(restored.size())) != 0) {
        std::fprintf(stderr, "%s: hot-path decompress failed\n", name);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }
    const double decompress_latency_ms = elapsed_ms(decompress_begin, Clock::now());

    const auto restore_copy_begin = Clock::now();
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
        release_runtimes();
        return 1;
    }
    const double restore_copy_latency_ms = elapsed_ms(restore_copy_begin, Clock::now());

    if (compressed->num_rows != rows || compressed->num_columns != cols || compressed->num_elements != source.size()) {
        std::fprintf(stderr, "%s: shape metadata mismatch\n", name);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }

    double mae = 0.0;
    double mse = 0.0;
    measure_metrics(source, restored, &mae, &mse);

    const uint64_t packed = bitsqz_llm_get_packed_size(compressed);
    if (packed <= sizeof(bitsqz_llm_array_t)) {
        std::fprintf(stderr, "%s: packed size is invalid\n", name);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }
    const uint64_t baseline =
        static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) * sizeof(float);
    const double normalized_bits_per_value =
        static_cast<double>(packed) * 32.0 / static_cast<double>(baseline);

    const auto load_begin = Clock::now();
    bitsqz_llm_array_t *loaded = load_bitsqz_llm_from_buffer(compressed, packed);
    const double load_latency_ms = elapsed_ms(load_begin, Clock::now());
    if (!loaded) {
        std::fprintf(stderr, "%s: load_bitsqz_llm_from_buffer failed\n", name);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }

    std::vector<float> restored_after_load(source.size(), 0.0f);
    const auto decompress_reload_begin = Clock::now();
    if (bitsqz_llm_decompress(loaded, buffers.d_restored, static_cast<uint32_t>(restored_after_load.size())) != 0) {
        std::fprintf(stderr, "%s: decompress after load failed\n", name);
        bitsqz_llm_free(loaded);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }
    const double decompress_after_load_latency_ms = elapsed_ms(decompress_reload_begin, Clock::now());

    const auto reload_restore_copy_begin = Clock::now();
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
        release_runtimes();
        return 1;
    }
    const double reload_restore_copy_latency_ms = elapsed_ms(reload_restore_copy_begin, Clock::now());

    for (size_t i = 0; i < restored.size(); ++i) {
        if (!almost_equal(restored[i], restored_after_load[i], 1e-5f)) {
            std::fprintf(stderr, "%s: reload mismatch at %zu\n", name, i);
            bitsqz_llm_free(loaded);
            bitsqz_llm_free(compressed);
            release_runtimes();
            return 1;
        }
    }

    release_runtimes();
    if (bitsqz_llm_decompress(compressed, buffers.d_restored, static_cast<uint32_t>(restored.size())) == 0) {
        std::fprintf(stderr, "%s: decompress should fail after decompressor release\n", name);
        bitsqz_llm_free(loaded);
        bitsqz_llm_free(compressed);
        return 1;
    }

    if (initialize_decompressor(rows, static_cast<uint16_t>(cols - 1), outlier_ratio, error_ratio, direct_format) != 0) {
        std::fprintf(stderr, "%s: incompatible shape decompressor initialize failed\n", name);
        bitsqz_llm_free(loaded);
        bitsqz_llm_free(compressed);
        return 1;
    }
    if (bitsqz_llm_decompress(compressed, buffers.d_restored, static_cast<uint32_t>(restored.size())) == 0) {
        std::fprintf(stderr, "%s: decompress should fail for shape mismatch\n", name);
        bitsqz_llm_free(loaded);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }

    if (initialize_decompressor(rows, cols, outlier_ratio, error_ratio, quantization_NONE) != 0) {
        std::fprintf(stderr, "%s: missing-quantization decompressor initialize failed\n", name);
        bitsqz_llm_free(loaded);
        bitsqz_llm_free(compressed);
        return 1;
    }
    if (bitsqz_llm_decompress(compressed, buffers.d_restored, static_cast<uint32_t>(restored.size())) == 0) {
        std::fprintf(stderr, "%s: decompress should fail without quantization runtime\n", name);
        bitsqz_llm_free(loaded);
        bitsqz_llm_free(compressed);
        release_runtimes();
        return 1;
    }

    const float smaller_outlier_ratio = smaller_topk_ratio(cols, outlier_ratio);
    const float smaller_error_ratio = smaller_topk_ratio(cols, error_ratio);
    if (smaller_outlier_ratio != outlier_ratio || smaller_error_ratio != error_ratio) {
        if (initialize_decompressor(rows, cols, smaller_outlier_ratio, smaller_error_ratio, direct_format) != 0) {
            std::fprintf(stderr, "%s: reduced-topk-capacity decompressor initialize failed\n", name);
            bitsqz_llm_free(loaded);
            bitsqz_llm_free(compressed);
            return 1;
        }
        if (bitsqz_llm_decompress(compressed, buffers.d_restored, static_cast<uint32_t>(restored.size())) == 0) {
            std::fprintf(stderr, "%s: decompress should fail with insufficient topk capacity\n", name);
            bitsqz_llm_free(loaded);
            bitsqz_llm_free(compressed);
            release_runtimes();
            return 1;
        }
    }

    std::printf(
        "[%s] input=%s, packed=%llu bytes, baseline=%llu bytes, norm=%.6f bits/value, MAE=%.6f, MSE=%.6f, init=%.3f ms, compress=%.3f ms, decompress_only=%.3f ms, restore_copy=%.3f ms, load=%.3f ms, decompress_reload_only=%.3f ms, reload_restore_copy=%.3f ms\n"
        "  compress_profile(ms): topk=%.3f, svd=%.3f, q_compress=%.3f, reconstruct_q_decompress=%.3f, reconstruct_svd=%.3f, error_extract=%.3f, other=%.3f\n",
                name,
                input_location == BITSQZ_LLM_INPUT_HOST ? "host" : "device",
                static_cast<unsigned long long>(packed),
                static_cast<unsigned long long>(baseline),
                normalized_bits_per_value,
                mae,
                mse,
                initialize_latency_ms,
                compress_latency_ms,
                decompress_latency_ms,
                restore_copy_latency_ms,
                load_latency_ms,
                decompress_after_load_latency_ms,
                reload_restore_copy_latency_ms,
                compress_profile.topk_separation_latency_ms,
                compress_profile.svd_lowrank_cuda_latency_ms,
                compress_profile.quantization_compress_latency_ms,
                compress_profile.reconsturct_quantization_decompress_latency_ms,
                compress_profile.reconstruct_svd_latency_ms,
                compress_profile.error_extraction_latency_ms,
                compress_profile.other_latency_ms);

    bitsqz_llm_free(loaded);
    bitsqz_llm_free(compressed);
    release_runtimes();
    return 0;
}

} // namespace

int main(void) {
    const uint16_t rows = 512;
    const uint16_t cols = 8192;

    if (bitsqz_llm_compress_initialize(
            rows,
            cols,
            0.0f,
            0.0f,
            -1,
            2,
            quantization_NONE,
            quantization_NONE,
            Q8_0,
            BITSQZ_LLM_INPUT_DEVICE) == 0) {
        std::fprintf(stderr, "bitsqz_llm_compress_initialize should reject Q8_0\n");
        bitsqz_llm_compress_release();
        return EXIT_FAILURE;
    }

    std::vector<float> source(static_cast<size_t>(rows) * cols, 0.0f);

    for (uint16_t r = 0; r < rows; ++r) {
        for (uint16_t c = 0; c < cols; ++c) {
            const size_t idx = static_cast<size_t>(r) * cols + c;
            source[idx] = std::sin(0.11f * static_cast<float>(idx)) + std::cos(0.07f * static_cast<float>(r + c));
        }
    }

    const uint16_t lifecycle_rows = 128;
    const uint16_t lifecycle_cols = 512;
    std::vector<float> lifecycle_source(static_cast<size_t>(lifecycle_rows) * lifecycle_cols, 0.0f);
    for (uint16_t r = 0; r < lifecycle_rows; ++r) {
        for (uint16_t c = 0; c < lifecycle_cols; ++c) {
            const size_t idx = static_cast<size_t>(r) * lifecycle_cols + c;
            lifecycle_source[idx] = std::sin(0.09f * static_cast<float>(idx)) + std::cos(0.05f * static_cast<float>(r + c));
        }
    }

    if (run_lifecycle_checks(
            "quantization_lifecycle",
            lifecycle_source,
            lifecycle_rows,
            lifecycle_cols,
            0.02f,
            0.03f,
            NF4,
            BITSQZ_LLM_INPUT_DEVICE) != 0) {
        return EXIT_FAILURE;
    }

    if (run_case("quantization_only_nf4", source, rows, cols, 0.0f, 0.0f, NF4) != 0) return EXIT_FAILURE;
    if (run_case("quantization_only_nf4_host", source, rows, cols, 0.0f, 0.0f, NF4, BITSQZ_LLM_INPUT_HOST) != 0) return EXIT_FAILURE;
    if (run_case("quantization_plus_error_nf4", source, rows, cols, 0.0f, 0.10f, NF4) != 0) return EXIT_FAILURE;
    if (run_case("outlier_plus_quantization_nf4_dq", source, rows, cols, 0.05f, 0.0f, NF4_DQ) != 0) return EXIT_FAILURE;
    if (run_case("outlier_quantization_error_nf4_dq", source, rows, cols, 0.05f, 0.10f, NF4_DQ) != 0) return EXIT_FAILURE;

    std::printf("bitsqz_llm no-svd test passed\n");
    return EXIT_SUCCESS;
}
