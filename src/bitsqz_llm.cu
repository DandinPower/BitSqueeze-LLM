#include <bitsqz_llm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <reconstruct_lowrank_cuda/reconstruct_lowrank_cuda.cuh>
#include <svd_lowrank_cuda/svd_lowrank_cuda.cuh>
#include <topk.cuh>

namespace {

enum : uint32_t {
    BITSQZ_FLAG_HAS_SVD = 1u << 0,
    BITSQZ_FLAG_HAS_OUTLIER_TOPK = 1u << 1,
    BITSQZ_FLAG_HAS_ERROR_TOPK = 1u << 2,
};

constexpr int kBlockSize = 256;

struct BitsqzCUDADeviceBuffers {
    float *d_residual = nullptr;
    float *d_u = nullptr;
    float *d_s = nullptr;
    float *d_v = nullptr;
    float *d_reconstructed = nullptr;
    float *d_error = nullptr;
    int *d_has_nonzero = nullptr;

    void free_all() noexcept {
        if (d_residual) cudaFree(d_residual);
        if (d_u) cudaFree(d_u);
        if (d_s) cudaFree(d_s);
        if (d_v) cudaFree(d_v);
        if (d_reconstructed) cudaFree(d_reconstructed);
        if (d_error) cudaFree(d_error);
        if (d_has_nonzero) cudaFree(d_has_nonzero);

        d_residual = nullptr;
        d_u = nullptr;
        d_s = nullptr;
        d_v = nullptr;
        d_reconstructed = nullptr;
        d_error = nullptr;
        d_has_nonzero = nullptr;
    }
};

struct BitsqzTopkRuntime {
    bool initialized = false;
    uint16_t capacity_topk_columns = 0;
    uint16_t *d_topk_indices = nullptr;
    float *d_values = nullptr;
    topk_array_t array{};
    TopkCUDAContext ctx{};

    void reset_array(uint16_t num_rows, uint16_t num_columns) {
        if (!initialized) {
            return;
        }
        topk_bind_array(
            &array,
            num_rows,
            num_columns,
            capacity_topk_columns,
            d_topk_indices,
            d_values);
    }

    void release() noexcept {
        if (initialized) {
            topk_release(&ctx);
        }
        if (d_topk_indices) cudaFree(d_topk_indices);
        if (d_values) cudaFree(d_values);

        initialized = false;
        capacity_topk_columns = 0;
        d_topk_indices = nullptr;
        d_values = nullptr;
        array = topk_array_t{};
    }
};

struct BitsqzRuntimeConfig {
    bool initialized = false;
    bool svd_initialized = false;
    bool reconstruct_initialized = false;

    uint16_t num_rows = 0;
    uint16_t num_columns = 0;
    float outlier_topk_ratio = 0.0f;
    float error_correction_topk_ratio = 0.0f;
    int svd_ranks = -1;
    int svd_niters = 1;
    int svd_rank_capacity = 0;
    quantization_method_t svd_uv_format = quantization_INVALID;
    quantization_method_t svd_s_format = quantization_INVALID;
    quantization_method_t quantization_only_format = quantization_INVALID;
    ReconstructLowrankCUDAContext reconstruct_ctx{};
    BitsqzCUDADeviceBuffers device_buffers{};
    BitsqzTopkRuntime outlier_topk{};
    BitsqzTopkRuntime error_topk{};
};

BitsqzRuntimeConfig g_bitsqz_runtime;

struct TempResources {
    quantization_buffer_t *q_u = nullptr;
    quantization_buffer_t *q_s = nullptr;
    quantization_buffer_t *q_v = nullptr;
    quantization_buffer_t *q_direct = nullptr;

    ~TempResources() {
        quantization_free(q_u);
        quantization_free(q_s);
        quantization_free(q_v);
        quantization_free(q_direct);
    }
};

static bool ratio_is_valid(float ratio) {
    return std::isfinite(ratio) && ratio >= 0.0f && ratio <= 1.0f;
}

static bool quant_method_is_valid(quantization_method_t method) {
    if (method == quantization_INVALID) return true;
    return method >= Q8_0 && method <= Q2_K_FAST;
}

static void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

__global__ void subtract_kernel(
    const float *lhs,
    const float *rhs,
    float *out,
    std::size_t count) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = lhs[idx] - rhs[idx];
    }
}

__global__ void has_nonzero_kernel(
    const float *input,
    std::size_t count,
    float epsilon,
    int *out_flag) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count && fabsf(input[idx]) > epsilon) {
        *out_flag = 1;
    }
}

static void allocate_bitsqz_device_buffers(
    BitsqzCUDADeviceBuffers *buffers,
    uint16_t rows,
    uint16_t cols,
    int rank,
    bool allocate_svd_buffers) {
    if (!buffers) {
        throw std::invalid_argument("buffers must not be null");
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("rows and cols must be positive");
    }
    if (allocate_svd_buffers && rank <= 0) {
        throw std::invalid_argument("rank must be positive when SVD buffers are required");
    }

    const std::size_t matrix_count = static_cast<std::size_t>(rows) * cols;
    buffers->free_all();

    check_cuda(cudaMalloc(&buffers->d_residual, matrix_count * sizeof(float)), "cudaMalloc d_residual failed");
    check_cuda(cudaMalloc(&buffers->d_reconstructed, matrix_count * sizeof(float)), "cudaMalloc d_reconstructed failed");
    check_cuda(cudaMalloc(&buffers->d_error, matrix_count * sizeof(float)), "cudaMalloc d_error failed");
    check_cuda(cudaMalloc(&buffers->d_has_nonzero, sizeof(int)), "cudaMalloc d_has_nonzero failed");

    if (allocate_svd_buffers) {
        const std::size_t u_count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(rank);
        const std::size_t s_count = static_cast<std::size_t>(rank);
        const std::size_t v_count = static_cast<std::size_t>(cols) * static_cast<std::size_t>(rank);

        check_cuda(cudaMalloc(&buffers->d_u, u_count * sizeof(float)), "cudaMalloc d_u failed");
        check_cuda(cudaMalloc(&buffers->d_s, s_count * sizeof(float)), "cudaMalloc d_s failed");
        check_cuda(cudaMalloc(&buffers->d_v, v_count * sizeof(float)), "cudaMalloc d_v failed");
    }
}

static void initialize_topk_runtime(
    BitsqzTopkRuntime *runtime,
    uint16_t rows,
    uint16_t cols,
    float ratio) {
    if (!runtime) {
        throw std::invalid_argument("runtime must not be null");
    }

    runtime->release();
    const uint16_t topk_columns = topk_compute_num_topk_columns(cols, ratio);
    runtime->capacity_topk_columns = topk_columns;
    if (topk_columns == 0) {
        return;
    }

    const uint32_t topk_elements = topk_compute_num_topk_elements(rows, topk_columns);
    check_cuda(cudaMalloc(&runtime->d_topk_indices, static_cast<std::size_t>(topk_elements) * sizeof(uint16_t)),
               "cudaMalloc topk indices failed");
    check_cuda(cudaMalloc(&runtime->d_values, static_cast<std::size_t>(topk_elements) * sizeof(float)),
               "cudaMalloc topk values failed");
    topk_bind_array(
        &runtime->array,
        rows,
        cols,
        topk_columns,
        runtime->d_topk_indices,
        runtime->d_values);
    topk_initialize(&runtime->ctx, rows, cols, 1234ULL);
    runtime->initialized = true;
}

static uint64_t push_blob(std::vector<uint8_t> *payload, const void *src, uint64_t size) {
    if (!payload || !src || size == 0) return 0;
    const uint64_t offset = static_cast<uint64_t>(payload->size());
    const size_t old_size = payload->size();
    payload->resize(old_size + static_cast<size_t>(size));
    std::memcpy(payload->data() + old_size, src, static_cast<size_t>(size));
    return offset;
}

static int append_packed_topk_blob(
    std::vector<uint8_t> *payload,
    const topk_array_t *topk_array,
    bitsqz_section_t *section) {
    if (!payload || !topk_array || !section) return 1;

    section->storage = BITSQZ_SECTION_TOPK;
    section->size = topk_get_packed_size(topk_array);
    if (section->size == 0) return 1;

    section->offset = static_cast<uint64_t>(payload->size());
    payload->resize(payload->size() + static_cast<size_t>(section->size));
    if (topk_pack_to_buffer(
            topk_array,
            payload->data() + static_cast<size_t>(section->offset),
            section->size) != 0) {
        return 1;
    }
    return 0;
}

static int decode_section_to_fp32(
    const bitsqz_llm_array_t *compressed,
    const bitsqz_section_t &section,
    uint32_t expected_count,
    std::vector<float> *out) {
    if (!compressed || !out) return 1;

    out->assign(expected_count, 0.0f);
    if (section.storage == BITSQZ_SECTION_NONE) {
        return expected_count == 0 ? 0 : 1;
    }

    if (!compressed->payload) return 1;
    const uint8_t *payload = static_cast<const uint8_t *>(compressed->payload);
    const uint8_t *ptr = payload + section.offset;

    if (section.storage == BITSQZ_SECTION_FP32) {
        const uint64_t expected_size = static_cast<uint64_t>(expected_count) * sizeof(float);
        if (section.size != expected_size) return 1;
        std::memcpy(out->data(), ptr, static_cast<size_t>(expected_size));
        return 0;
    }

    if (section.storage == BITSQZ_SECTION_QUANT) {
        quantization_buffer_t *q = load_quantization_from_buffer(ptr, static_cast<long long>(section.size));
        if (!q) return 1;
        const int rc = quantization_decompress(q, out->data(), expected_count);
        quantization_free(q);
        return rc;
    }

    return 1;
}

static int validate_section_bounds(const bitsqz_section_t &section, uint64_t payload_size) {
    if (section.storage == BITSQZ_SECTION_NONE) {
        return section.size == 0 ? 0 : 1;
    }
    if (section.size == 0) return 1;
    if (section.offset > payload_size) return 1;
    if (section.size > payload_size - section.offset) return 1;
    if (section.storage != BITSQZ_SECTION_FP32 &&
        section.storage != BITSQZ_SECTION_QUANT &&
        section.storage != BITSQZ_SECTION_TOPK) {
        return 1;
    }
    return 0;
}

static int copy_device_buffer_to_host(
    const float *d_src,
    uint32_t count,
    std::vector<float> *dst) {
    if (!d_src || !dst) return 1;

    try {
        dst->resize(static_cast<size_t>(count));
        check_cuda(cudaMemcpyAsync(
                       dst->data(),
                       d_src,
                       static_cast<size_t>(count) * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync D2H failed");
        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}

static int upload_low_rank_factors_to_device(
    float *d_u,
    float *d_s,
    float *d_v,
    cudaStream_t stream,
    int padded_rank,
    const std::vector<float> &U,
    const std::vector<float> &S,
    const std::vector<float> &V,
    uint16_t rows,
    uint16_t cols,
    uint16_t rank) {
    if (!d_u || !d_s || !d_v) return 1;
    if (rank == 0 || padded_rank < static_cast<int>(rank)) return 1;
    if (U.size() != static_cast<size_t>(rows) * rank) return 1;
    if (S.size() != static_cast<size_t>(rank)) return 1;
    if (V.size() != static_cast<size_t>(cols) * rank) return 1;

    try {
        const size_t rows_sz = static_cast<size_t>(rows);
        const size_t cols_sz = static_cast<size_t>(cols);
        const size_t rank_sz = static_cast<size_t>(rank);
        const size_t padded_rank_sz = static_cast<size_t>(padded_rank);

        const float *u_src = U.data();
        const float *s_src = S.data();
        const float *v_src = V.data();

        std::vector<float> u_padded;
        std::vector<float> s_padded;
        std::vector<float> v_padded;

        if (padded_rank != static_cast<int>(rank)) {
            u_padded.assign(rows_sz * padded_rank_sz, 0.0f);
            s_padded.assign(padded_rank_sz, 0.0f);
            v_padded.assign(cols_sz * padded_rank_sz, 0.0f);

            for (size_t r = 0; r < rows_sz; ++r) {
                std::memcpy(u_padded.data() + r * padded_rank_sz, U.data() + r * rank_sz, rank_sz * sizeof(float));
            }
            std::memcpy(s_padded.data(), S.data(), rank_sz * sizeof(float));
            for (size_t c = 0; c < cols_sz; ++c) {
                std::memcpy(v_padded.data() + c * padded_rank_sz, V.data() + c * rank_sz, rank_sz * sizeof(float));
            }

            u_src = u_padded.data();
            s_src = s_padded.data();
            v_src = v_padded.data();
        }

        check_cuda(
            cudaMemcpyAsync(d_u, u_src, rows_sz * padded_rank_sz * sizeof(float), cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync d_u failed");
        check_cuda(
            cudaMemcpyAsync(d_s, s_src, padded_rank_sz * sizeof(float), cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync d_s failed");
        check_cuda(
            cudaMemcpyAsync(d_v, v_src, cols_sz * padded_rank_sz * sizeof(float), cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync d_v failed");
        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}

static int reconstruct_low_rank_cuda_with_context(
    ReconstructLowrankCUDAContext *ctx,
    BitsqzCUDADeviceBuffers *buffers,
    const std::vector<float> &U,
    const std::vector<float> &S,
    const std::vector<float> &V,
    uint16_t rows,
    uint16_t cols,
    uint16_t rank,
    float *d_out) {
    if (!ctx || !buffers || !d_out) return 1;
    if (rank == 0) return 1;
    if (ctx->m != static_cast<int>(rows) || ctx->n != static_cast<int>(cols)) return 1;
    if (ctx->k < static_cast<int>(rank)) return 1;

    try {
        if (upload_low_rank_factors_to_device(
                buffers->d_u,
                buffers->d_s,
                buffers->d_v,
                ctx->stream,
                ctx->k,
                U,
                S,
                V,
                rows,
                cols,
                rank) != 0) {
            return 1;
        }

        reconstruct_lowrank_cuda(ctx, buffers->d_u, buffers->d_s, buffers->d_v, d_out);
        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}

static int compute_error_buffer(
    const float *d_expected,
    const float *d_approx,
    float *d_error,
    uint32_t count) {
    if (!d_expected || !d_approx || !d_error) return 1;

    try {
        const int grid = static_cast<int>((static_cast<std::size_t>(count) + kBlockSize - 1) / kBlockSize);
        subtract_kernel<<<grid, kBlockSize>>>(
            d_expected,
            d_approx,
            d_error,
            static_cast<std::size_t>(count));
        check_cuda(cudaGetLastError(), "subtract kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after subtract failed");
        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}

static int device_buffer_has_nonzero(
    BitsqzCUDADeviceBuffers *buffers,
    const float *d_input,
    uint32_t count,
    bool *out_has_nonzero) {
    if (!buffers || !buffers->d_has_nonzero || !d_input || !out_has_nonzero) return 1;

    try {
        check_cuda(cudaMemset(buffers->d_has_nonzero, 0, sizeof(int)), "cudaMemset d_has_nonzero failed");
        const int grid = static_cast<int>((static_cast<std::size_t>(count) + kBlockSize - 1) / kBlockSize);
        has_nonzero_kernel<<<grid, kBlockSize>>>(
            d_input,
            static_cast<std::size_t>(count),
            std::numeric_limits<float>::epsilon(),
            buffers->d_has_nonzero);
        check_cuda(cudaGetLastError(), "has_nonzero kernel launch failed");

        int host_flag = 0;
        check_cuda(cudaMemcpyAsync(&host_flag, buffers->d_has_nonzero, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync d_has_nonzero failed");
        *out_has_nonzero = (host_flag != 0);
        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}

} // namespace

int bitsqz_llm_initialize(
    uint16_t num_rows,
    uint16_t num_columns,
    float outlier_topk_ratio,
    float error_correction_topk_ratio,
    int svd_ranks,
    int svd_niters,
    quantization_method_t svd_uv_format,
    quantization_method_t svd_s_format,
    quantization_method_t quantization_only_format) {
    if (num_rows == 0 || num_columns == 0) return 1;
    if (!ratio_is_valid(outlier_topk_ratio) || !ratio_is_valid(error_correction_topk_ratio)) return 1;
    if (!quant_method_is_valid(svd_uv_format) ||
        !quant_method_is_valid(svd_s_format) ||
        !quant_method_is_valid(quantization_only_format)) {
        return 1;
    }
    if (svd_niters < 1) return 1;

    bitsqz_llm_release();

    g_bitsqz_runtime.num_rows = num_rows;
    g_bitsqz_runtime.num_columns = num_columns;
    g_bitsqz_runtime.outlier_topk_ratio = outlier_topk_ratio;
    g_bitsqz_runtime.error_correction_topk_ratio = error_correction_topk_ratio;
    g_bitsqz_runtime.svd_ranks = svd_ranks;
    g_bitsqz_runtime.svd_niters = svd_niters;
    g_bitsqz_runtime.svd_rank_capacity = 0;
    g_bitsqz_runtime.svd_uv_format = svd_uv_format;
    g_bitsqz_runtime.svd_s_format = svd_s_format;
    g_bitsqz_runtime.quantization_only_format = quantization_only_format;

    const bool has_svd = (svd_ranks >= 1);
    const int rank_limit = std::min<int>(num_rows, num_columns);
    const int q = has_svd ? std::max(1, std::min(svd_ranks, rank_limit)) : 0;

    try {
        allocate_bitsqz_device_buffers(
            &g_bitsqz_runtime.device_buffers,
            num_rows,
            num_columns,
            q,
            has_svd);

        if (has_svd) {
            svd_lowrank_cuda_initialize(
                static_cast<int>(num_rows),
                static_cast<int>(num_columns),
                q,
                svd_niters,
                1234ULL);
            g_bitsqz_runtime.svd_initialized = true;

            reconstruct_lowrank_cuda_initialize(
                &g_bitsqz_runtime.reconstruct_ctx,
                static_cast<int>(num_rows),
                static_cast<int>(num_columns),
                q,
                1234ULL);
            g_bitsqz_runtime.reconstruct_initialized = true;
            g_bitsqz_runtime.svd_rank_capacity = q;
        }

        if (outlier_topk_ratio > 0.0f) {
            initialize_topk_runtime(
                &g_bitsqz_runtime.outlier_topk,
                num_rows,
                num_columns,
                outlier_topk_ratio);
        }
        if (error_correction_topk_ratio > 0.0f) {
            initialize_topk_runtime(
                &g_bitsqz_runtime.error_topk,
                num_rows,
                num_columns,
                error_correction_topk_ratio);
        }
    } catch (const std::exception &) {
        bitsqz_llm_release();
        return 1;
    }

    g_bitsqz_runtime.initialized = true;
    return 0;
}

void bitsqz_llm_release() {
    g_bitsqz_runtime.outlier_topk.release();
    g_bitsqz_runtime.error_topk.release();
    g_bitsqz_runtime.device_buffers.free_all();
    if (g_bitsqz_runtime.svd_initialized) {
        svd_lowrank_cuda_release();
    }
    if (g_bitsqz_runtime.reconstruct_initialized) {
        reconstruct_lowrank_cuda_release(&g_bitsqz_runtime.reconstruct_ctx);
    }
    g_bitsqz_runtime = BitsqzRuntimeConfig{};
}

int bitsqz_llm_compress(
    const float *d_row_major_matrix_float_data,
    bitsqz_llm_array_t **out,
    bitsqz_llm_compress_profile_t *profile) {
    if (!g_bitsqz_runtime.initialized) return 1;
    if (!d_row_major_matrix_float_data || !out || *out) return 1;

    if (profile) {
        *profile = bitsqz_llm_compress_profile_t{};
    }

    using Clock = std::chrono::steady_clock;
    const auto total_begin = Clock::now();
    auto elapsed_ms = [](const Clock::time_point &begin, const Clock::time_point &end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    double topk_separation_latency_ms = 0.0;
    double svd_lowrank_cuda_latency_ms = 0.0;
    double quantization_compress_latency_ms = 0.0;
    double reconsturct_quantization_decompress_latency_ms = 0.0;
    double reconstruct_svd_latency_ms = 0.0;
    double error_extraction_latency_ms = 0.0;

    BitsqzRuntimeConfig &cfg = g_bitsqz_runtime;
    const uint16_t num_rows = cfg.num_rows;
    const uint16_t num_columns = cfg.num_columns;
    const uint32_t num_elements = static_cast<uint32_t>(num_rows) * num_columns;
    const std::size_t matrix_bytes = static_cast<std::size_t>(num_elements) * sizeof(float);

    TempResources temp;
    uint32_t flags = 0;
    bitsqz_section_t section_u{};
    bitsqz_section_t section_s{};
    bitsqz_section_t section_v{};
    bitsqz_section_t section_direct{};
    bitsqz_section_t section_outlier{};
    bitsqz_section_t section_error{};

    std::vector<float> residual_fp32;
    std::vector<float> u_fp32;
    std::vector<float> s_fp32;
    std::vector<float> v_fp32;
    uint16_t effective_rank = 0;

    try {
        check_cuda(cudaMemcpyAsync(
                       cfg.device_buffers.d_residual,
                       d_row_major_matrix_float_data,
                       matrix_bytes,
                       cudaMemcpyDeviceToDevice),
                   "cudaMemcpyAsync input D2D failed");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after residual copy failed");

        if (cfg.outlier_topk.initialized) {
            cfg.outlier_topk.reset_array(num_rows, num_columns);
            const auto step_begin = Clock::now();
            topk_separation(
                &cfg.outlier_topk.ctx,
                cfg.device_buffers.d_residual,
                &cfg.outlier_topk.array);
            topk_separation_latency_ms += elapsed_ms(step_begin, Clock::now());
            if (cfg.outlier_topk.array.num_topk_columns > 0) {
                flags |= BITSQZ_FLAG_HAS_OUTLIER_TOPK;
            }
        }

        const bool has_svd = (cfg.svd_ranks >= 1);
        if (has_svd) {
            flags |= BITSQZ_FLAG_HAS_SVD;
            const int rank_limit = std::min<int>(num_rows, num_columns);
            const int expected_q = std::max(1, std::min(cfg.svd_ranks, rank_limit));
            if (cfg.svd_rank_capacity != expected_q) return 1;

            try {
                const auto step_begin = Clock::now();
                u_fp32.resize(static_cast<size_t>(num_rows) * static_cast<size_t>(expected_q));
                s_fp32.resize(static_cast<size_t>(expected_q));
                v_fp32.resize(static_cast<size_t>(num_columns) * static_cast<size_t>(expected_q));

                svd_lowrank_cuda(
                    cfg.device_buffers.d_residual,
                    cfg.device_buffers.d_u,
                    cfg.device_buffers.d_s,
                    cfg.device_buffers.d_v,
                    1241ULL);

                check_cuda(cudaMemcpyAsync(
                               u_fp32.data(),
                               cfg.device_buffers.d_u,
                               u_fp32.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                           "cudaMemcpyAsync d_u to host failed");
                check_cuda(cudaMemcpyAsync(
                               s_fp32.data(),
                               cfg.device_buffers.d_s,
                               s_fp32.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                           "cudaMemcpyAsync d_s to host failed");
                check_cuda(cudaMemcpyAsync(
                               v_fp32.data(),
                               cfg.device_buffers.d_v,
                               v_fp32.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                           "cudaMemcpyAsync d_v to host failed");
                svd_lowrank_cuda_latency_ms += elapsed_ms(step_begin, Clock::now());
                effective_rank = static_cast<uint16_t>(expected_q);
            } catch (const std::exception &) {
                return 1;
            }

            if (cfg.svd_uv_format != quantization_INVALID) {
                const auto q_begin = Clock::now();
                if (quantization_compress(
                        u_fp32.data(),
                        static_cast<unsigned long long>(u_fp32.size()),
                        cfg.svd_uv_format,
                        &temp.q_u) != 0 || !temp.q_u) {
                    return 1;
                }
                quantization_compress_latency_ms += elapsed_ms(q_begin, Clock::now());
                section_u.storage = BITSQZ_SECTION_QUANT;
            } else {
                section_u.storage = BITSQZ_SECTION_FP32;
            }

            if (cfg.svd_s_format != quantization_INVALID) {
                const auto q_begin = Clock::now();
                if (quantization_compress(
                        s_fp32.data(),
                        static_cast<unsigned long long>(s_fp32.size()),
                        cfg.svd_s_format,
                        &temp.q_s) != 0 || !temp.q_s) {
                    return 1;
                }
                quantization_compress_latency_ms += elapsed_ms(q_begin, Clock::now());
                section_s.storage = BITSQZ_SECTION_QUANT;
            } else {
                section_s.storage = BITSQZ_SECTION_FP32;
            }

            if (cfg.svd_uv_format != quantization_INVALID) {
                const auto q_begin = Clock::now();
                if (quantization_compress(
                        v_fp32.data(),
                        static_cast<unsigned long long>(v_fp32.size()),
                        cfg.svd_uv_format,
                        &temp.q_v) != 0 || !temp.q_v) {
                    return 1;
                }
                quantization_compress_latency_ms += elapsed_ms(q_begin, Clock::now());
                section_v.storage = BITSQZ_SECTION_QUANT;
            } else {
                section_v.storage = BITSQZ_SECTION_FP32;
            }

            if (cfg.error_topk.initialized) {
                std::vector<float> reconstructed_u;
                std::vector<float> reconstructed_s;
                std::vector<float> reconstructed_v;
                const std::vector<float> *u_for_reconstruct = &u_fp32;
                const std::vector<float> *s_for_reconstruct = &s_fp32;
                const std::vector<float> *v_for_reconstruct = &v_fp32;

                if (section_u.storage == BITSQZ_SECTION_QUANT) {
                    reconstructed_u.assign(u_fp32.size(), 0.0f);
                    const auto dq_begin = Clock::now();
                    if (quantization_decompress(
                            temp.q_u,
                            reconstructed_u.data(),
                            static_cast<unsigned long long>(reconstructed_u.size())) != 0) {
                        return 1;
                    }
                    reconsturct_quantization_decompress_latency_ms += elapsed_ms(dq_begin, Clock::now());
                    u_for_reconstruct = &reconstructed_u;
                }
                if (section_s.storage == BITSQZ_SECTION_QUANT) {
                    reconstructed_s.assign(s_fp32.size(), 0.0f);
                    const auto dq_begin = Clock::now();
                    if (quantization_decompress(
                            temp.q_s,
                            reconstructed_s.data(),
                            static_cast<unsigned long long>(reconstructed_s.size())) != 0) {
                        return 1;
                    }
                    reconsturct_quantization_decompress_latency_ms += elapsed_ms(dq_begin, Clock::now());
                    s_for_reconstruct = &reconstructed_s;
                }
                if (section_v.storage == BITSQZ_SECTION_QUANT) {
                    reconstructed_v.assign(v_fp32.size(), 0.0f);
                    const auto dq_begin = Clock::now();
                    if (quantization_decompress(
                            temp.q_v,
                            reconstructed_v.data(),
                            static_cast<unsigned long long>(reconstructed_v.size())) != 0) {
                        return 1;
                    }
                    reconsturct_quantization_decompress_latency_ms += elapsed_ms(dq_begin, Clock::now());
                    v_for_reconstruct = &reconstructed_v;
                }

                const auto step_begin = Clock::now();
                if (reconstruct_low_rank_cuda_with_context(
                        &cfg.reconstruct_ctx,
                        &cfg.device_buffers,
                        *u_for_reconstruct,
                        *s_for_reconstruct,
                        *v_for_reconstruct,
                        num_rows,
                        num_columns,
                        effective_rank,
                        cfg.device_buffers.d_reconstructed) != 0) {
                    return 1;
                }
                reconstruct_svd_latency_ms += elapsed_ms(step_begin, Clock::now());

                const auto error_begin = Clock::now();
                if (compute_error_buffer(
                        cfg.device_buffers.d_residual,
                        cfg.device_buffers.d_reconstructed,
                        cfg.device_buffers.d_error,
                        num_elements) != 0) {
                    return 1;
                }

                bool has_nonzero = false;
                if (device_buffer_has_nonzero(
                        &cfg.device_buffers,
                        cfg.device_buffers.d_error,
                        num_elements,
                        &has_nonzero) != 0) {
                    return 1;
                }

                if (has_nonzero) {
                    cfg.error_topk.reset_array(num_rows, num_columns);
                    topk_extraction(
                        &cfg.error_topk.ctx,
                        cfg.device_buffers.d_error,
                        &cfg.error_topk.array);
                    if (cfg.error_topk.array.num_topk_columns > 0) {
                        flags |= BITSQZ_FLAG_HAS_ERROR_TOPK;
                    }
                }
                error_extraction_latency_ms += elapsed_ms(error_begin, Clock::now());
            }
        } else {
            effective_rank = 0;
            if (copy_device_buffer_to_host(
                    cfg.device_buffers.d_residual,
                    num_elements,
                    &residual_fp32) != 0) {
                return 1;
            }

            if (cfg.quantization_only_format != quantization_INVALID) {
                const auto q_begin = Clock::now();
                if (quantization_compress(
                        residual_fp32.data(),
                        num_elements,
                        cfg.quantization_only_format,
                        &temp.q_direct) != 0 || !temp.q_direct) {
                    return 1;
                }
                quantization_compress_latency_ms += elapsed_ms(q_begin, Clock::now());
                section_direct.storage = BITSQZ_SECTION_QUANT;

                if (cfg.error_topk.initialized) {
                    std::vector<float> deq(num_elements, 0.0f);
                    const auto dq_begin = Clock::now();
                    if (quantization_decompress(temp.q_direct, deq.data(), num_elements) != 0) {
                        return 1;
                    }
                    reconsturct_quantization_decompress_latency_ms += elapsed_ms(dq_begin, Clock::now());

                    check_cuda(cudaMemcpyAsync(
                                   cfg.device_buffers.d_reconstructed,
                                   deq.data(),
                                   matrix_bytes,
                                   cudaMemcpyHostToDevice),
                               "cudaMemcpyAsync dequantized direct matrix H2D failed");

                    const auto error_begin = Clock::now();
                    if (compute_error_buffer(
                            cfg.device_buffers.d_residual,
                            cfg.device_buffers.d_reconstructed,
                            cfg.device_buffers.d_error,
                            num_elements) != 0) {
                        return 1;
                    }

                    bool has_nonzero = false;
                    if (device_buffer_has_nonzero(
                            &cfg.device_buffers,
                            cfg.device_buffers.d_error,
                            num_elements,
                            &has_nonzero) != 0) {
                        return 1;
                    }

                    if (has_nonzero) {
                        cfg.error_topk.reset_array(num_rows, num_columns);
                        topk_extraction(
                            &cfg.error_topk.ctx,
                            cfg.device_buffers.d_error,
                            &cfg.error_topk.array);
                        if (cfg.error_topk.array.num_topk_columns > 0) {
                            flags |= BITSQZ_FLAG_HAS_ERROR_TOPK;
                        }
                    }
                    error_extraction_latency_ms += elapsed_ms(error_begin, Clock::now());
                }
            } else {
                section_direct.storage = BITSQZ_SECTION_FP32;
            }
        }

        std::vector<uint8_t> payload;
        payload.reserve(static_cast<size_t>(num_elements) * sizeof(float));

        if ((flags & BITSQZ_FLAG_HAS_SVD) != 0) {
            if (section_u.storage == BITSQZ_SECTION_QUANT) {
                section_u.size = static_cast<uint64_t>(quantization_get_packed_size(temp.q_u));
                section_u.offset = push_blob(&payload, temp.q_u, section_u.size);
            } else {
                section_u.size = static_cast<uint64_t>(u_fp32.size()) * sizeof(float);
                section_u.offset = push_blob(&payload, u_fp32.data(), section_u.size);
            }

            if (section_s.storage == BITSQZ_SECTION_QUANT) {
                section_s.size = static_cast<uint64_t>(quantization_get_packed_size(temp.q_s));
                section_s.offset = push_blob(&payload, temp.q_s, section_s.size);
            } else {
                section_s.size = static_cast<uint64_t>(s_fp32.size()) * sizeof(float);
                section_s.offset = push_blob(&payload, s_fp32.data(), section_s.size);
            }

            if (section_v.storage == BITSQZ_SECTION_QUANT) {
                section_v.size = static_cast<uint64_t>(quantization_get_packed_size(temp.q_v));
                section_v.offset = push_blob(&payload, temp.q_v, section_v.size);
            } else {
                section_v.size = static_cast<uint64_t>(v_fp32.size()) * sizeof(float);
                section_v.offset = push_blob(&payload, v_fp32.data(), section_v.size);
            }
        } else {
            if (section_direct.storage == BITSQZ_SECTION_QUANT) {
                section_direct.size = static_cast<uint64_t>(quantization_get_packed_size(temp.q_direct));
                section_direct.offset = push_blob(&payload, temp.q_direct, section_direct.size);
            } else {
                section_direct.size = static_cast<uint64_t>(num_elements) * sizeof(float);
                section_direct.offset = push_blob(&payload, residual_fp32.data(), section_direct.size);
            }
        }

        if ((flags & BITSQZ_FLAG_HAS_OUTLIER_TOPK) != 0) {
            if (append_packed_topk_blob(&payload, &cfg.outlier_topk.array, &section_outlier) != 0) {
                return 1;
            }
        }

        if ((flags & BITSQZ_FLAG_HAS_ERROR_TOPK) != 0) {
            if (append_packed_topk_blob(&payload, &cfg.error_topk.array, &section_error) != 0) {
                return 1;
            }
        }

        const uint64_t total_size = static_cast<uint64_t>(sizeof(bitsqz_llm_array_t)) + payload.size();
        bitsqz_llm_array_t *buf = static_cast<bitsqz_llm_array_t *>(std::calloc(1, static_cast<size_t>(total_size)));
        if (!buf) return 1;

        buf->num_rows = num_rows;
        buf->num_columns = num_columns;
        buf->num_elements = num_elements;
        buf->outlier_topk_ratio = cfg.outlier_topk_ratio;
        buf->error_correction_topk_ratio = cfg.error_correction_topk_ratio;
        buf->svd_ranks = static_cast<int16_t>(cfg.svd_ranks);
        buf->svd_niters = static_cast<int16_t>(cfg.svd_niters);
        buf->effective_rank = static_cast<int16_t>(effective_rank);
        buf->svd_uv_format = cfg.svd_uv_format;
        buf->svd_s_format = cfg.svd_s_format;
        buf->quantization_only_format = cfg.quantization_only_format;
        buf->flags = flags;
        buf->section_u = section_u;
        buf->section_s = section_s;
        buf->section_v = section_v;
        buf->section_direct = section_direct;
        buf->section_outlier_topk = section_outlier;
        buf->section_error_topk = section_error;
        buf->payload = reinterpret_cast<uint8_t *>(buf) + sizeof(bitsqz_llm_array_t);

        if (!payload.empty()) {
            std::memcpy(buf->payload, payload.data(), payload.size());
        }

        *out = buf;

        if (profile) {
            const double total_latency_ms = elapsed_ms(total_begin, Clock::now());
            const double accounted_latency_ms =
                topk_separation_latency_ms +
                svd_lowrank_cuda_latency_ms +
                quantization_compress_latency_ms +
                reconsturct_quantization_decompress_latency_ms +
                reconstruct_svd_latency_ms +
                error_extraction_latency_ms;

            profile->topk_separation_latency_ms = topk_separation_latency_ms;
            profile->svd_lowrank_cuda_latency_ms = svd_lowrank_cuda_latency_ms;
            profile->quantization_compress_latency_ms = quantization_compress_latency_ms;
            profile->reconsturct_quantization_decompress_latency_ms = reconsturct_quantization_decompress_latency_ms;
            profile->reconstruct_svd_latency_ms = reconstruct_svd_latency_ms;
            profile->error_extraction_latency_ms = error_extraction_latency_ms;
            profile->other_latency_ms = std::max(0.0, total_latency_ms - accounted_latency_ms);
        }

        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}

int bitsqz_llm_decompress(const bitsqz_llm_array_t *compressed, float *d_dst, uint32_t dst_num_elements) {
    if (!g_bitsqz_runtime.initialized) return 1;
    if (!compressed || !d_dst || !compressed->payload) return 1;
    if (dst_num_elements < compressed->num_elements) return 1;

    BitsqzRuntimeConfig &cfg = g_bitsqz_runtime;
    if (cfg.num_rows != compressed->num_rows || cfg.num_columns != compressed->num_columns) {
        return 1;
    }

    try {
        if ((compressed->flags & BITSQZ_FLAG_HAS_SVD) != 0) {
            const uint16_t rank = static_cast<uint16_t>(compressed->effective_rank);
            if (rank == 0) return 1;
            if (!cfg.reconstruct_initialized) return 1;
            if (cfg.svd_rank_capacity != static_cast<int>(rank)) {
                return 1;
            }

            std::vector<float> U;
            std::vector<float> S;
            std::vector<float> V;

            if (decode_section_to_fp32(
                    compressed,
                    compressed->section_u,
                    static_cast<uint32_t>(compressed->num_rows) * rank,
                    &U) != 0) {
                return 1;
            }
            if (decode_section_to_fp32(compressed, compressed->section_s, rank, &S) != 0) return 1;
            if (decode_section_to_fp32(
                    compressed,
                    compressed->section_v,
                    static_cast<uint32_t>(compressed->num_columns) * rank,
                    &V) != 0) {
                return 1;
            }

            if (reconstruct_low_rank_cuda_with_context(
                    &cfg.reconstruct_ctx,
                    &cfg.device_buffers,
                    U,
                    S,
                    V,
                    compressed->num_rows,
                    compressed->num_columns,
                    rank,
                    d_dst) != 0) {
                return 1;
            }
        } else {
            std::vector<float> restored;
            if (decode_section_to_fp32(compressed, compressed->section_direct, compressed->num_elements, &restored) != 0) {
                return 1;
            }
            check_cuda(cudaMemcpyAsync(
                           d_dst,
                           restored.data(),
                           static_cast<size_t>(compressed->num_elements) * sizeof(float),
                           cudaMemcpyHostToDevice),
                       "cudaMemcpyAsync restored direct matrix H2D failed");
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after direct restore copy failed");
        }

        const uint8_t *payload = static_cast<const uint8_t *>(compressed->payload);
        if ((compressed->flags & BITSQZ_FLAG_HAS_ERROR_TOPK) != 0) {
            if (!cfg.error_topk.initialized) return 1;
            cfg.error_topk.reset_array(compressed->num_rows, compressed->num_columns);
            const bitsqz_section_t &sec = compressed->section_error_topk;
            if (topk_unpack_from_buffer(payload + sec.offset, sec.size, &cfg.error_topk.array) != 0) return 1;
            topk_apply(&cfg.error_topk.ctx, &cfg.error_topk.array, d_dst);
        }

        if ((compressed->flags & BITSQZ_FLAG_HAS_OUTLIER_TOPK) != 0) {
            if (!cfg.outlier_topk.initialized) return 1;
            cfg.outlier_topk.reset_array(compressed->num_rows, compressed->num_columns);
            const bitsqz_section_t &sec = compressed->section_outlier_topk;
            if (topk_unpack_from_buffer(payload + sec.offset, sec.size, &cfg.outlier_topk.array) != 0) return 1;
            topk_apply(&cfg.outlier_topk.ctx, &cfg.outlier_topk.array, d_dst);
        }

        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}

uint64_t bitsqz_llm_get_packed_size(const bitsqz_llm_array_t *compressed) {
    if (!compressed) return 0;

    const bitsqz_section_t sections[] = {
        compressed->section_u,
        compressed->section_s,
        compressed->section_v,
        compressed->section_direct,
        compressed->section_outlier_topk,
        compressed->section_error_topk,
    };

    uint64_t payload_size = 0;
    for (const bitsqz_section_t &section : sections) {
        if (section.storage == BITSQZ_SECTION_NONE) continue;
        const uint64_t end = section.offset + section.size;
        if (end > payload_size) payload_size = end;
    }

    return static_cast<uint64_t>(sizeof(bitsqz_llm_array_t)) + payload_size;
}

bitsqz_llm_array_t *load_bitsqz_llm_from_buffer(const void *buffer, uint64_t buffer_size) {
    if (!buffer || buffer_size < sizeof(bitsqz_llm_array_t)) return nullptr;

    bitsqz_llm_array_t *compressed = static_cast<bitsqz_llm_array_t *>(std::calloc(1, static_cast<size_t>(buffer_size)));
    if (!compressed) return nullptr;

    std::memcpy(compressed, buffer, static_cast<size_t>(buffer_size));
    compressed->payload = reinterpret_cast<uint8_t *>(compressed) + sizeof(bitsqz_llm_array_t);

    const uint64_t packed_size = bitsqz_llm_get_packed_size(compressed);
    if (packed_size == 0 || buffer_size < packed_size) {
        std::free(compressed);
        return nullptr;
    }

    const uint64_t payload_size = packed_size - sizeof(bitsqz_llm_array_t);
    const bitsqz_section_t sections[] = {
        compressed->section_u,
        compressed->section_s,
        compressed->section_v,
        compressed->section_direct,
        compressed->section_outlier_topk,
        compressed->section_error_topk,
    };

    for (const bitsqz_section_t &section : sections) {
        if (validate_section_bounds(section, payload_size) != 0) {
            std::free(compressed);
            return nullptr;
        }
    }

    return compressed;
}

void bitsqz_llm_free(bitsqz_llm_array_t *compressed) {
    if (!compressed) return;
    std::free(compressed);
}
