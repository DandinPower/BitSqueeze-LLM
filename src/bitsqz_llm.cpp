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
#include <vector>

#include <reconstruct_lowrank_cuda/reconstruct_lowrank_cuda.hpp>
#include <svd_lowrank_cuda/svd_lowrank_cuda.hpp>
#include <topk.hpp>

namespace {

enum : uint32_t {
    BITSQZ_FLAG_HAS_SVD = 1u << 0,
    BITSQZ_FLAG_HAS_OUTLIER_TOPK = 1u << 1,
    BITSQZ_FLAG_HAS_ERROR_TOPK = 1u << 2,
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
    quantization_method_t svd_uv_format = quantization_INVALID;
    quantization_method_t svd_s_format = quantization_INVALID;
    quantization_method_t quantization_only_format = quantization_INVALID;
    ReconstructLowrankCUDAContext reconstruct_ctx{};
};

BitsqzRuntimeConfig g_bitsqz_runtime;

struct TempResources {
    topk_array_t *outlier_topk = nullptr;
    topk_array_t *error_topk = nullptr;
    quantization_buffer_t *q_u = nullptr;
    quantization_buffer_t *q_s = nullptr;
    quantization_buffer_t *q_v = nullptr;
    quantization_buffer_t *q_direct = nullptr;

    ~TempResources() {
        free_topk_array(outlier_topk);
        free_topk_array(error_topk);
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

static uint64_t push_blob(std::vector<uint8_t> *payload, const void *src, uint64_t size) {
    if (!payload || !src || size == 0) return 0;
    const uint64_t offset = static_cast<uint64_t>(payload->size());
    const size_t old_size = payload->size();
    payload->resize(old_size + static_cast<size_t>(size));
    std::memcpy(payload->data() + old_size, src, static_cast<size_t>(size));
    return offset;
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

static int reconstruct_low_rank_cuda_with_context(
    ReconstructLowrankCUDAContext *ctx,
    const std::vector<float> &U,
    const std::vector<float> &S,
    const std::vector<float> &V,
    uint16_t rows,
    uint16_t cols,
    uint16_t rank,
    std::vector<float> *out) {
    if (!ctx || !out) return 1;
    if (rank == 0) return 1;
    if (ctx->m != static_cast<int>(rows) || ctx->n != static_cast<int>(cols)) return 1;
    if (U.size() != static_cast<size_t>(rows) * rank) return 1;
    if (S.size() != static_cast<size_t>(rank)) return 1;
    if (V.size() != static_cast<size_t>(cols) * rank) return 1;

    try {
        if (ctx->k == static_cast<int>(rank)) {
            reconstruct_lowrank_cuda(ctx, U, S, V, out);
            return 0;
        }
        if (ctx->k < static_cast<int>(rank)) {
            return 1;
        }

        const size_t rows_sz = static_cast<size_t>(rows);
        const size_t cols_sz = static_cast<size_t>(cols);
        const size_t rank_sz = static_cast<size_t>(rank);
        const size_t padded_rank_sz = static_cast<size_t>(ctx->k);

        std::vector<float> U_padded(rows_sz * padded_rank_sz, 0.0f);
        std::vector<float> S_padded(padded_rank_sz, 0.0f);
        std::vector<float> V_padded(cols_sz * padded_rank_sz, 0.0f);

        for (size_t r = 0; r < rows_sz; ++r) {
            std::memcpy(U_padded.data() + r * padded_rank_sz, U.data() + r * rank_sz, rank_sz * sizeof(float));
        }
        std::memcpy(S_padded.data(), S.data(), rank_sz * sizeof(float));
        for (size_t c = 0; c < cols_sz; ++c) {
            std::memcpy(V_padded.data() + c * padded_rank_sz, V.data() + c * rank_sz, rank_sz * sizeof(float));
        }

        reconstruct_lowrank_cuda(ctx, U_padded, S_padded, V_padded, out);
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
    g_bitsqz_runtime.svd_uv_format = svd_uv_format;
    g_bitsqz_runtime.svd_s_format = svd_s_format;
    g_bitsqz_runtime.quantization_only_format = quantization_only_format;

    if (svd_ranks >= 1) {
        const int rank_limit = std::min<int>(num_rows, num_columns);
        const int q = std::max(1, std::min(svd_ranks, rank_limit));
        try {
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
        } catch (const std::exception &) {
            bitsqz_llm_release();
            return 1;
        }
    }

    g_bitsqz_runtime.initialized = true;
    return 0;
}

void bitsqz_llm_release() {
    if (g_bitsqz_runtime.svd_initialized) {
        svd_lowrank_cuda_release();
    }
    if (g_bitsqz_runtime.reconstruct_initialized) {
        reconstruct_lowrank_cuda_release(&g_bitsqz_runtime.reconstruct_ctx);
    }
    g_bitsqz_runtime = BitsqzRuntimeConfig{};
}

int bitsqz_llm_compress(
    const float *row_major_matrix_float_data,
    bitsqz_llm_array_t **out,
    bitsqz_llm_compress_profile_t *profile) {
    if (!g_bitsqz_runtime.initialized) return 1;
    if (!row_major_matrix_float_data || !out || *out) return 1;

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

    const BitsqzRuntimeConfig &cfg = g_bitsqz_runtime;
    const uint16_t num_rows = cfg.num_rows;
    const uint16_t num_columns = cfg.num_columns;
    const float outlier_topk_ratio = cfg.outlier_topk_ratio;
    const float error_correction_topk_ratio = cfg.error_correction_topk_ratio;
    const int svd_ranks = cfg.svd_ranks;
    const quantization_method_t svd_uv_format = cfg.svd_uv_format;
    const quantization_method_t svd_s_format = cfg.svd_s_format;
    const quantization_method_t quantization_only_format = cfg.quantization_only_format;

    const uint32_t num_elements = static_cast<uint32_t>(num_rows) * num_columns;
    std::vector<float> residual(row_major_matrix_float_data, row_major_matrix_float_data + num_elements);
    TempResources temp;

    uint32_t flags = 0;
    if (outlier_topk_ratio > 0.0f) {
        const auto step_begin = Clock::now();
        if (topk_separation(residual.data(), num_rows, num_columns, outlier_topk_ratio, &temp.outlier_topk) != 0) {
            return 1;
        }
        topk_separation_latency_ms += elapsed_ms(step_begin, Clock::now());
        if (temp.outlier_topk && temp.outlier_topk->num_topk_columns > 0) {
            flags |= BITSQZ_FLAG_HAS_OUTLIER_TOPK;
        }
    }

    bitsqz_section_t section_u{};
    bitsqz_section_t section_s{};
    bitsqz_section_t section_v{};
    bitsqz_section_t section_direct{};
    bitsqz_section_t section_outlier{};
    bitsqz_section_t section_error{};

    std::vector<float> error_matrix;
    std::vector<float> u_fp32;
    std::vector<float> s_fp32;
    std::vector<float> v_fp32;
    uint16_t effective_rank = 0;

    const bool has_svd = (svd_ranks >= 1);
    if (has_svd) {
        flags |= BITSQZ_FLAG_HAS_SVD;
        const int rank_limit = std::min<int>(num_rows, num_columns);
        const int expected_q = std::max(1, std::min(svd_ranks, rank_limit));

        try {
            const auto step_begin = Clock::now();
            const SVDLowrankCPUResult &svd = svd_lowrank_cuda(residual.data(), 1241ULL);
            svd_lowrank_cuda_latency_ms += elapsed_ms(step_begin, Clock::now());

            if (svd.k <= 0) return 1;
            if (svd.k > expected_q) return 1;
            effective_rank = static_cast<uint16_t>(svd.k);

            u_fp32 = svd.U_row_major;
            s_fp32 = svd.S;
            v_fp32 = svd.V_row_major;
        } catch (const std::exception &) {
            return 1;
        }

        if (svd_uv_format != quantization_INVALID) {
            const auto q_begin = Clock::now();
            if (quantization_compress(u_fp32.data(), static_cast<unsigned long long>(u_fp32.size()), svd_uv_format, &temp.q_u) != 0 ||
                !temp.q_u) {
                return 1;
            }
            quantization_compress_latency_ms += elapsed_ms(q_begin, Clock::now());

            std::vector<float> deq_u(u_fp32.size(), 0.0f);
            const auto dq_begin = Clock::now();
            if (quantization_decompress(temp.q_u, deq_u.data(), static_cast<unsigned long long>(deq_u.size())) != 0) return 1;
            reconsturct_quantization_decompress_latency_ms += elapsed_ms(dq_begin, Clock::now());
            u_fp32.swap(deq_u);
            section_u.storage = BITSQZ_SECTION_QUANT;
        } else {
            section_u.storage = BITSQZ_SECTION_FP32;
        }

        if (svd_s_format != quantization_INVALID) {
            const auto q_begin = Clock::now();
            if (quantization_compress(s_fp32.data(), static_cast<unsigned long long>(s_fp32.size()), svd_s_format, &temp.q_s) != 0 ||
                !temp.q_s) {
                return 1;
            }
            quantization_compress_latency_ms += elapsed_ms(q_begin, Clock::now());

            std::vector<float> deq_s(s_fp32.size(), 0.0f);
            const auto dq_begin = Clock::now();
            if (quantization_decompress(temp.q_s, deq_s.data(), static_cast<unsigned long long>(deq_s.size())) != 0) return 1;
            reconsturct_quantization_decompress_latency_ms += elapsed_ms(dq_begin, Clock::now());
            s_fp32.swap(deq_s);
            section_s.storage = BITSQZ_SECTION_QUANT;
        } else {
            section_s.storage = BITSQZ_SECTION_FP32;
        }

        if (svd_uv_format != quantization_INVALID) {
            const auto q_begin = Clock::now();
            if (quantization_compress(v_fp32.data(), static_cast<unsigned long long>(v_fp32.size()), svd_uv_format, &temp.q_v) != 0 ||
                !temp.q_v) {
                return 1;
            }
            quantization_compress_latency_ms += elapsed_ms(q_begin, Clock::now());

            std::vector<float> deq_v(v_fp32.size(), 0.0f);
            const auto dq_begin = Clock::now();
            if (quantization_decompress(temp.q_v, deq_v.data(), static_cast<unsigned long long>(deq_v.size())) != 0) return 1;
            reconsturct_quantization_decompress_latency_ms += elapsed_ms(dq_begin, Clock::now());
            v_fp32.swap(deq_v);
            section_v.storage = BITSQZ_SECTION_QUANT;
        } else {
            section_v.storage = BITSQZ_SECTION_FP32;
        }

        std::vector<float> low_rank;
        const auto step_begin = Clock::now();
        if (reconstruct_low_rank_cuda_with_context(
                &g_bitsqz_runtime.reconstruct_ctx,
                u_fp32,
                s_fp32,
                v_fp32,
                num_rows,
                num_columns,
                effective_rank,
                &low_rank) != 0) {
            return 1;
        }
        reconstruct_svd_latency_ms += elapsed_ms(step_begin, Clock::now());

        error_matrix.resize(num_elements);
#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
        for (uint32_t i = 0; i < num_elements; ++i) {
            error_matrix[i] = residual[i] - low_rank[i];
        }
    } else {
        effective_rank = 0;
        if (quantization_only_format != quantization_INVALID) {
            const auto q_begin = Clock::now();
            if (quantization_compress(residual.data(), num_elements, quantization_only_format, &temp.q_direct) != 0 ||
                !temp.q_direct) {
                return 1;
            }
            quantization_compress_latency_ms += elapsed_ms(q_begin, Clock::now());
            section_direct.storage = BITSQZ_SECTION_QUANT;

            std::vector<float> deq(num_elements, 0.0f);
            const auto dq_begin = Clock::now();
            if (quantization_decompress(temp.q_direct, deq.data(), num_elements) != 0) return 1;
            reconsturct_quantization_decompress_latency_ms += elapsed_ms(dq_begin, Clock::now());
            error_matrix.resize(num_elements);
#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
            for (uint32_t i = 0; i < num_elements; ++i) {
                error_matrix[i] = residual[i] - deq[i];
            }
        } else {
            section_direct.storage = BITSQZ_SECTION_FP32;
            error_matrix.assign(num_elements, 0.0f);
        }
    }

    if (error_correction_topk_ratio > 0.0f) {
        const auto step_begin = Clock::now();
        bool has_nonzero = false;
        for (float v : error_matrix) {
            if (std::fabs(v) > std::numeric_limits<float>::epsilon()) {
                has_nonzero = true;
                break;
            }
        }

        if (has_nonzero && topk_extraction(error_matrix.data(), num_rows, num_columns, error_correction_topk_ratio, &temp.error_topk) != 0) {
            return 1;
        }
        if (temp.error_topk && temp.error_topk->num_topk_columns > 0) {
            flags |= BITSQZ_FLAG_HAS_ERROR_TOPK;
        }
        error_extraction_latency_ms += elapsed_ms(step_begin, Clock::now());
    }

    std::vector<uint8_t> payload;
    payload.reserve(static_cast<size_t>(num_elements) * sizeof(float));

    if (has_svd) {
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
            section_direct.offset = push_blob(&payload, residual.data(), section_direct.size);
        }
    }

    if ((flags & BITSQZ_FLAG_HAS_OUTLIER_TOPK) != 0) {
        section_outlier.storage = BITSQZ_SECTION_TOPK;
        section_outlier.size = get_topk_array_size(temp.outlier_topk);
        section_outlier.offset = push_blob(&payload, temp.outlier_topk, section_outlier.size);
    }

    if ((flags & BITSQZ_FLAG_HAS_ERROR_TOPK) != 0) {
        section_error.storage = BITSQZ_SECTION_TOPK;
        section_error.size = get_topk_array_size(temp.error_topk);
        section_error.offset = push_blob(&payload, temp.error_topk, section_error.size);
    }

    const uint64_t total_size = static_cast<uint64_t>(sizeof(bitsqz_llm_array_t)) + payload.size();
    bitsqz_llm_array_t *buf = static_cast<bitsqz_llm_array_t *>(std::calloc(1, static_cast<size_t>(total_size)));
    if (!buf) return 1;

    buf->num_rows = num_rows;
    buf->num_columns = num_columns;
    buf->num_elements = num_elements;
    buf->outlier_topk_ratio = outlier_topk_ratio;
    buf->error_correction_topk_ratio = error_correction_topk_ratio;
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
}

int bitsqz_llm_decompress(const bitsqz_llm_array_t *compressed, float *dst, uint32_t dst_num_elements) {
    if (!compressed || !dst || !compressed->payload) return 1;
    if (dst_num_elements < compressed->num_elements) return 1;

    std::vector<float> restored(compressed->num_elements, 0.0f);

    if ((compressed->flags & BITSQZ_FLAG_HAS_SVD) != 0) {
        const uint16_t rank = static_cast<uint16_t>(compressed->effective_rank);
        if (rank == 0) return 1;

        std::vector<float> U;
        std::vector<float> S;
        std::vector<float> V;

        if (decode_section_to_fp32(compressed, compressed->section_u, static_cast<uint32_t>(compressed->num_rows) * rank, &U) != 0) {
            return 1;
        }
        if (decode_section_to_fp32(compressed, compressed->section_s, rank, &S) != 0) return 1;
        if (decode_section_to_fp32(compressed, compressed->section_v, static_cast<uint32_t>(compressed->num_columns) * rank, &V) != 0) {
            return 1;
        }

        ReconstructLowrankCUDAContext reconstruct_ctx;
        try {
            reconstruct_lowrank_cuda_initialize(
                &reconstruct_ctx,
                static_cast<int>(compressed->num_rows),
                static_cast<int>(compressed->num_columns),
                static_cast<int>(rank),
                1234ULL);
        } catch (const std::exception &) {
            return 1;
        }
        const int rc = reconstruct_low_rank_cuda_with_context(
            &reconstruct_ctx, U, S, V, compressed->num_rows, compressed->num_columns, rank, &restored);
        reconstruct_lowrank_cuda_release(&reconstruct_ctx);
        if (rc != 0) return 1;
    } else {
        if (decode_section_to_fp32(compressed, compressed->section_direct, compressed->num_elements, &restored) != 0) {
            return 1;
        }
    }

    if ((compressed->flags & BITSQZ_FLAG_HAS_ERROR_TOPK) != 0) {
        const uint8_t *payload = static_cast<const uint8_t *>(compressed->payload);
        const bitsqz_section_t &sec = compressed->section_error_topk;
        topk_array_t *arr = load_topk_array_from_buffer(payload + sec.offset, sec.size);
        if (!arr) return 1;
        const int rc = topk_apply(arr, restored.data());
        free_topk_array(arr);
        if (rc != 0) return 1;
    }

    if ((compressed->flags & BITSQZ_FLAG_HAS_OUTLIER_TOPK) != 0) {
        const uint8_t *payload = static_cast<const uint8_t *>(compressed->payload);
        const bitsqz_section_t &sec = compressed->section_outlier_topk;
        topk_array_t *arr = load_topk_array_from_buffer(payload + sec.offset, sec.size);
        if (!arr) return 1;
        const int rc = topk_apply(arr, restored.data());
        free_topk_array(arr);
        if (rc != 0) return 1;
    }

    std::memcpy(dst, restored.data(), static_cast<size_t>(compressed->num_elements) * sizeof(float));
    return 0;
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
