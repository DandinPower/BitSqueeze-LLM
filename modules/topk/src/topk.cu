#include <topk.cuh>

#include "topk_utils.cuh"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using topk_detail::check_cuda;
using topk_detail::launch_scatter;
using topk_detail::launch_scatter_add;
using topk_detail::launch_zero_selected;
using topk_detail::run_topk_extraction;
using topk_detail::validate_array_descriptor;
using topk_detail::validate_context;
using topk_detail::warmup;

void TopkCUDADevicePtrs::free_all() noexcept {
    if (d_sort_keys_in) cudaFree(d_sort_keys_in);
    if (d_sort_keys_out) cudaFree(d_sort_keys_out);
    if (d_sort_indices_in) cudaFree(d_sort_indices_in);
    if (d_sort_indices_out) cudaFree(d_sort_indices_out);
    if (d_segment_offsets_begin) cudaFree(d_segment_offsets_begin);
    if (d_segment_offsets_end) cudaFree(d_segment_offsets_end);
    if (d_cub_temp_storage) cudaFree(d_cub_temp_storage);

    d_sort_keys_in = nullptr;
    d_sort_keys_out = nullptr;
    d_sort_indices_in = nullptr;
    d_sort_indices_out = nullptr;
    d_segment_offsets_begin = nullptr;
    d_segment_offsets_end = nullptr;
    d_cub_temp_storage = nullptr;
    cub_temp_storage_bytes = 0;
}

void TopkCUDAContext::destroy_all() noexcept {
    cuda_ptrs.free_all();

    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }

    num_rows = 0;
    num_columns = 0;
    initialized = false;
}

uint16_t topk_compute_num_topk_columns(uint16_t num_columns, float topk_ratio) {
    if (num_columns == 0) {
        return 0;
    }
    if (!(topk_ratio >= 0.0f && topk_ratio <= 1.0f)) {
        throw std::invalid_argument("topk_ratio must be in [0, 1]");
    }

    const float raw_topk = static_cast<float>(num_columns) * topk_ratio;
    uint16_t num_topk_columns = static_cast<uint16_t>(roundf(raw_topk));
    if (num_topk_columns > num_columns) {
        num_topk_columns = num_columns;
    } else if (num_topk_columns == 0 && topk_ratio > 0.0f) {
        num_topk_columns = 1;
    }
    return num_topk_columns;
}

uint32_t topk_compute_num_topk_elements(uint16_t num_rows, uint16_t num_topk_columns) {
    return static_cast<uint32_t>(num_rows) * static_cast<uint32_t>(num_topk_columns);
}

void topk_bind_array(
    topk_array_t *topk_array,
    uint16_t num_rows,
    uint16_t num_columns,
    uint16_t num_topk_columns,
    uint16_t *d_topk_indices,
    float *d_values) {
    if (topk_array == nullptr) {
        throw std::invalid_argument("topk_array must not be null");
    }
    topk_array->num_rows = num_rows;
    topk_array->num_columns = num_columns;
    topk_array->num_topk_columns = num_topk_columns;
    topk_array->d_topk_indices = d_topk_indices;
    topk_array->d_values = d_values;
}

void topk_release(TopkCUDAContext *ctx) noexcept {
    if (ctx == nullptr) {
        return;
    }
    ctx->destroy_all();
}

void topk_initialize(
    TopkCUDAContext *ctx,
    uint16_t num_rows,
    uint16_t num_columns,
    unsigned long long warmup_seed) {
    if (ctx == nullptr) {
        throw std::invalid_argument("ctx must not be null");
    }
    if (num_rows == 0 || num_columns == 0) {
        throw std::invalid_argument("num_rows and num_columns must be positive");
    }

    const uint32_t total_elements = static_cast<uint32_t>(num_rows) * static_cast<uint32_t>(num_columns);
    if (total_elements > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("num_rows * num_columns exceeds the supported segmented sort range");
    }

    topk_release(ctx);

    ctx->num_rows = num_rows;
    ctx->num_columns = num_columns;

    try {
        check_cuda(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed");

        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_sort_keys_in, static_cast<std::size_t>(total_elements) * sizeof(float)),
                   "cudaMalloc d_sort_keys_in failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_sort_keys_out, static_cast<std::size_t>(total_elements) * sizeof(float)),
                   "cudaMalloc d_sort_keys_out failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_sort_indices_in, static_cast<std::size_t>(total_elements) * sizeof(uint16_t)),
                   "cudaMalloc d_sort_indices_in failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_sort_indices_out, static_cast<std::size_t>(total_elements) * sizeof(uint16_t)),
                   "cudaMalloc d_sort_indices_out failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_segment_offsets_begin, static_cast<std::size_t>(num_rows) * sizeof(int)),
                   "cudaMalloc d_segment_offsets_begin failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_segment_offsets_end, static_cast<std::size_t>(num_rows) * sizeof(int)),
                   "cudaMalloc d_segment_offsets_end failed");

        std::vector<int> segment_offsets_begin(num_rows);
        std::vector<int> segment_offsets_end(num_rows);
        for (uint16_t row = 0; row < num_rows; ++row) {
            const int begin = static_cast<int>(row) * static_cast<int>(num_columns);
            segment_offsets_begin[row] = begin;
            segment_offsets_end[row] = begin + static_cast<int>(num_columns);
        }

        check_cuda(cudaMemcpyAsync(
                       ctx->cuda_ptrs.d_segment_offsets_begin,
                       segment_offsets_begin.data(),
                       segment_offsets_begin.size() * sizeof(int),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpyAsync segment_offsets_begin failed");
        check_cuda(cudaMemcpyAsync(
                       ctx->cuda_ptrs.d_segment_offsets_end,
                       segment_offsets_end.data(),
                       segment_offsets_end.size() * sizeof(int),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpyAsync segment_offsets_end failed");

        check_cuda(
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr,
                ctx->cuda_ptrs.cub_temp_storage_bytes,
                ctx->cuda_ptrs.d_sort_keys_in,
                ctx->cuda_ptrs.d_sort_keys_out,
                ctx->cuda_ptrs.d_sort_indices_in,
                ctx->cuda_ptrs.d_sort_indices_out,
                static_cast<int>(total_elements),
                static_cast<int>(num_rows),
                ctx->cuda_ptrs.d_segment_offsets_begin,
                ctx->cuda_ptrs.d_segment_offsets_end,
                0,
                static_cast<int>(sizeof(float) * 8),
                ctx->stream),
            "cub segmented sort size query failed");

        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_cub_temp_storage, ctx->cuda_ptrs.cub_temp_storage_bytes),
                   "cudaMalloc d_cub_temp_storage failed");

        ctx->initialized = true;
        warmup(ctx, warmup_seed);
    } catch (...) {
        ctx->destroy_all();
        throw;
    }
}

void topk_extraction(
    TopkCUDAContext *ctx,
    const float *d_float_array,
    topk_array_t *topk_array) {
    run_topk_extraction(ctx, d_float_array, topk_array);
    check_cuda(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize failed");
}

void topk_separation(
    TopkCUDAContext *ctx,
    float *d_float_array,
    topk_array_t *topk_array) {
    run_topk_extraction(ctx, d_float_array, topk_array);
    launch_zero_selected(ctx, d_float_array, topk_array);
    check_cuda(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize failed");
}

void topk_decompress(
    TopkCUDAContext *ctx,
    const topk_array_t *topk_array,
    float *d_float_array) {
    validate_context(ctx);
    if (d_float_array == nullptr) {
        throw std::invalid_argument("d_float_array must not be null");
    }
    validate_array_descriptor(ctx->num_rows, ctx->num_columns, topk_array);

    const std::size_t dense_elements =
        static_cast<std::size_t>(topk_array->num_rows) * static_cast<std::size_t>(topk_array->num_columns);
    check_cuda(cudaMemsetAsync(d_float_array, 0, dense_elements * sizeof(float), ctx->stream),
               "cudaMemsetAsync failed");
    launch_scatter(ctx, topk_array, d_float_array);
    check_cuda(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize failed");
}

void topk_apply(
    TopkCUDAContext *ctx,
    const topk_array_t *topk_array,
    float *d_float_array) {
    validate_context(ctx);
    if (d_float_array == nullptr) {
        throw std::invalid_argument("d_float_array must not be null");
    }
    validate_array_descriptor(ctx->num_rows, ctx->num_columns, topk_array);

    launch_scatter_add(ctx, topk_array, d_float_array);
    check_cuda(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize failed");
}

uint64_t topk_get_packed_size_for_shape(
    uint16_t num_rows,
    uint16_t num_columns,
    uint16_t num_topk_columns) {
    const uint64_t topk_elements =
        static_cast<uint64_t>(num_rows) * static_cast<uint64_t>(num_topk_columns);
    return sizeof(topk_packed_header_t) +
           topk_elements * (sizeof(uint16_t) + sizeof(float));
}

uint64_t topk_get_packed_size(const topk_array_t *topk_array) {
    if (topk_array == nullptr) {
        return 0;
    }
    return topk_get_packed_size_for_shape(
        topk_array->num_rows,
        topk_array->num_columns,
        topk_array->num_topk_columns);
}

int topk_validate_packed_buffer(
    const void *buffer,
    uint64_t buffer_size,
    topk_packed_header_t *header_out) noexcept {
    if (buffer == nullptr || buffer_size < sizeof(topk_packed_header_t)) {
        return 1;
    }

    const topk_packed_header_t *header = static_cast<const topk_packed_header_t *>(buffer);
    const uint64_t expected_size = topk_get_packed_size_for_shape(
        header->num_rows,
        header->num_columns,
        header->num_topk_columns);
    if (header->num_rows == 0 || header->num_columns == 0) {
        return 1;
    }
    if (header->num_topk_columns > header->num_columns) {
        return 1;
    }
    if (buffer_size != expected_size) {
        return 1;
    }

    if (header_out != nullptr) {
        *header_out = *header;
    }
    return 0;
}

int topk_pack_to_buffer(
    const topk_array_t *topk_array,
    void *buffer,
    uint64_t buffer_size) noexcept {
    try {
        if (topk_array == nullptr || buffer == nullptr) {
            return 1;
        }
        const uint64_t expected_size = topk_get_packed_size(topk_array);
        if (buffer_size != expected_size) {
            return 1;
        }

        topk_packed_header_t header{};
        header.num_rows = topk_array->num_rows;
        header.num_columns = topk_array->num_columns;
        header.num_topk_columns = topk_array->num_topk_columns;

        auto *bytes = static_cast<uint8_t *>(buffer);
        memcpy(bytes, &header, sizeof(header));

        const uint64_t topk_elements = topk_compute_num_topk_elements(
            topk_array->num_rows,
            topk_array->num_topk_columns);
        if (topk_elements == 0) {
            return 0;
        }

        uint8_t *indices_dst = bytes + sizeof(header);
        uint8_t *values_dst = indices_dst + topk_elements * sizeof(uint16_t);
        check_cuda(cudaMemcpyAsync(
                       indices_dst,
                       topk_array->d_topk_indices,
                       topk_elements * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync topk indices D2H failed");
        check_cuda(cudaMemcpyAsync(
                       values_dst,
                       topk_array->d_values,
                       topk_elements * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync topk values D2H failed");
        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}

int topk_unpack_from_buffer(
    const void *buffer,
    uint64_t buffer_size,
    topk_array_t *topk_array) noexcept {
    try {
        if (topk_array == nullptr) {
            return 1;
        }

        topk_packed_header_t header{};
        if (topk_validate_packed_buffer(buffer, buffer_size, &header) != 0) {
            return 1;
        }
        if (topk_array->num_rows != header.num_rows || topk_array->num_columns != header.num_columns) {
            return 1;
        }
        if (topk_array->num_topk_columns < header.num_topk_columns) {
            return 1;
        }

        const auto *bytes = static_cast<const uint8_t *>(buffer);
        const uint64_t topk_elements = topk_compute_num_topk_elements(
            header.num_rows,
            header.num_topk_columns);
        if (topk_elements > 0 && (topk_array->d_topk_indices == nullptr || topk_array->d_values == nullptr)) {
            return 1;
        }

        const uint8_t *indices_src = bytes + sizeof(header);
        const uint8_t *values_src = indices_src + topk_elements * sizeof(uint16_t);
        if (topk_elements > 0) {
            check_cuda(cudaMemcpyAsync(
                           topk_array->d_topk_indices,
                           indices_src,
                           topk_elements * sizeof(uint16_t),
                           cudaMemcpyHostToDevice),
                       "cudaMemcpyAsync topk indices H2D failed");
            check_cuda(cudaMemcpyAsync(
                           topk_array->d_values,
                           values_src,
                           topk_elements * sizeof(float),
                           cudaMemcpyHostToDevice),
                       "cudaMemcpyAsync topk values H2D failed");
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after topk unpack failed");
        }

        topk_array->num_topk_columns = header.num_topk_columns;
        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}
