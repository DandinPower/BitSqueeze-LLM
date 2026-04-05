#include <topk.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;

inline void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

inline void validate_context(const TopkCUDAContext *ctx) {
    if (ctx == nullptr) {
        throw std::invalid_argument("ctx must not be null");
    }
    if (!ctx->initialized) {
        throw std::runtime_error("context is not initialized");
    }
}

inline void validate_array_descriptor(
    uint16_t expected_rows,
    uint16_t expected_columns,
    const topk_array_t *topk_array) {
    if (topk_array == nullptr) {
        throw std::invalid_argument("topk_array must not be null");
    }
    if (topk_array->num_rows != expected_rows || topk_array->num_columns != expected_columns) {
        throw std::invalid_argument("topk_array shape does not match the initialized context");
    }

    const uint32_t topk_elements =
        static_cast<uint32_t>(topk_array->num_rows) * static_cast<uint32_t>(topk_array->num_topk_columns);
    if (topk_elements > 0 && (topk_array->d_topk_indices == nullptr || topk_array->d_values == nullptr)) {
        throw std::invalid_argument("topk_array device pointers must not be null when k > 0");
    }
}

__global__ void build_sort_inputs_kernel(
    const float *input,
    float *sort_keys,
    uint16_t *sort_indices,
    std::size_t num_columns,
    std::size_t total_elements) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    const float value = input[idx];
    float importance = fabsf(value);
    if (isnan(importance)) {
        importance = -INFINITY;
    }
    sort_keys[idx] = importance;
    sort_indices[idx] = static_cast<uint16_t>(idx % num_columns);
}

__global__ void gather_topk_kernel(
    const float *input,
    const uint16_t *sorted_indices,
    uint16_t *topk_indices,
    float *topk_values,
    uint16_t num_rows,
    uint16_t num_columns,
    uint16_t num_topk_columns) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_topk =
        static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(num_topk_columns);
    if (idx >= total_topk) {
        return;
    }

    const uint16_t row = static_cast<uint16_t>(idx / num_topk_columns);
    const uint16_t local_topk = static_cast<uint16_t>(idx % num_topk_columns);
    const std::size_t sorted_offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(num_columns) + local_topk;
    const uint16_t column = sorted_indices[sorted_offset];
    topk_indices[idx] = column;
    topk_values[idx] = input[static_cast<std::size_t>(row) * num_columns + column];
}

__global__ void zero_selected_entries_kernel(
    float *dense,
    const uint16_t *topk_indices,
    uint16_t num_rows,
    uint16_t num_columns,
    uint16_t num_topk_columns) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_topk =
        static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(num_topk_columns);
    if (idx >= total_topk) {
        return;
    }

    const uint16_t row = static_cast<uint16_t>(idx / num_topk_columns);
    const uint16_t column = topk_indices[idx];
    dense[static_cast<std::size_t>(row) * num_columns + column] = 0.0f;
}

__global__ void scatter_topk_kernel(
    const uint16_t *topk_indices,
    const float *topk_values,
    float *dense,
    uint16_t num_rows,
    uint16_t num_columns,
    uint16_t num_topk_columns) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_topk =
        static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(num_topk_columns);
    if (idx >= total_topk) {
        return;
    }

    const uint16_t row = static_cast<uint16_t>(idx / num_topk_columns);
    const uint16_t column = topk_indices[idx];
    dense[static_cast<std::size_t>(row) * num_columns + column] = topk_values[idx];
}

__global__ void scatter_add_topk_kernel(
    const uint16_t *topk_indices,
    const float *topk_values,
    float *dense,
    uint16_t num_rows,
    uint16_t num_columns,
    uint16_t num_topk_columns) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_topk =
        static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(num_topk_columns);
    if (idx >= total_topk) {
        return;
    }

    const uint16_t row = static_cast<uint16_t>(idx / num_topk_columns);
    const uint16_t column = topk_indices[idx];
    dense[static_cast<std::size_t>(row) * num_columns + column] += topk_values[idx];
}

void launch_build_sort_inputs(
    TopkCUDAContext *ctx,
    const float *input) {
    const std::size_t total_elements =
        static_cast<std::size_t>(ctx->num_rows) * static_cast<std::size_t>(ctx->num_columns);
    const int grid = static_cast<int>((total_elements + kBlockSize - 1) / kBlockSize);
    build_sort_inputs_kernel<<<grid, kBlockSize, 0, ctx->stream>>>(
        input,
        ctx->cuda_ptrs.d_sort_keys_in,
        ctx->cuda_ptrs.d_sort_indices_in,
        static_cast<std::size_t>(ctx->num_columns),
        total_elements);
    check_cuda(cudaGetLastError(), "build_sort_inputs kernel launch failed");
}

void launch_gather_topk(
    TopkCUDAContext *ctx,
    const float *input,
    topk_array_t *topk_array) {
    const std::size_t total_topk = topk_compute_num_topk_elements(
        topk_array->num_rows,
        topk_array->num_topk_columns);
    if (total_topk == 0) {
        return;
    }

    const int grid = static_cast<int>((total_topk + kBlockSize - 1) / kBlockSize);
    gather_topk_kernel<<<grid, kBlockSize, 0, ctx->stream>>>(
        input,
        ctx->cuda_ptrs.d_sort_indices_out,
        topk_array->d_topk_indices,
        topk_array->d_values,
        topk_array->num_rows,
        topk_array->num_columns,
        topk_array->num_topk_columns);
    check_cuda(cudaGetLastError(), "gather_topk kernel launch failed");
}

void launch_zero_selected(
    TopkCUDAContext *ctx,
    float *dense,
    const topk_array_t *topk_array) {
    const std::size_t total_topk = topk_compute_num_topk_elements(
        topk_array->num_rows,
        topk_array->num_topk_columns);
    if (total_topk == 0) {
        return;
    }

    const int grid = static_cast<int>((total_topk + kBlockSize - 1) / kBlockSize);
    zero_selected_entries_kernel<<<grid, kBlockSize, 0, ctx->stream>>>(
        dense,
        topk_array->d_topk_indices,
        topk_array->num_rows,
        topk_array->num_columns,
        topk_array->num_topk_columns);
    check_cuda(cudaGetLastError(), "zero_selected_entries kernel launch failed");
}

void launch_scatter(
    TopkCUDAContext *ctx,
    const topk_array_t *topk_array,
    float *dense) {
    const std::size_t total_topk = topk_compute_num_topk_elements(
        topk_array->num_rows,
        topk_array->num_topk_columns);
    if (total_topk == 0) {
        return;
    }

    const int grid = static_cast<int>((total_topk + kBlockSize - 1) / kBlockSize);
    scatter_topk_kernel<<<grid, kBlockSize, 0, ctx->stream>>>(
        topk_array->d_topk_indices,
        topk_array->d_values,
        dense,
        topk_array->num_rows,
        topk_array->num_columns,
        topk_array->num_topk_columns);
    check_cuda(cudaGetLastError(), "scatter_topk kernel launch failed");
}

void launch_scatter_add(
    TopkCUDAContext *ctx,
    const topk_array_t *topk_array,
    float *dense) {
    const std::size_t total_topk = topk_compute_num_topk_elements(
        topk_array->num_rows,
        topk_array->num_topk_columns);
    if (total_topk == 0) {
        return;
    }

    const int grid = static_cast<int>((total_topk + kBlockSize - 1) / kBlockSize);
    scatter_add_topk_kernel<<<grid, kBlockSize, 0, ctx->stream>>>(
        topk_array->d_topk_indices,
        topk_array->d_values,
        dense,
        topk_array->num_rows,
        topk_array->num_columns,
        topk_array->num_topk_columns);
    check_cuda(cudaGetLastError(), "scatter_add_topk kernel launch failed");
}

void run_segmented_sort(TopkCUDAContext *ctx) {
    const int total_elements =
        static_cast<int>(static_cast<uint32_t>(ctx->num_rows) * static_cast<uint32_t>(ctx->num_columns));
    check_cuda(
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            ctx->cuda_ptrs.d_cub_temp_storage,
            ctx->cuda_ptrs.cub_temp_storage_bytes,
            ctx->cuda_ptrs.d_sort_keys_in,
            ctx->cuda_ptrs.d_sort_keys_out,
            ctx->cuda_ptrs.d_sort_indices_in,
            ctx->cuda_ptrs.d_sort_indices_out,
            total_elements,
            static_cast<int>(ctx->num_rows),
            ctx->cuda_ptrs.d_segment_offsets_begin,
            ctx->cuda_ptrs.d_segment_offsets_end,
            0,
            static_cast<int>(sizeof(float) * 8),
            ctx->stream),
        "cub segmented sort failed");
}

void run_topk_extraction(
    TopkCUDAContext *ctx,
    const float *d_float_array,
    topk_array_t *topk_array) {
    validate_context(ctx);
    if (d_float_array == nullptr) {
        throw std::invalid_argument("d_float_array must not be null");
    }
    validate_array_descriptor(ctx->num_rows, ctx->num_columns, topk_array);
    if (topk_array->num_topk_columns > ctx->num_columns) {
        throw std::invalid_argument("topk k must not exceed the number of columns");
    }

    if (topk_array->num_topk_columns == 0) {
        return;
    }

    launch_build_sort_inputs(ctx, d_float_array);
    run_segmented_sort(ctx);
    launch_gather_topk(ctx, d_float_array, topk_array);
}

void warmup(TopkCUDAContext *ctx, unsigned long long warmup_seed) {
    (void) warmup_seed;

    float *d_dense = nullptr;
    float *d_decompressed = nullptr;
    uint16_t *d_topk_indices = nullptr;
    float *d_topk_values = nullptr;

    try {
        const std::size_t dense_elements =
            static_cast<std::size_t>(ctx->num_rows) * static_cast<std::size_t>(ctx->num_columns);
        const uint16_t warmup_k = ctx->num_columns == 0 ? 0 : static_cast<uint16_t>(1);
        const std::size_t topk_elements =
            static_cast<std::size_t>(ctx->num_rows) * static_cast<std::size_t>(warmup_k);

        check_cuda(cudaMalloc(&d_dense, dense_elements * sizeof(float)), "warmup cudaMalloc d_dense failed");
        check_cuda(cudaMalloc(&d_decompressed, dense_elements * sizeof(float)), "warmup cudaMalloc d_decompressed failed");
        if (topk_elements > 0) {
            check_cuda(cudaMalloc(&d_topk_indices, topk_elements * sizeof(uint16_t)),
                       "warmup cudaMalloc d_topk_indices failed");
            check_cuda(cudaMalloc(&d_topk_values, topk_elements * sizeof(float)),
                       "warmup cudaMalloc d_topk_values failed");
        }

        check_cuda(cudaMemsetAsync(d_dense, 0, dense_elements * sizeof(float), ctx->stream),
                   "warmup cudaMemsetAsync d_dense failed");
        check_cuda(cudaMemsetAsync(d_decompressed, 0, dense_elements * sizeof(float), ctx->stream),
                   "warmup cudaMemsetAsync d_decompressed failed");

        topk_array_t warmup_array{};
        topk_bind_array(&warmup_array, ctx->num_rows, ctx->num_columns, warmup_k, d_topk_indices, d_topk_values);

        run_topk_extraction(ctx, d_dense, &warmup_array);
        launch_zero_selected(ctx, d_dense, &warmup_array);
        check_cuda(cudaMemsetAsync(d_decompressed, 0, dense_elements * sizeof(float), ctx->stream),
                   "warmup cudaMemsetAsync d_decompressed reset failed");
        launch_scatter(ctx, &warmup_array, d_decompressed);
        launch_scatter_add(ctx, &warmup_array, d_dense);
        check_cuda(cudaStreamSynchronize(ctx->stream), "warmup cudaStreamSynchronize failed");
    } catch (...) {
        cudaFree(d_dense);
        cudaFree(d_decompressed);
        cudaFree(d_topk_indices);
        cudaFree(d_topk_values);
        throw;
    }

    cudaFree(d_dense);
    cudaFree(d_decompressed);
    cudaFree(d_topk_indices);
    cudaFree(d_topk_values);
}

}  // namespace

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

        check_cuda(cudaMemcpy(
                       ctx->cuda_ptrs.d_segment_offsets_begin,
                       segment_offsets_begin.data(),
                       segment_offsets_begin.size() * sizeof(int),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpy segment_offsets_begin failed");
        check_cuda(cudaMemcpy(
                       ctx->cuda_ptrs.d_segment_offsets_end,
                       segment_offsets_end.data(),
                       segment_offsets_end.size() * sizeof(int),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpy segment_offsets_end failed");

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
        check_cuda(cudaMemcpy(
                       indices_dst,
                       topk_array->d_topk_indices,
                       topk_elements * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy topk indices D2H failed");
        check_cuda(cudaMemcpy(
                       values_dst,
                       topk_array->d_values,
                       topk_elements * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy topk values D2H failed");
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
            check_cuda(cudaMemcpy(
                           topk_array->d_topk_indices,
                           indices_src,
                           topk_elements * sizeof(uint16_t),
                           cudaMemcpyHostToDevice),
                       "cudaMemcpy topk indices H2D failed");
            check_cuda(cudaMemcpy(
                           topk_array->d_values,
                           values_src,
                           topk_elements * sizeof(float),
                           cudaMemcpyHostToDevice),
                       "cudaMemcpy topk values H2D failed");
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after topk unpack failed");
        }

        topk_array->num_topk_columns = header.num_topk_columns;
        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}
