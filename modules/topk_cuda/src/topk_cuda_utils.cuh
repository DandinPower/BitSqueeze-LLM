#pragma once

#include <topk_cuda.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace topk_detail {

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

inline void launch_build_sort_inputs(
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

inline void launch_gather_topk(
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

inline void launch_zero_selected(
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

inline void launch_scatter(
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

inline void launch_scatter_add(
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

inline void run_segmented_sort(TopkCUDAContext *ctx) {
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

inline void run_topk_extraction(
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

inline void warmup(TopkCUDAContext *ctx, unsigned long long warmup_seed) {
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
        check_cuda(
            cudaMalloc(&d_decompressed, dense_elements * sizeof(float)),
            "warmup cudaMalloc d_decompressed failed");
        if (topk_elements > 0) {
            check_cuda(
                cudaMalloc(&d_topk_indices, topk_elements * sizeof(uint16_t)),
                "warmup cudaMalloc d_topk_indices failed");
            check_cuda(cudaMalloc(&d_topk_values, topk_elements * sizeof(float)), "warmup cudaMalloc d_topk_values failed");
        }

        check_cuda(
            cudaMemsetAsync(d_dense, 0, dense_elements * sizeof(float), ctx->stream),
            "warmup cudaMemsetAsync d_dense failed");
        check_cuda(
            cudaMemsetAsync(d_decompressed, 0, dense_elements * sizeof(float), ctx->stream),
            "warmup cudaMemsetAsync d_decompressed failed");

        topk_array_t warmup_array{};
        topk_bind_array(&warmup_array, ctx->num_rows, ctx->num_columns, warmup_k, d_topk_indices, d_topk_values);

        run_topk_extraction(ctx, d_dense, &warmup_array);
        launch_zero_selected(ctx, d_dense, &warmup_array);
        check_cuda(
            cudaMemsetAsync(d_decompressed, 0, dense_elements * sizeof(float), ctx->stream),
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

}  // namespace topk_detail
