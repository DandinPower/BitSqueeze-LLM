#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>

namespace svd_lowrank_cuda_detail {

constexpr int kTransposeTileSize = 16;

inline std::size_t matrix_elements(int rows, int cols) {
    return static_cast<std::size_t>(rows) * cols;
}

inline void check_cuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

inline void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": cublas error code " + std::to_string(status));
    }
}

inline void check_curand(curandStatus_t status, const char* msg) {
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": curand error code " + std::to_string(status));
    }
}

inline void check_cusolver(cusolverStatus_t status, const char* msg) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": cusolver error code " + std::to_string(status));
    }
}

struct PendingCudaGraphCapture {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    bool active = false;

    ~PendingCudaGraphCapture() {
        if (!active) {
            return;
        }
        cudaStreamEndCapture(stream, &graph);
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }
};

inline PendingCudaGraphCapture begin_graph_capture(cudaStream_t stream, cudaStreamCaptureMode mode, const char* msg) {
    check_cuda(cudaStreamBeginCapture(stream, mode), msg);
    return PendingCudaGraphCapture{stream, nullptr, true};
}

struct OwnedCudaGraph {
    cudaGraph_t graph = nullptr;

    ~OwnedCudaGraph() {
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }

    cudaGraph_t release() noexcept {
        cudaGraph_t released = graph;
        graph = nullptr;
        return released;
    }
};

inline dim3 make_transpose_block() {
    return dim3(kTransposeTileSize, kTransposeTileSize);
}

inline dim3 make_transpose_grid(int rows, int cols) {
    return dim3(
        (cols + kTransposeTileSize - 1) / kTransposeTileSize,
        (rows + kTransposeTileSize - 1) / kTransposeTileSize);
}

__global__ void transpose_row_to_col_kernel(const float* src_row_major, float* dst_col_major, int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        dst_col_major[static_cast<std::size_t>(col) * rows + row] =
            src_row_major[static_cast<std::size_t>(row) * cols + col];
    }
}

__global__ void transpose_col_to_row_kernel(const float* src_col_major, float* dst_row_major, int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        dst_row_major[static_cast<std::size_t>(row) * cols + col] =
            src_col_major[static_cast<std::size_t>(col) * rows + row];
    }
}

inline void launch_row_to_col_transpose(
    const float* src_row_major,
    float* dst_col_major,
    int rows,
    int cols,
    cudaStream_t stream,
    const char* msg) {
    transpose_row_to_col_kernel<<<make_transpose_grid(rows, cols), make_transpose_block(), 0, stream>>>(
        src_row_major,
        dst_col_major,
        rows,
        cols);
    check_cuda(cudaGetLastError(), msg);
}

inline void launch_col_to_row_transpose(
    const float* src_col_major,
    float* dst_row_major,
    int rows,
    int cols,
    cudaStream_t stream,
    const char* msg) {
    transpose_col_to_row_kernel<<<make_transpose_grid(rows, cols), make_transpose_block(), 0, stream>>>(
        src_col_major,
        dst_row_major,
        rows,
        cols);
    check_cuda(cudaGetLastError(), msg);
}

} // namespace svd_lowrank_cuda_detail
