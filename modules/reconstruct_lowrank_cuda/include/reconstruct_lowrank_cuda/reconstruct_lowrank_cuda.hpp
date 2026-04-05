#pragma once

#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

struct ReconstructLowrankCUDAPtrs {
    float *d_scaled_u = nullptr;

    void free_all() noexcept {
        if (d_scaled_u) cudaFree(d_scaled_u);

        d_scaled_u = nullptr;
    }
};

struct ReconstructLowrankCUDAContext {
    int m = 0;
    int n = 0;
    int k = 0;
    bool initialized = false;

    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handler = nullptr;
    ReconstructLowrankCUDAPtrs cuda_ptrs;

    void destroy_all() noexcept {
        cuda_ptrs.free_all();

        if (cublas_handler) {
            cublasDestroy(cublas_handler);
            cublas_handler = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }

        m = 0;
        n = 0;
        k = 0;
        initialized = false;
    }
};

namespace reconstruct_lowrank_cuda_detail {

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

} // namespace reconstruct_lowrank_cuda_detail

void reconstruct_lowrank_cuda_release(ReconstructLowrankCUDAContext* ctx) noexcept;

void reconstruct_lowrank_cuda_initialize(
    ReconstructLowrankCUDAContext* ctx,
    int m,
    int n,
    int k,
    unsigned long long warmup_seed = 1234ULL);

void reconstruct_lowrank_cuda(
    ReconstructLowrankCUDAContext* ctx,
    const float* d_U_row_major,
    const float* d_S,
    const float* d_V_row_major,
    float* d_out_row_major);
