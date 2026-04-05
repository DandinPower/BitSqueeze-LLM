#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

struct ReconstructLowrankCUDAPtrs {
    float *d_U = nullptr;
    float *d_S = nullptr;
    float *d_V = nullptr;
    float *d_result = nullptr;

    void free_all() noexcept {
        cudaFree(d_U);
        cudaFree(d_S);
        cudaFree(d_V);
        cudaFree(d_result);

        d_U = nullptr;
        d_S = nullptr;
        d_V = nullptr;
        d_result = nullptr;
    }
};

struct ReconstructLowrankHostPtrs {
    std::vector<float> h_result;

    void free_all() noexcept {
        h_result.clear();
    }
};

struct ReconstructLowrankCUDAContext {
    int m = 0;
    int n = 0;
    int k = 0;
    bool initialized = false;

    cublasHandle_t cublas_handler = nullptr;
    ReconstructLowrankCUDAPtrs cuda_ptrs;
    ReconstructLowrankHostPtrs host_ptrs;

    void destroy_all() noexcept {
        cuda_ptrs.free_all();
        host_ptrs.free_all();

        if (cublas_handler) {
            cublasDestroy(cublas_handler);
            cublas_handler = nullptr;
        }

        m = 0;
        n = 0;
        k = 0;
        initialized = false;
    }
};

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

void reconstruct_lowrank_cuda_release(ReconstructLowrankCUDAContext* ctx) noexcept;

void reconstruct_lowrank_cuda_initialize(
    ReconstructLowrankCUDAContext* ctx,
    int m,
    int n,
    int k,
    unsigned long long warmup_seed = 1234ULL);

void reconstruct_lowrank_cuda(
    ReconstructLowrankCUDAContext* ctx,
    const std::vector<float>& U_row_major,
    const std::vector<float>& S,
    const std::vector<float>& V_row_major,
    std::vector<float>* out_row_major);
