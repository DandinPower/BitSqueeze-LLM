#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>

struct SVDLowrankCPUResult {
    int m = 0;
    int n = 0;
    int k = 0;
    std::vector<float> U_row_major;
    std::vector<float> S;
    std::vector<float> V_row_major;
};

struct SVDLowrankCUDADevicePtrs {
    float *d_A_row_major = nullptr, *d_A = nullptr, *d_R = nullptr, *d_X = nullptr, *d_Q = nullptr;
    float *d_X_temp = nullptr, *d_Q_temp = nullptr, *d_tau_M = nullptr, *d_tau_N = nullptr;
    float *d_work = nullptr, *d_B = nullptr, *d_U_hat = nullptr, *d_S = nullptr, *d_V_hat = nullptr;
    float *d_U_work = nullptr;
    int* d_devInfo = nullptr;

    void free_all() noexcept {
        cudaFree(d_A_row_major);
        cudaFree(d_A);
        cudaFree(d_R);
        cudaFree(d_X);
        cudaFree(d_Q);
        cudaFree(d_X_temp);
        cudaFree(d_Q_temp);
        cudaFree(d_tau_M);
        cudaFree(d_tau_N);
        cudaFree(d_work);
        cudaFree(d_B);
        cudaFree(d_U_hat);
        cudaFree(d_S);
        cudaFree(d_V_hat);
        cudaFree(d_U_work);
        cudaFree(d_devInfo);

        d_A_row_major = nullptr;
        d_A = nullptr;
        d_R = nullptr;
        d_X = nullptr;
        d_Q = nullptr;
        d_X_temp = nullptr;
        d_Q_temp = nullptr;
        d_tau_M = nullptr;
        d_tau_N = nullptr;
        d_work = nullptr;
        d_B = nullptr;
        d_U_hat = nullptr;
        d_S = nullptr;
        d_V_hat = nullptr;
        d_U_work = nullptr;
        d_devInfo = nullptr;
    }
};

struct SVDLowrankCUDAContext {
    int m = 0;
    int n = 0;
    int k = 0;
    int M_work = 0;
    int N_work = 0;
    int lwork = 0;
    int niter = 0;
    bool is_transposed = false;
    bool initialized = false;

    cudaStream_t stream = nullptr;
    cudaGraph_t lowrank_graph = nullptr;
    cudaGraphExec_t lowrank_graph_exec = nullptr;
    int graph_niter = -1;

    cublasHandle_t cublasH = nullptr;
    cusolverDnHandle_t cusolverH = nullptr;
    curandGenerator_t curandGen = nullptr;
    gesvdjInfo_t svdj_params = nullptr;
    SVDLowrankCUDADevicePtrs d;

    std::vector<float> h_U_work_col_major;
    std::vector<float> h_V_hat_col_major;
    SVDLowrankCPUResult host_result;

    void destroy_graph() noexcept {
        if (lowrank_graph_exec) {
            cudaGraphExecDestroy(lowrank_graph_exec);
            lowrank_graph_exec = nullptr;
        }
        if (lowrank_graph) {
            cudaGraphDestroy(lowrank_graph);
            lowrank_graph = nullptr;
        }
        graph_niter = -1;
    }

    void destroy_all() noexcept {
        destroy_graph();
        d.free_all();
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        if (svdj_params) {
            cusolverDnDestroyGesvdjInfo(svdj_params);
            svdj_params = nullptr;
        }
        if (curandGen) {
            curandDestroyGenerator(curandGen);
            curandGen = nullptr;
        }
        if (cusolverH) {
            cusolverDnDestroy(cusolverH);
            cusolverH = nullptr;
        }
        if (cublasH) {
            cublasDestroy(cublasH);
            cublasH = nullptr;
        }

        m = 0;
        n = 0;
        k = 0;
        M_work = 0;
        N_work = 0;
        lwork = 0;
        niter = 0;
        is_transposed = false;
        h_U_work_col_major.clear();
        h_V_hat_col_major.clear();
        host_result = SVDLowrankCPUResult{};
        initialized = false;
    }
};

const SVDLowrankCPUResult& svd_lowrank_cuda(
    const float* A_row_major,
    unsigned long long seed = 1234ULL);

void svd_lowrank_cuda_initialize(
    int m,
    int n,
    int q,
    int niter,
    unsigned long long warmup_seed = 1234ULL);

void svd_lowrank_cuda_release();

namespace svd_lowrank_cuda_detail {

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

inline void col_major_to_row_major(const float* src, int rows, int cols, float* dst) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[static_cast<std::size_t>(r) * cols + c] = src[static_cast<std::size_t>(c) * rows + r];
        }
    }
}

} // namespace svd_lowrank_cuda_detail
