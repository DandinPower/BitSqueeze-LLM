#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>

struct SVDLowrankCUDADevicePtrs {
    float* d_A = nullptr;
    float* d_R = nullptr;
    float* d_X = nullptr;
    float* d_X_temp = nullptr;
    float* d_tau_M = nullptr;
    float* d_tau_N = nullptr;
    float* d_work = nullptr;
    float* d_B = nullptr;
    float* d_U_hat = nullptr;
    float* d_S = nullptr;
    float* d_V_hat = nullptr;
    float* d_U_work = nullptr;
    int* d_devInfo = nullptr;

    void free_all() noexcept;
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

    void destroy_graph() noexcept;
    void destroy_all() noexcept;
};

void svd_lowrank_cuda(
    const float* d_A_row_major,
    float* d_U_row_major,
    float* d_S,
    float* d_V_row_major,
    unsigned long long seed = 1234ULL);

void svd_lowrank_cuda_initialize(
    int m,
    int n,
    int q,
    int niter,
    unsigned long long warmup_seed = 1234ULL);

void svd_lowrank_cuda_release();
