#include "svd_lowrank_cuda/svd_lowrank_cuda.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>

namespace {

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

inline void check_devinfo(int info, const char* msg) {
    if (info != 0) {
        throw std::runtime_error(std::string(msg) + ": devInfo=" + std::to_string(info));
    }
}

inline void col_major_to_row_major(const float* src, int rows, int cols, float* dst) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[static_cast<std::size_t>(r) * cols + c] = src[static_cast<std::size_t>(c) * rows + r];
        }
    }
}

struct DevicePtrs {
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

__global__ void transpose_row_to_col_kernel(const float* src_row_major, float* dst_col_major, int m, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < m && c < n) {
        dst_col_major[static_cast<std::size_t>(c) * m + r] = src_row_major[static_cast<std::size_t>(r) * n + c];
    }
}

struct SVDLowrankCUDAContext {
    int m = 0;
    int n = 0;
    int k = 0;
    int M_work = 0;
    int N_work = 0;
    int lwork = 0;
    bool is_transposed = false;
    bool initialized = false;

    cublasHandle_t cublasH = nullptr;
    cusolverDnHandle_t cusolverH = nullptr;
    curandGenerator_t curandGen = nullptr;
    gesvdjInfo_t svdj_params = nullptr;
    DevicePtrs d;
    int h_info = 0;

    std::vector<float> h_U_work_col_major;
    std::vector<float> h_V_hat_col_major;
    SVDLowrankCPUResult host_result;

    void destroy_all() noexcept {
        d.free_all();
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
        is_transposed = false;
        h_info = 0;
        h_U_work_col_major.clear();
        h_V_hat_col_major.clear();
        host_result = SVDLowrankCPUResult{};
        initialized = false;
    }
};

SVDLowrankCUDAContext g_ctx;

void initialize_workspace_and_lwork(SVDLowrankCUDAContext* ctx) {
    int lwork_geqrf_M = 0;
    int lwork_orgqr_M = 0;
    int lwork_geqrf_N = 0;
    int lwork_orgqr_N = 0;
    int lwork_gesvdj = 0;

    check_cusolver(cusolverDnSgeqrf_bufferSize(ctx->cusolverH, ctx->M_work, ctx->k, ctx->d.d_X, ctx->M_work, &lwork_geqrf_M),
                   "cusolverDnSgeqrf_bufferSize M failed");
    check_cusolver(cusolverDnSorgqr_bufferSize(ctx->cusolverH, ctx->M_work, ctx->k, ctx->k, ctx->d.d_Q, ctx->M_work,
                                               ctx->d.d_tau_M, &lwork_orgqr_M),
                   "cusolverDnSorgqr_bufferSize M failed");
    check_cusolver(cusolverDnSgeqrf_bufferSize(ctx->cusolverH, ctx->N_work, ctx->k, ctx->d.d_X_temp, ctx->N_work,
                                               &lwork_geqrf_N),
                   "cusolverDnSgeqrf_bufferSize N failed");
    check_cusolver(cusolverDnSorgqr_bufferSize(ctx->cusolverH, ctx->N_work, ctx->k, ctx->k, ctx->d.d_Q_temp, ctx->N_work,
                                               ctx->d.d_tau_N, &lwork_orgqr_N),
                   "cusolverDnSorgqr_bufferSize N failed");
    check_cusolver(cusolverDnSgesvdj_bufferSize(
                       ctx->cusolverH,
                       CUSOLVER_EIG_MODE_VECTOR,
                       1,
                       ctx->k,
                       ctx->N_work,
                       ctx->d.d_B,
                       ctx->k,
                       ctx->d.d_S,
                       ctx->d.d_U_hat,
                       ctx->k,
                       ctx->d.d_V_hat,
                       ctx->N_work,
                       &lwork_gesvdj,
                       ctx->svdj_params),
                   "cusolverDnSgesvdj_bufferSize failed");

    ctx->lwork = std::max(1, std::max({lwork_geqrf_M, lwork_orgqr_M, lwork_geqrf_N, lwork_orgqr_N, lwork_gesvdj}));
    check_cuda(cudaMalloc(&ctx->d.d_work, static_cast<std::size_t>(ctx->lwork) * sizeof(float)), "cudaMalloc d_work failed");
}

void run_warmup(SVDLowrankCUDAContext* ctx, std::uint64_t seed) {
    check_curand(curandSetPseudoRandomGeneratorSeed(ctx->curandGen, seed), "warmup curandSetPseudoRandomGeneratorSeed failed");
    check_cuda(cudaMemset(ctx->d.d_A_row_major, 0, static_cast<std::size_t>(ctx->m) * ctx->n * sizeof(float)),
               "warmup cudaMemset d_A_row_major failed");

    constexpr int TILE = 16;
    const dim3 block(TILE, TILE);
    const dim3 grid((ctx->n + TILE - 1) / TILE, (ctx->m + TILE - 1) / TILE);
    transpose_row_to_col_kernel<<<grid, block>>>(ctx->d.d_A_row_major, ctx->d.d_A, ctx->m, ctx->n);
    check_cuda(cudaGetLastError(), "warmup transpose kernel launch failed");

    const cublasOperation_t op_A = ctx->is_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_A_H = ctx->is_transposed ? CUBLAS_OP_N : CUBLAS_OP_T;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    ctx->h_info = 0;

    check_curand(curandGenerateNormal(ctx->curandGen, ctx->d.d_R, static_cast<std::size_t>(ctx->N_work) * ctx->k, 0.0f, 1.0f),
                 "warmup curandGenerateNormal failed");
    check_cublas(cublasSgemm(ctx->cublasH, op_A, CUBLAS_OP_N, ctx->M_work, ctx->k, ctx->N_work, &alpha, ctx->d.d_A, ctx->m,
                             ctx->d.d_R, ctx->N_work, &beta, ctx->d.d_X, ctx->M_work),
                 "warmup cublasSgemm A*R failed");

    check_cusolver(cusolverDnSgeqrf(ctx->cusolverH, ctx->M_work, ctx->k, ctx->d.d_X, ctx->M_work, ctx->d.d_tau_M, ctx->d.d_work,
                                    ctx->lwork, ctx->d.d_devInfo),
                   "warmup cusolverDnSgeqrf failed");
    check_cuda(cudaMemcpy(&ctx->h_info, ctx->d.d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
               "warmup memcpy devInfo geqrf failed");
    check_devinfo(ctx->h_info, "warmup geqrf failed");

    check_cuda(cudaMemcpy(ctx->d.d_Q, ctx->d.d_X, static_cast<std::size_t>(ctx->M_work) * ctx->k * sizeof(float),
                          cudaMemcpyDeviceToDevice),
               "warmup memcpy X->Q failed");
    check_cusolver(cusolverDnSorgqr(ctx->cusolverH, ctx->M_work, ctx->k, ctx->k, ctx->d.d_Q, ctx->M_work, ctx->d.d_tau_M,
                                    ctx->d.d_work, ctx->lwork, ctx->d.d_devInfo),
                   "warmup cusolverDnSorgqr failed");
    check_cuda(cudaMemcpy(&ctx->h_info, ctx->d.d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
               "warmup memcpy devInfo orgqr failed");
    check_devinfo(ctx->h_info, "warmup orgqr failed");

    check_cublas(cublasSgemm(ctx->cublasH, op_A_H, CUBLAS_OP_N, ctx->N_work, ctx->k, ctx->M_work, &alpha, ctx->d.d_A, ctx->m,
                             ctx->d.d_Q, ctx->M_work, &beta, ctx->d.d_X_temp, ctx->N_work),
                 "warmup cublasSgemm A^T*Q failed");

    check_cublas(cublasSgemm(ctx->cublasH, CUBLAS_OP_T, op_A, ctx->k, ctx->N_work, ctx->M_work, &alpha, ctx->d.d_Q,
                             ctx->M_work, ctx->d.d_A, ctx->m, &beta, ctx->d.d_B, ctx->k),
                 "warmup projection gemm failed");
    check_cusolver(cusolverDnSgesvdj(
                       ctx->cusolverH,
                       CUSOLVER_EIG_MODE_VECTOR,
                       1,
                       ctx->k,
                       ctx->N_work,
                       ctx->d.d_B,
                       ctx->k,
                       ctx->d.d_S,
                       ctx->d.d_U_hat,
                       ctx->k,
                       ctx->d.d_V_hat,
                       ctx->N_work,
                       ctx->d.d_work,
                       ctx->lwork,
                       ctx->d.d_devInfo,
                       ctx->svdj_params),
                   "warmup cusolverDnSgesvdj failed");
    check_cuda(cudaMemcpy(&ctx->h_info, ctx->d.d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
               "warmup memcpy devInfo gesvdj failed");
    check_devinfo(ctx->h_info, "warmup gesvdj failed");

    check_cublas(cublasSgemm(ctx->cublasH, CUBLAS_OP_N, CUBLAS_OP_N, ctx->M_work, ctx->k, ctx->k, &alpha, ctx->d.d_Q,
                             ctx->M_work, ctx->d.d_U_hat, ctx->k, &beta, ctx->d.d_U_work, ctx->M_work),
                 "warmup recover U_work gemm failed");
    check_cuda(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize failed");
}

} // namespace

void svd_lowrank_cuda_initialize(int m, int n, int q, std::uint64_t warmup_seed) {
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("m and n must be positive");
    }
    if (q <= 0) {
        throw std::invalid_argument("q must be positive");
    }

    svd_lowrank_cuda_release();

    g_ctx.m = m;
    g_ctx.n = n;
    g_ctx.k = std::max(1, std::min({q, m, n}));
    g_ctx.is_transposed = (m < n);
    g_ctx.M_work = g_ctx.is_transposed ? n : m;
    g_ctx.N_work = g_ctx.is_transposed ? m : n;

    g_ctx.h_U_work_col_major.resize(static_cast<std::size_t>(g_ctx.M_work) * g_ctx.k);
    g_ctx.h_V_hat_col_major.resize(static_cast<std::size_t>(g_ctx.N_work) * g_ctx.k);
    g_ctx.host_result.m = m;
    g_ctx.host_result.n = n;
    g_ctx.host_result.k = g_ctx.k;
    g_ctx.host_result.U_row_major.resize(static_cast<std::size_t>(m) * g_ctx.k);
    g_ctx.host_result.S.resize(static_cast<std::size_t>(g_ctx.k));
    g_ctx.host_result.V_row_major.resize(static_cast<std::size_t>(n) * g_ctx.k);

    try {
        check_cublas(cublasCreate(&g_ctx.cublasH), "cublasCreate failed");
        check_cusolver(cusolverDnCreate(&g_ctx.cusolverH), "cusolverDnCreate failed");
        check_curand(curandCreateGenerator(&g_ctx.curandGen, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator failed");
        check_cusolver(cusolverDnCreateGesvdjInfo(&g_ctx.svdj_params), "cusolverDnCreateGesvdjInfo failed");

        check_cuda(cudaMalloc(&g_ctx.d.d_A_row_major, static_cast<std::size_t>(m) * n * sizeof(float)),
                   "cudaMalloc d_A_row_major failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_A, static_cast<std::size_t>(m) * n * sizeof(float)), "cudaMalloc d_A failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_R, static_cast<std::size_t>(g_ctx.N_work) * g_ctx.k * sizeof(float)),
                   "cudaMalloc d_R failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_X, static_cast<std::size_t>(g_ctx.M_work) * g_ctx.k * sizeof(float)),
                   "cudaMalloc d_X failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_Q, static_cast<std::size_t>(g_ctx.M_work) * g_ctx.k * sizeof(float)),
                   "cudaMalloc d_Q failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_X_temp, static_cast<std::size_t>(g_ctx.N_work) * g_ctx.k * sizeof(float)),
                   "cudaMalloc d_X_temp failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_Q_temp, static_cast<std::size_t>(g_ctx.N_work) * g_ctx.k * sizeof(float)),
                   "cudaMalloc d_Q_temp failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_tau_M, static_cast<std::size_t>(g_ctx.k) * sizeof(float)), "cudaMalloc d_tau_M failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_tau_N, static_cast<std::size_t>(g_ctx.k) * sizeof(float)), "cudaMalloc d_tau_N failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_B, static_cast<std::size_t>(g_ctx.k) * g_ctx.N_work * sizeof(float)),
                   "cudaMalloc d_B failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_U_hat, static_cast<std::size_t>(g_ctx.k) * g_ctx.k * sizeof(float)),
                   "cudaMalloc d_U_hat failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_S, static_cast<std::size_t>(g_ctx.k) * sizeof(float)), "cudaMalloc d_S failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_V_hat, static_cast<std::size_t>(g_ctx.N_work) * g_ctx.k * sizeof(float)),
                   "cudaMalloc d_V_hat failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_U_work, static_cast<std::size_t>(g_ctx.M_work) * g_ctx.k * sizeof(float)),
                   "cudaMalloc d_U_work failed");
        check_cuda(cudaMalloc(&g_ctx.d.d_devInfo, sizeof(int)), "cudaMalloc d_devInfo failed");

        initialize_workspace_and_lwork(&g_ctx);
        run_warmup(&g_ctx, warmup_seed);

        g_ctx.initialized = true;
    } catch (...) {
        g_ctx.destroy_all();
        throw;
    }
}

void svd_lowrank_cuda_release() {
    g_ctx.destroy_all();
}

const SVDLowrankCPUResult& svd_lowrank_cuda(const float* A_row_major, int niter, std::uint64_t seed, SVDLowrankTimings* timings) {
    if (!g_ctx.initialized) {
        throw std::logic_error("svd_lowrank_cuda is not initialized; call svd_lowrank_cuda_initialize(m, n, q) first");
    }
    if (A_row_major == nullptr) {
        throw std::invalid_argument("A_row_major must not be null");
    }

    const int iters = std::max(0, niter);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto h2d_start = std::chrono::high_resolution_clock::now();
    check_cuda(cudaMemcpy(g_ctx.d.d_A_row_major,
                          A_row_major,
                          static_cast<std::size_t>(g_ctx.m) * g_ctx.n * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy H2D A failed");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after H2D failed");
    auto h2d_end = std::chrono::high_resolution_clock::now();

    cudaEvent_t ev_transpose_start = nullptr;
    cudaEvent_t ev_transpose_stop = nullptr;
    float transpose_ms_f = 0.0f;

    check_cuda(cudaEventCreate(&ev_transpose_start), "cudaEventCreate transpose start failed");
    check_cuda(cudaEventCreate(&ev_transpose_stop), "cudaEventCreate transpose stop failed");
    check_cuda(cudaEventRecord(ev_transpose_start), "cudaEventRecord transpose start failed");

    constexpr int TILE = 16;
    const dim3 transpose_block(TILE, TILE);
    const dim3 transpose_grid((g_ctx.n + TILE - 1) / TILE, (g_ctx.m + TILE - 1) / TILE);
    transpose_row_to_col_kernel<<<transpose_grid, transpose_block>>>(g_ctx.d.d_A_row_major, g_ctx.d.d_A, g_ctx.m, g_ctx.n);
    check_cuda(cudaGetLastError(), "transpose kernel launch failed");

    check_cuda(cudaEventRecord(ev_transpose_stop), "cudaEventRecord transpose stop failed");
    check_cuda(cudaEventSynchronize(ev_transpose_stop), "cudaEventSynchronize transpose stop failed");
    check_cuda(cudaEventElapsedTime(&transpose_ms_f, ev_transpose_start, ev_transpose_stop),
               "cudaEventElapsedTime transpose failed");
    check_cuda(cudaEventDestroy(ev_transpose_start), "cudaEventDestroy transpose start failed");
    check_cuda(cudaEventDestroy(ev_transpose_stop), "cudaEventDestroy transpose stop failed");

    cudaEvent_t ev_compute_start = nullptr;
    cudaEvent_t ev_compute_stop = nullptr;
    float compute_ms_f = 0.0f;

    try {
        check_cuda(cudaEventCreate(&ev_compute_start), "cudaEventCreate start failed");
        check_cuda(cudaEventCreate(&ev_compute_stop), "cudaEventCreate stop failed");
        check_cuda(cudaEventRecord(ev_compute_start), "cudaEventRecord start failed");

        const cublasOperation_t op_A = g_ctx.is_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t op_A_H = g_ctx.is_transposed ? CUBLAS_OP_N : CUBLAS_OP_T;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        g_ctx.h_info = 0;

        check_curand(curandSetPseudoRandomGeneratorSeed(g_ctx.curandGen, seed),
                     "curandSetPseudoRandomGeneratorSeed failed");
        check_curand(curandGenerateNormal(g_ctx.curandGen,
                                          g_ctx.d.d_R,
                                          static_cast<std::size_t>(g_ctx.N_work) * g_ctx.k,
                                          0.0f,
                                          1.0f),
                     "curandGenerateNormal failed");

        check_cublas(cublasSgemm(g_ctx.cublasH,
                                 op_A,
                                 CUBLAS_OP_N,
                                 g_ctx.M_work,
                                 g_ctx.k,
                                 g_ctx.N_work,
                                 &alpha,
                                 g_ctx.d.d_A,
                                 g_ctx.m,
                                 g_ctx.d.d_R,
                                 g_ctx.N_work,
                                 &beta,
                                 g_ctx.d.d_X,
                                 g_ctx.M_work),
                     "Initial cublasSgemm A*R failed");

        check_cusolver(cusolverDnSgeqrf(g_ctx.cusolverH,
                                        g_ctx.M_work,
                                        g_ctx.k,
                                        g_ctx.d.d_X,
                                        g_ctx.M_work,
                                        g_ctx.d.d_tau_M,
                                        g_ctx.d.d_work,
                                        g_ctx.lwork,
                                        g_ctx.d.d_devInfo),
                       "cusolverDnSgeqrf initial failed");
        check_cuda(cudaMemcpy(&g_ctx.h_info, g_ctx.d.d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
                   "Memcpy devInfo geqrf failed");
        check_devinfo(g_ctx.h_info, "Initial geqrf failed");

        check_cuda(cudaMemcpy(g_ctx.d.d_Q,
                              g_ctx.d.d_X,
                              static_cast<std::size_t>(g_ctx.M_work) * g_ctx.k * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "Memcpy X->Q failed");
        check_cusolver(cusolverDnSorgqr(g_ctx.cusolverH,
                                        g_ctx.M_work,
                                        g_ctx.k,
                                        g_ctx.k,
                                        g_ctx.d.d_Q,
                                        g_ctx.M_work,
                                        g_ctx.d.d_tau_M,
                                        g_ctx.d.d_work,
                                        g_ctx.lwork,
                                        g_ctx.d.d_devInfo),
                       "cusolverDnSorgqr initial failed");
        check_cuda(cudaMemcpy(&g_ctx.h_info, g_ctx.d.d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
                   "Memcpy devInfo orgqr failed");
        check_devinfo(g_ctx.h_info, "Initial orgqr failed");

        for (int i = 0; i < iters; ++i) {
            check_cublas(cublasSgemm(g_ctx.cublasH,
                                     op_A_H,
                                     CUBLAS_OP_N,
                                     g_ctx.N_work,
                                     g_ctx.k,
                                     g_ctx.M_work,
                                     &alpha,
                                     g_ctx.d.d_A,
                                     g_ctx.m,
                                     g_ctx.d.d_Q,
                                     g_ctx.M_work,
                                     &beta,
                                     g_ctx.d.d_X_temp,
                                     g_ctx.N_work),
                         "Power iteration forward gemm failed");

            check_cusolver(cusolverDnSgeqrf(g_ctx.cusolverH,
                                            g_ctx.N_work,
                                            g_ctx.k,
                                            g_ctx.d.d_X_temp,
                                            g_ctx.N_work,
                                            g_ctx.d.d_tau_N,
                                            g_ctx.d.d_work,
                                            g_ctx.lwork,
                                            g_ctx.d.d_devInfo),
                           "Power geqrf N failed");
            check_cuda(cudaMemcpy(&g_ctx.h_info, g_ctx.d.d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
                       "Memcpy devInfo power geqrf N failed");
            check_devinfo(g_ctx.h_info, "Power geqrf N failed");

            check_cuda(cudaMemcpy(g_ctx.d.d_Q_temp,
                                  g_ctx.d.d_X_temp,
                                  static_cast<std::size_t>(g_ctx.N_work) * g_ctx.k * sizeof(float),
                                  cudaMemcpyDeviceToDevice),
                       "Memcpy X_temp->Q_temp failed");
            check_cusolver(cusolverDnSorgqr(g_ctx.cusolverH,
                                            g_ctx.N_work,
                                            g_ctx.k,
                                            g_ctx.k,
                                            g_ctx.d.d_Q_temp,
                                            g_ctx.N_work,
                                            g_ctx.d.d_tau_N,
                                            g_ctx.d.d_work,
                                            g_ctx.lwork,
                                            g_ctx.d.d_devInfo),
                           "Power orgqr N failed");
            check_cuda(cudaMemcpy(&g_ctx.h_info, g_ctx.d.d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
                       "Memcpy devInfo power orgqr N failed");
            check_devinfo(g_ctx.h_info, "Power orgqr N failed");

            check_cublas(cublasSgemm(g_ctx.cublasH,
                                     op_A,
                                     CUBLAS_OP_N,
                                     g_ctx.M_work,
                                     g_ctx.k,
                                     g_ctx.N_work,
                                     &alpha,
                                     g_ctx.d.d_A,
                                     g_ctx.m,
                                     g_ctx.d.d_Q_temp,
                                     g_ctx.N_work,
                                     &beta,
                                     g_ctx.d.d_X,
                                     g_ctx.M_work),
                         "Power iteration backward gemm failed");

            check_cusolver(cusolverDnSgeqrf(g_ctx.cusolverH,
                                            g_ctx.M_work,
                                            g_ctx.k,
                                            g_ctx.d.d_X,
                                            g_ctx.M_work,
                                            g_ctx.d.d_tau_M,
                                            g_ctx.d.d_work,
                                            g_ctx.lwork,
                                            g_ctx.d.d_devInfo),
                           "Power geqrf M failed");
            check_cuda(cudaMemcpy(&g_ctx.h_info, g_ctx.d.d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
                       "Memcpy devInfo power geqrf M failed");
            check_devinfo(g_ctx.h_info, "Power geqrf M failed");

            check_cuda(cudaMemcpy(g_ctx.d.d_Q,
                                  g_ctx.d.d_X,
                                  static_cast<std::size_t>(g_ctx.M_work) * g_ctx.k * sizeof(float),
                                  cudaMemcpyDeviceToDevice),
                       "Memcpy X->Q in power failed");
            check_cusolver(cusolverDnSorgqr(g_ctx.cusolverH,
                                            g_ctx.M_work,
                                            g_ctx.k,
                                            g_ctx.k,
                                            g_ctx.d.d_Q,
                                            g_ctx.M_work,
                                            g_ctx.d.d_tau_M,
                                            g_ctx.d.d_work,
                                            g_ctx.lwork,
                                            g_ctx.d.d_devInfo),
                           "Power orgqr M failed");
            check_cuda(cudaMemcpy(&g_ctx.h_info, g_ctx.d.d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
                       "Memcpy devInfo power orgqr M failed");
            check_devinfo(g_ctx.h_info, "Power orgqr M failed");
        }

        check_cublas(cublasSgemm(g_ctx.cublasH,
                                 CUBLAS_OP_T,
                                 op_A,
                                 g_ctx.k,
                                 g_ctx.N_work,
                                 g_ctx.M_work,
                                 &alpha,
                                 g_ctx.d.d_Q,
                                 g_ctx.M_work,
                                 g_ctx.d.d_A,
                                 g_ctx.m,
                                 &beta,
                                 g_ctx.d.d_B,
                                 g_ctx.k),
                     "Projection gemm Q^T A failed");

        check_cusolver(cusolverDnSgesvdj(
                           g_ctx.cusolverH,
                           CUSOLVER_EIG_MODE_VECTOR,
                           1,
                           g_ctx.k,
                           g_ctx.N_work,
                           g_ctx.d.d_B,
                           g_ctx.k,
                           g_ctx.d.d_S,
                           g_ctx.d.d_U_hat,
                           g_ctx.k,
                           g_ctx.d.d_V_hat,
                           g_ctx.N_work,
                           g_ctx.d.d_work,
                           g_ctx.lwork,
                           g_ctx.d.d_devInfo,
                           g_ctx.svdj_params),
                       "cusolverDnSgesvdj failed");
        check_cuda(cudaMemcpy(&g_ctx.h_info, g_ctx.d.d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
                   "Memcpy devInfo gesvdj failed");
        check_devinfo(g_ctx.h_info, "gesvdj failed");

        check_cublas(cublasSgemm(g_ctx.cublasH,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 g_ctx.M_work,
                                 g_ctx.k,
                                 g_ctx.k,
                                 &alpha,
                                 g_ctx.d.d_Q,
                                 g_ctx.M_work,
                                 g_ctx.d.d_U_hat,
                                 g_ctx.k,
                                 &beta,
                                 g_ctx.d.d_U_work,
                                 g_ctx.M_work),
                     "Recover U_work gemm failed");

        check_cuda(cudaEventRecord(ev_compute_stop), "cudaEventRecord stop failed");
        check_cuda(cudaEventSynchronize(ev_compute_stop), "cudaEventSynchronize stop failed");
        check_cuda(cudaEventElapsedTime(&compute_ms_f, ev_compute_start, ev_compute_stop), "cudaEventElapsedTime failed");
    } catch (...) {
        if (ev_compute_start) {
            cudaEventDestroy(ev_compute_start);
        }
        if (ev_compute_stop) {
            cudaEventDestroy(ev_compute_stop);
        }
        throw;
    }

    check_cuda(cudaEventDestroy(ev_compute_start), "cudaEventDestroy start failed");
    check_cuda(cudaEventDestroy(ev_compute_stop), "cudaEventDestroy stop failed");

    auto d2h_start = std::chrono::high_resolution_clock::now();
    check_cuda(cudaMemcpy(g_ctx.h_U_work_col_major.data(),
                          g_ctx.d.d_U_work,
                          static_cast<std::size_t>(g_ctx.M_work) * g_ctx.k * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "Memcpy U_work D2H failed");
    check_cuda(cudaMemcpy(g_ctx.h_V_hat_col_major.data(),
                          g_ctx.d.d_V_hat,
                          static_cast<std::size_t>(g_ctx.N_work) * g_ctx.k * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "Memcpy V_hat D2H failed");
    check_cuda(cudaMemcpy(g_ctx.host_result.S.data(),
                          g_ctx.d.d_S,
                          static_cast<std::size_t>(g_ctx.k) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "Memcpy S D2H failed");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after D2H failed");
    auto d2h_end = std::chrono::high_resolution_clock::now();

    if (g_ctx.is_transposed) {
        col_major_to_row_major(g_ctx.h_V_hat_col_major.data(), g_ctx.N_work, g_ctx.k, g_ctx.host_result.U_row_major.data());
        col_major_to_row_major(g_ctx.h_U_work_col_major.data(), g_ctx.M_work, g_ctx.k, g_ctx.host_result.V_row_major.data());
    } else {
        col_major_to_row_major(g_ctx.h_U_work_col_major.data(), g_ctx.M_work, g_ctx.k, g_ctx.host_result.U_row_major.data());
        col_major_to_row_major(g_ctx.h_V_hat_col_major.data(), g_ctx.N_work, g_ctx.k, g_ctx.host_result.V_row_major.data());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (timings != nullptr) {
        timings->h2d_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
        timings->transpose_ms = static_cast<double>(transpose_ms_f);
        timings->compute_ms = static_cast<double>(compute_ms_f);
        timings->d2h_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
        timings->total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return g_ctx.host_result;
}
