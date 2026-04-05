#include <svd_lowrank_cuda/svd_lowrank_cuda.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using svd_lowrank_cuda_detail::check_cublas;
using svd_lowrank_cuda_detail::check_cuda;
using svd_lowrank_cuda_detail::check_curand;
using svd_lowrank_cuda_detail::check_cusolver;
using svd_lowrank_cuda_detail::col_major_to_row_major;

__global__ void transpose_row_to_col_kernel(const float* src_row_major, float* dst_col_major, int m, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < m && c < n) {
        dst_col_major[static_cast<std::size_t>(c) * m + r] = src_row_major[static_cast<std::size_t>(r) * n + c];
    }
}

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

void load_and_transpose_input(SVDLowrankCUDAContext* ctx, const float* A_row_major) {
    const std::size_t matrix_bytes = static_cast<std::size_t>(ctx->m) * ctx->n * sizeof(float);
    if (A_row_major != nullptr) {
        check_cuda(cudaMemcpyAsync(ctx->d.d_A_row_major, A_row_major, matrix_bytes, cudaMemcpyHostToDevice, ctx->stream),
                   "cudaMemcpyAsync H2D A failed");
    } else {
        check_cuda(cudaMemsetAsync(ctx->d.d_A_row_major, 0, matrix_bytes, ctx->stream), "cudaMemsetAsync d_A_row_major failed");
    }

    constexpr int TILE = 16;
    const dim3 block(TILE, TILE);
    const dim3 grid((ctx->n + TILE - 1) / TILE, (ctx->m + TILE - 1) / TILE);
    transpose_row_to_col_kernel<<<grid, block, 0, ctx->stream>>>(ctx->d.d_A_row_major, ctx->d.d_A, ctx->m, ctx->n);
    check_cuda(cudaGetLastError(), "transpose kernel launch failed");
}

void set_rng_seed(SVDLowrankCUDAContext* ctx, unsigned long long seed) {
    check_curand(curandSetPseudoRandomGeneratorSeed(ctx->curandGen, seed), "curandSetPseudoRandomGeneratorSeed failed");
}

void run_lowrank_presvd_body(SVDLowrankCUDAContext* ctx, int niter) {
    const int iters = std::max(0, niter);
    const cublasOperation_t op_A = ctx->is_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_A_H = ctx->is_transposed ? CUBLAS_OP_N : CUBLAS_OP_T;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    check_curand(curandGenerateNormal(ctx->curandGen, ctx->d.d_R, static_cast<std::size_t>(ctx->N_work) * ctx->k, 0.0f, 1.0f),
                 "curandGenerateNormal failed");

    check_cublas(cublasSgemm(ctx->cublasH, op_A, CUBLAS_OP_N, ctx->M_work, ctx->k, ctx->N_work, &alpha, ctx->d.d_A, ctx->m,
                             ctx->d.d_R, ctx->N_work, &beta, ctx->d.d_X, ctx->M_work),
                 "Initial cublasSgemm A*R failed");
    check_cusolver(cusolverDnSgeqrf(ctx->cusolverH, ctx->M_work, ctx->k, ctx->d.d_X, ctx->M_work, ctx->d.d_tau_M, ctx->d.d_work,
                                    ctx->lwork, ctx->d.d_devInfo),
                   "cusolverDnSgeqrf initial failed");
    check_cuda(cudaMemcpyAsync(ctx->d.d_Q,
                               ctx->d.d_X,
                               static_cast<std::size_t>(ctx->M_work) * ctx->k * sizeof(float),
                               cudaMemcpyDeviceToDevice,
                               ctx->stream),
               "cudaMemcpyAsync X->Q failed");
    check_cusolver(cusolverDnSorgqr(ctx->cusolverH, ctx->M_work, ctx->k, ctx->k, ctx->d.d_Q, ctx->M_work, ctx->d.d_tau_M,
                                    ctx->d.d_work, ctx->lwork, ctx->d.d_devInfo),
                   "cusolverDnSorgqr initial failed");

    for (int i = 0; i < iters; ++i) {
        check_cublas(cublasSgemm(ctx->cublasH, op_A_H, CUBLAS_OP_N, ctx->N_work, ctx->k, ctx->M_work, &alpha, ctx->d.d_A, ctx->m,
                                 ctx->d.d_Q, ctx->M_work, &beta, ctx->d.d_X_temp, ctx->N_work),
                     "Power iteration forward gemm failed");
        check_cusolver(cusolverDnSgeqrf(ctx->cusolverH,
                                        ctx->N_work,
                                        ctx->k,
                                        ctx->d.d_X_temp,
                                        ctx->N_work,
                                        ctx->d.d_tau_N,
                                        ctx->d.d_work,
                                        ctx->lwork,
                                        ctx->d.d_devInfo),
                       "Power geqrf N failed");
        check_cuda(cudaMemcpyAsync(ctx->d.d_Q_temp,
                                   ctx->d.d_X_temp,
                                   static_cast<std::size_t>(ctx->N_work) * ctx->k * sizeof(float),
                                   cudaMemcpyDeviceToDevice,
                                   ctx->stream),
                   "cudaMemcpyAsync X_temp->Q_temp failed");
        check_cusolver(cusolverDnSorgqr(ctx->cusolverH,
                                        ctx->N_work,
                                        ctx->k,
                                        ctx->k,
                                        ctx->d.d_Q_temp,
                                        ctx->N_work,
                                        ctx->d.d_tau_N,
                                        ctx->d.d_work,
                                        ctx->lwork,
                                        ctx->d.d_devInfo),
                       "Power orgqr N failed");

        check_cublas(cublasSgemm(ctx->cublasH, op_A, CUBLAS_OP_N, ctx->M_work, ctx->k, ctx->N_work, &alpha, ctx->d.d_A, ctx->m,
                                 ctx->d.d_Q_temp, ctx->N_work, &beta, ctx->d.d_X, ctx->M_work),
                     "Power iteration backward gemm failed");
        check_cusolver(cusolverDnSgeqrf(ctx->cusolverH, ctx->M_work, ctx->k, ctx->d.d_X, ctx->M_work, ctx->d.d_tau_M, ctx->d.d_work,
                                        ctx->lwork, ctx->d.d_devInfo),
                       "Power geqrf M failed");
        check_cuda(cudaMemcpyAsync(ctx->d.d_Q,
                                   ctx->d.d_X,
                                   static_cast<std::size_t>(ctx->M_work) * ctx->k * sizeof(float),
                                   cudaMemcpyDeviceToDevice,
                                   ctx->stream),
                   "cudaMemcpyAsync X->Q in power failed");
        check_cusolver(cusolverDnSorgqr(ctx->cusolverH,
                                        ctx->M_work,
                                        ctx->k,
                                        ctx->k,
                                        ctx->d.d_Q,
                                        ctx->M_work,
                                        ctx->d.d_tau_M,
                                        ctx->d.d_work,
                                        ctx->lwork,
                                        ctx->d.d_devInfo),
                       "Power orgqr M failed");
    }

    check_cublas(cublasSgemm(ctx->cublasH, CUBLAS_OP_T, op_A, ctx->k, ctx->N_work, ctx->M_work, &alpha, ctx->d.d_Q, ctx->M_work,
                             ctx->d.d_A, ctx->m, &beta, ctx->d.d_B, ctx->k),
                 "Projection gemm Q^T A failed");
}

void run_lowrank_postsvd_body(SVDLowrankCUDAContext* ctx) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

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
                   "cusolverDnSgesvdj failed");
    check_cublas(cublasSgemm(ctx->cublasH, CUBLAS_OP_N, CUBLAS_OP_N, ctx->M_work, ctx->k, ctx->k, &alpha, ctx->d.d_Q, ctx->M_work,
                             ctx->d.d_U_hat, ctx->k, &beta, ctx->d.d_U_work, ctx->M_work),
                 "Recover U_work gemm failed");
}

void run_lowrank_compute_body(SVDLowrankCUDAContext* ctx, int niter) {
    run_lowrank_presvd_body(ctx, niter);
    run_lowrank_postsvd_body(ctx);
}

[[noreturn]] void abort_with_message(const std::string& message) {
    std::fprintf(stderr, "Fatal error: %s\n", message.c_str());
    std::fflush(stderr);
    std::abort();
}

template <typename BodyFn>
void capture_lowrank_graph_or_throw(SVDLowrankCUDAContext* ctx, int niter, BodyFn&& body) {
    ctx->destroy_graph();

    cudaGraph_t captured_graph = nullptr;
    cudaError_t begin_status = cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeGlobal);
    if (begin_status != cudaSuccess) {
        (void) cudaGetLastError();
        throw std::runtime_error(
            std::string("cudaStreamBeginCapture for lowrank graph failed: ") + cudaGetErrorString(begin_status));
    }

    std::string body_error;
    try {
        body();
    } catch (const std::exception& e) {
        body_error = e.what();
    } catch (...) {
        body_error = "unknown exception";
    }

    cudaError_t end_status = cudaStreamEndCapture(ctx->stream, &captured_graph);
    if (!body_error.empty()) {
        if (captured_graph) {
            cudaGraphDestroy(captured_graph);
        }
        (void) cudaGetLastError();
        throw std::runtime_error(std::string("lowrank pre-SVD graph capture body failed: ") + body_error);
    }
    if (end_status != cudaSuccess || captured_graph == nullptr) {
        if (captured_graph) {
            cudaGraphDestroy(captured_graph);
        }
        (void) cudaGetLastError();
        throw std::runtime_error(
            std::string("cudaStreamEndCapture for lowrank pre-SVD graph failed: ") + cudaGetErrorString(end_status));
    }

    cudaGraphExec_t graph_exec = nullptr;
    cudaError_t inst_status = cudaGraphInstantiate(&graph_exec, captured_graph, nullptr, nullptr, 0);
    if (inst_status != cudaSuccess) {
        cudaGraphDestroy(captured_graph);
        (void) cudaGetLastError();
        throw std::runtime_error(
            std::string("cudaGraphInstantiate for lowrank pre-SVD graph failed: ") + cudaGetErrorString(inst_status));
    }

    ctx->lowrank_graph = captured_graph;
    ctx->lowrank_graph_exec = graph_exec;
    ctx->graph_niter = std::max(0, niter);
}

void run_lowrank_compute(SVDLowrankCUDAContext* ctx, unsigned long long seed) {
    set_rng_seed(ctx, seed);
    if (!ctx->lowrank_graph_exec) {
        throw std::logic_error("lowrank CUDA graph is not captured; initialization must capture successfully");
    }
    if (ctx->graph_niter != ctx->niter) {
        throw std::logic_error("lowrank CUDA graph niter mismatch; recapture is required before execution");
    }
    check_cuda(cudaGraphLaunch(ctx->lowrank_graph_exec, ctx->stream), "cudaGraphLaunch lowrank graph failed");
    run_lowrank_postsvd_body(ctx);
}

void copy_result_to_host(SVDLowrankCUDAContext* ctx) {
    check_cuda(cudaMemcpyAsync(ctx->h_U_work_col_major.data(),
                               ctx->d.d_U_work,
                               static_cast<std::size_t>(ctx->M_work) * ctx->k * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               ctx->stream),
               "cudaMemcpyAsync U_work D2H failed");
    check_cuda(cudaMemcpyAsync(ctx->h_V_hat_col_major.data(),
                               ctx->d.d_V_hat,
                               static_cast<std::size_t>(ctx->N_work) * ctx->k * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               ctx->stream),
               "cudaMemcpyAsync V_hat D2H failed");
    check_cuda(cudaMemcpyAsync(ctx->host_result.S.data(),
                               ctx->d.d_S,
                               static_cast<std::size_t>(ctx->k) * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               ctx->stream),
               "cudaMemcpyAsync S D2H failed");
    check_cuda(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize after D2H failed");

    if (ctx->is_transposed) {
        col_major_to_row_major(ctx->h_V_hat_col_major.data(), ctx->N_work, ctx->k, ctx->host_result.U_row_major.data());
        col_major_to_row_major(ctx->h_U_work_col_major.data(), ctx->M_work, ctx->k, ctx->host_result.V_row_major.data());
    } else {
        col_major_to_row_major(ctx->h_U_work_col_major.data(), ctx->M_work, ctx->k, ctx->host_result.U_row_major.data());
        col_major_to_row_major(ctx->h_V_hat_col_major.data(), ctx->N_work, ctx->k, ctx->host_result.V_row_major.data());
    }
}

void run_svd_pipeline(
    SVDLowrankCUDAContext* ctx,
    const float* A_row_major,
    unsigned long long seed,
    bool copy_result) {
    load_and_transpose_input(ctx, A_row_major);
    run_lowrank_compute(ctx, seed);
    if (copy_result) {
        copy_result_to_host(ctx);
    } else {
        check_cuda(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize after warmup failed");
    }
}

} // namespace

void svd_lowrank_cuda_initialize(int m, int n, int q, int niter, unsigned long long warmup_seed) {
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("m and n must be positive");
    }
    if (q <= 0) {
        throw std::invalid_argument("q must be positive");
    }
    if (niter < 0) {
        throw std::invalid_argument("niter must be non-negative");
    }

    svd_lowrank_cuda_release();

    g_ctx.m = m;
    g_ctx.n = n;
    g_ctx.k = std::max(1, std::min({q, m, n}));
    g_ctx.is_transposed = (m < n);
    g_ctx.M_work = g_ctx.is_transposed ? n : m;
    g_ctx.N_work = g_ctx.is_transposed ? m : n;
    g_ctx.niter = niter;

    g_ctx.h_U_work_col_major.resize(static_cast<std::size_t>(g_ctx.M_work) * g_ctx.k);
    g_ctx.h_V_hat_col_major.resize(static_cast<std::size_t>(g_ctx.N_work) * g_ctx.k);
    g_ctx.host_result.m = m;
    g_ctx.host_result.n = n;
    g_ctx.host_result.k = g_ctx.k;
    g_ctx.host_result.U_row_major.resize(static_cast<std::size_t>(m) * g_ctx.k);
    g_ctx.host_result.S.resize(static_cast<std::size_t>(g_ctx.k));
    g_ctx.host_result.V_row_major.resize(static_cast<std::size_t>(n) * g_ctx.k);

    try {
        check_cuda(cudaStreamCreateWithFlags(&g_ctx.stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed");
        check_cublas(cublasCreate(&g_ctx.cublasH), "cublasCreate failed");
        check_cusolver(cusolverDnCreate(&g_ctx.cusolverH), "cusolverDnCreate failed");
        check_curand(curandCreateGenerator(&g_ctx.curandGen, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator failed");
        check_cusolver(cusolverDnCreateGesvdjInfo(&g_ctx.svdj_params), "cusolverDnCreateGesvdjInfo failed");
        check_cublas(cublasSetStream(g_ctx.cublasH, g_ctx.stream), "cublasSetStream failed");
        check_cusolver(cusolverDnSetStream(g_ctx.cusolverH, g_ctx.stream), "cusolverDnSetStream failed");
        check_curand(curandSetStream(g_ctx.curandGen, g_ctx.stream), "curandSetStream failed");

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

        // Warm up kernels/handles once before graph capture.
        load_and_transpose_input(&g_ctx, nullptr);
        set_rng_seed(&g_ctx, warmup_seed);
        run_lowrank_compute_body(&g_ctx, g_ctx.niter);
        check_cuda(cudaStreamSynchronize(g_ctx.stream), "cudaStreamSynchronize after warmup failed");

        // Capture only the pre-SVD prefix. cusolverDnSgesvdj is not CUDA graph compatible.
        set_rng_seed(&g_ctx, warmup_seed);
        try {
            capture_lowrank_graph_or_throw(
                &g_ctx,
                g_ctx.niter,
                [&]() { run_lowrank_presvd_body(&g_ctx, g_ctx.niter); });
        } catch (const std::exception& presvd_capture_error) {
            abort_with_message(
                std::string("mandatory pre-SVD lowrank graph capture failed: ") + presvd_capture_error.what());
        }

        g_ctx.initialized = true;
    } catch (...) {
        g_ctx.destroy_all();
        throw;
    }
}

void svd_lowrank_cuda_release() {
    g_ctx.destroy_all();
}

const SVDLowrankCPUResult& svd_lowrank_cuda(const float* A_row_major, unsigned long long seed) {
    if (!g_ctx.initialized) {
        throw std::logic_error("svd_lowrank_cuda is not initialized; call svd_lowrank_cuda_initialize(m, n, q, niter) first");
    }
    if (A_row_major == nullptr) {
        throw std::invalid_argument("A_row_major must not be null");
    }
    run_svd_pipeline(&g_ctx, A_row_major, seed, true);
    return g_ctx.host_result;
}
