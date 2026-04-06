#include <svd_lowrank_cuda/svd_lowrank_cuda.hpp>

#include "svd_lowrank_cuda_utils.cuh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

using svd_lowrank_cuda_detail::OwnedCudaGraph;
using svd_lowrank_cuda_detail::begin_graph_capture;
using svd_lowrank_cuda_detail::check_cublas;
using svd_lowrank_cuda_detail::check_cuda;
using svd_lowrank_cuda_detail::check_curand;
using svd_lowrank_cuda_detail::check_cusolver;
using svd_lowrank_cuda_detail::launch_col_to_row_transpose;
using svd_lowrank_cuda_detail::launch_row_to_col_transpose;
using svd_lowrank_cuda_detail::matrix_elements;

void SVDLowrankCUDADevicePtrs::free_all() noexcept {
    if (d_A) cudaFree(d_A);
    if (d_R) cudaFree(d_R);
    if (d_X) cudaFree(d_X);
    if (d_X_temp) cudaFree(d_X_temp);
    if (d_tau_M) cudaFree(d_tau_M);
    if (d_tau_N) cudaFree(d_tau_N);
    if (d_work) cudaFree(d_work);
    if (d_B) cudaFree(d_B);
    if (d_U_hat) cudaFree(d_U_hat);
    if (d_S) cudaFree(d_S);
    if (d_V_hat) cudaFree(d_V_hat);
    if (d_U_work) cudaFree(d_U_work);
    if (d_devInfo) cudaFree(d_devInfo);

    d_A = nullptr;
    d_R = nullptr;
    d_X = nullptr;
    d_X_temp = nullptr;
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

void SVDLowrankCUDAContext::destroy_graph() noexcept {
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

void SVDLowrankCUDAContext::destroy_all() noexcept {
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
    initialized = false;
}

namespace {

constexpr float kAlpha = 1.0f;
constexpr float kBeta = 0.0f;

SVDLowrankCUDAContext g_ctx;

void validate_initialize_args(int m, int n, int q, int niter) {
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("m and n must be positive");
    }
    if (q <= 0) {
        throw std::invalid_argument("q must be positive");
    }
    if (niter < 0) {
        throw std::invalid_argument("niter must be non-negative");
    }
}

void validate_run_args(
    const float* d_A_row_major,
    float* d_U_row_major,
    float* d_S,
    float* d_V_row_major) {
    if (d_A_row_major == nullptr) {
        throw std::invalid_argument("d_A_row_major must not be null");
    }
    if (d_U_row_major == nullptr || d_S == nullptr || d_V_row_major == nullptr) {
        throw std::invalid_argument("output device pointers must not be null");
    }
}

void configure_context(SVDLowrankCUDAContext* ctx, int m, int n, int q, int niter) {
    ctx->m = m;
    ctx->n = n;
    ctx->k = std::max(1, std::min({q, m, n}));
    ctx->is_transposed = (m < n);
    ctx->M_work = ctx->is_transposed ? n : m;
    ctx->N_work = ctx->is_transposed ? m : n;
    ctx->niter = niter;
}

void create_runtime_handles(SVDLowrankCUDAContext* ctx) {
    check_cuda(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed");
    check_cublas(cublasCreate(&ctx->cublasH), "cublasCreate failed");
    check_cusolver(cusolverDnCreate(&ctx->cusolverH), "cusolverDnCreate failed");
    check_curand(curandCreateGenerator(&ctx->curandGen, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator failed");
    check_cusolver(cusolverDnCreateGesvdjInfo(&ctx->svdj_params), "cusolverDnCreateGesvdjInfo failed");

    check_cublas(cublasSetStream(ctx->cublasH, ctx->stream), "cublasSetStream failed");
    check_cusolver(cusolverDnSetStream(ctx->cusolverH, ctx->stream), "cusolverDnSetStream failed");
    check_curand(curandSetStream(ctx->curandGen, ctx->stream), "curandSetStream failed");
}

void allocate_device_buffers(SVDLowrankCUDAContext* ctx) {
    check_cuda(cudaMalloc(&ctx->d.d_A, matrix_elements(ctx->m, ctx->n) * sizeof(float)), "cudaMalloc d_A failed");
    check_cuda(cudaMalloc(&ctx->d.d_R, matrix_elements(ctx->N_work, ctx->k) * sizeof(float)), "cudaMalloc d_R failed");
    check_cuda(cudaMalloc(&ctx->d.d_X, matrix_elements(ctx->M_work, ctx->k) * sizeof(float)), "cudaMalloc d_X failed");
    check_cuda(cudaMalloc(&ctx->d.d_X_temp, matrix_elements(ctx->N_work, ctx->k) * sizeof(float)), "cudaMalloc d_X_temp failed");
    check_cuda(cudaMalloc(&ctx->d.d_tau_M, static_cast<std::size_t>(ctx->k) * sizeof(float)), "cudaMalloc d_tau_M failed");
    check_cuda(cudaMalloc(&ctx->d.d_tau_N, static_cast<std::size_t>(ctx->k) * sizeof(float)), "cudaMalloc d_tau_N failed");
    check_cuda(cudaMalloc(&ctx->d.d_B, matrix_elements(ctx->k, ctx->N_work) * sizeof(float)), "cudaMalloc d_B failed");
    check_cuda(cudaMalloc(&ctx->d.d_U_hat, matrix_elements(ctx->k, ctx->k) * sizeof(float)), "cudaMalloc d_U_hat failed");
    check_cuda(cudaMalloc(&ctx->d.d_S, static_cast<std::size_t>(ctx->k) * sizeof(float)), "cudaMalloc d_S failed");
    check_cuda(cudaMalloc(&ctx->d.d_V_hat, matrix_elements(ctx->N_work, ctx->k) * sizeof(float)), "cudaMalloc d_V_hat failed");
    check_cuda(cudaMalloc(&ctx->d.d_U_work, matrix_elements(ctx->M_work, ctx->k) * sizeof(float)), "cudaMalloc d_U_work failed");
    check_cuda(cudaMalloc(&ctx->d.d_devInfo, sizeof(int)), "cudaMalloc d_devInfo failed");
}

void initialize_workspace_and_lwork(SVDLowrankCUDAContext* ctx) {
    int lwork_geqrf_M = 0;
    int lwork_orgqr_M = 0;
    int lwork_geqrf_N = 0;
    int lwork_orgqr_N = 0;
    int lwork_gesvdj = 0;

    check_cusolver(
        cusolverDnSgeqrf_bufferSize(ctx->cusolverH, ctx->M_work, ctx->k, ctx->d.d_X, ctx->M_work, &lwork_geqrf_M),
        "cusolverDnSgeqrf_bufferSize M failed");
    check_cusolver(
        cusolverDnSorgqr_bufferSize(
            ctx->cusolverH,
            ctx->M_work,
            ctx->k,
            ctx->k,
            ctx->d.d_X,
            ctx->M_work,
            ctx->d.d_tau_M,
            &lwork_orgqr_M),
        "cusolverDnSorgqr_bufferSize M failed");
    check_cusolver(
        cusolverDnSgeqrf_bufferSize(
            ctx->cusolverH,
            ctx->N_work,
            ctx->k,
            ctx->d.d_X_temp,
            ctx->N_work,
            &lwork_geqrf_N),
        "cusolverDnSgeqrf_bufferSize N failed");
    check_cusolver(
        cusolverDnSorgqr_bufferSize(
            ctx->cusolverH,
            ctx->N_work,
            ctx->k,
            ctx->k,
            ctx->d.d_X_temp,
            ctx->N_work,
            ctx->d.d_tau_N,
            &lwork_orgqr_N),
        "cusolverDnSorgqr_bufferSize N failed");
    check_cusolver(
        cusolverDnSgesvdj_bufferSize(
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

void set_rng_seed(SVDLowrankCUDAContext* ctx, unsigned long long seed) {
    check_curand(curandSetPseudoRandomGeneratorSeed(ctx->curandGen, seed), "curandSetPseudoRandomGeneratorSeed failed");
}

void load_and_transpose_input(SVDLowrankCUDAContext* ctx, const float* d_A_row_major) {
    const std::size_t matrix_bytes = matrix_elements(ctx->m, ctx->n) * sizeof(float);

    if (d_A_row_major == nullptr) {
        check_cuda(cudaMemsetAsync(ctx->d.d_A, 0, matrix_bytes, ctx->stream), "cudaMemsetAsync d_A failed");
        return;
    }

    launch_row_to_col_transpose(
        d_A_row_major,
        ctx->d.d_A,
        ctx->m,
        ctx->n,
        ctx->stream,
        "transpose row-to-col kernel launch failed");
}

void factor_to_orthonormal_basis(
    SVDLowrankCUDAContext* ctx,
    float* d_factor,
    int rows,
    float* d_tau,
    const char* geqrf_msg,
    const char* orgqr_msg) {
    check_cusolver(
        cusolverDnSgeqrf(
            ctx->cusolverH,
            rows,
            ctx->k,
            d_factor,
            rows,
            d_tau,
            ctx->d.d_work,
            ctx->lwork,
            ctx->d.d_devInfo),
        geqrf_msg);
    check_cusolver(
        cusolverDnSorgqr(
            ctx->cusolverH,
            rows,
            ctx->k,
            ctx->k,
            d_factor,
            rows,
            d_tau,
            ctx->d.d_work,
            ctx->lwork,
            ctx->d.d_devInfo),
        orgqr_msg);
}

void run_lowrank_presvd_body(SVDLowrankCUDAContext* ctx, int niter) {
    const int iters = std::max(0, niter);
    const cublasOperation_t op_A = ctx->is_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_A_H = ctx->is_transposed ? CUBLAS_OP_N : CUBLAS_OP_T;

    check_curand(
        curandGenerateNormal(ctx->curandGen, ctx->d.d_R, matrix_elements(ctx->N_work, ctx->k), 0.0f, 1.0f),
        "curandGenerateNormal failed");

    check_cublas(
        cublasSgemm(
            ctx->cublasH,
            op_A,
            CUBLAS_OP_N,
            ctx->M_work,
            ctx->k,
            ctx->N_work,
            &kAlpha,
            ctx->d.d_A,
            ctx->m,
            ctx->d.d_R,
            ctx->N_work,
            &kBeta,
            ctx->d.d_X,
            ctx->M_work),
        "Initial cublasSgemm A*R failed");
    factor_to_orthonormal_basis(
        ctx,
        ctx->d.d_X,
        ctx->M_work,
        ctx->d.d_tau_M,
        "cusolverDnSgeqrf initial failed",
        "cusolverDnSorgqr initial failed");

    for (int i = 0; i < iters; ++i) {
        check_cublas(
            cublasSgemm(
                ctx->cublasH,
                op_A_H,
                CUBLAS_OP_N,
                ctx->N_work,
                ctx->k,
                ctx->M_work,
                &kAlpha,
                ctx->d.d_A,
                ctx->m,
                ctx->d.d_X,
                ctx->M_work,
                &kBeta,
                ctx->d.d_X_temp,
                ctx->N_work),
            "Power iteration forward gemm failed");
        factor_to_orthonormal_basis(
            ctx,
            ctx->d.d_X_temp,
            ctx->N_work,
            ctx->d.d_tau_N,
            "Power geqrf N failed",
            "Power orgqr N failed");

        check_cublas(
            cublasSgemm(
                ctx->cublasH,
                op_A,
                CUBLAS_OP_N,
                ctx->M_work,
                ctx->k,
                ctx->N_work,
                &kAlpha,
                ctx->d.d_A,
                ctx->m,
                ctx->d.d_X_temp,
                ctx->N_work,
                &kBeta,
                ctx->d.d_X,
                ctx->M_work),
            "Power iteration backward gemm failed");
        factor_to_orthonormal_basis(
            ctx,
            ctx->d.d_X,
            ctx->M_work,
            ctx->d.d_tau_M,
            "Power geqrf M failed",
            "Power orgqr M failed");
    }

    check_cublas(
        cublasSgemm(
            ctx->cublasH,
            CUBLAS_OP_T,
            op_A,
            ctx->k,
            ctx->N_work,
            ctx->M_work,
            &kAlpha,
            ctx->d.d_X,
            ctx->M_work,
            ctx->d.d_A,
            ctx->m,
            &kBeta,
            ctx->d.d_B,
            ctx->k),
        "Projection gemm Q^T A failed");
}

void run_lowrank_postsvd_body(SVDLowrankCUDAContext* ctx) {
    check_cusolver(
        cusolverDnSgesvdj(
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
    check_cublas(
        cublasSgemm(
            ctx->cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            ctx->M_work,
            ctx->k,
            ctx->k,
            &kAlpha,
            ctx->d.d_X,
            ctx->M_work,
            ctx->d.d_U_hat,
            ctx->k,
            &kBeta,
            ctx->d.d_U_work,
            ctx->M_work),
        "Recover U_work gemm failed");
}

void run_lowrank_compute_body(SVDLowrankCUDAContext* ctx, int niter) {
    run_lowrank_presvd_body(ctx, niter);
    run_lowrank_postsvd_body(ctx);
}

void warmup_lowrank_compute(SVDLowrankCUDAContext* ctx, unsigned long long warmup_seed) {
    load_and_transpose_input(ctx, nullptr);
    set_rng_seed(ctx, warmup_seed);
    run_lowrank_compute_body(ctx, ctx->niter);
    check_cuda(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize after warmup failed");
}

void capture_lowrank_graph(SVDLowrankCUDAContext* ctx, int niter) {
    ctx->destroy_graph();

    auto capture = begin_graph_capture(
        ctx->stream,
        cudaStreamCaptureModeGlobal,
        "cudaStreamBeginCapture for lowrank graph failed");
    run_lowrank_presvd_body(ctx, niter);

    const cudaError_t end_status = cudaStreamEndCapture(ctx->stream, &capture.graph);
    capture.active = false;
    check_cuda(end_status, "cudaStreamEndCapture for lowrank pre-SVD graph failed");

    OwnedCudaGraph graph{capture.graph};
    cudaGraphExec_t graph_exec = nullptr;
    check_cuda(
        cudaGraphInstantiate(&graph_exec, graph.graph, nullptr, nullptr, 0),
        "cudaGraphInstantiate for lowrank pre-SVD graph failed");

    ctx->lowrank_graph = graph.release();
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

void copy_result_to_device(
    SVDLowrankCUDAContext* ctx,
    float* d_U_row_major,
    float* d_S,
    float* d_V_row_major) {
    check_cuda(
        cudaMemcpyAsync(
            d_S,
            ctx->d.d_S,
            static_cast<std::size_t>(ctx->k) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            ctx->stream),
        "cudaMemcpyAsync S D2D failed");

    if (ctx->is_transposed) {
        launch_col_to_row_transpose(
            ctx->d.d_V_hat,
            d_U_row_major,
            ctx->N_work,
            ctx->k,
            ctx->stream,
            "transpose V_hat->U kernel launch failed");
        launch_col_to_row_transpose(
            ctx->d.d_U_work,
            d_V_row_major,
            ctx->M_work,
            ctx->k,
            ctx->stream,
            "transpose U_work->V kernel launch failed");
        return;
    }

    launch_col_to_row_transpose(
        ctx->d.d_U_work,
        d_U_row_major,
        ctx->M_work,
        ctx->k,
        ctx->stream,
        "transpose U_work->U kernel launch failed");
    launch_col_to_row_transpose(
        ctx->d.d_V_hat,
        d_V_row_major,
        ctx->N_work,
        ctx->k,
        ctx->stream,
        "transpose V_hat->V kernel launch failed");
}

void run_svd_pipeline(
    SVDLowrankCUDAContext* ctx,
    const float* d_A_row_major,
    float* d_U_row_major,
    float* d_S,
    float* d_V_row_major,
    unsigned long long seed) {
    load_and_transpose_input(ctx, d_A_row_major);
    run_lowrank_compute(ctx, seed);
    copy_result_to_device(ctx, d_U_row_major, d_S, d_V_row_major);
    check_cuda(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize after device output failed");
}

} // namespace

void svd_lowrank_cuda_initialize(int m, int n, int q, int niter, unsigned long long warmup_seed) {
    validate_initialize_args(m, n, q, niter);

    svd_lowrank_cuda_release();
    configure_context(&g_ctx, m, n, q, niter);

    try {
        create_runtime_handles(&g_ctx);
        allocate_device_buffers(&g_ctx);
        initialize_workspace_and_lwork(&g_ctx);

        warmup_lowrank_compute(&g_ctx, warmup_seed);

        // Capture only the pre-SVD prefix. cusolverDnSgesvdj is not CUDA graph compatible.
        set_rng_seed(&g_ctx, warmup_seed);
        capture_lowrank_graph(&g_ctx, g_ctx.niter);

        g_ctx.initialized = true;
    } catch (...) {
        g_ctx.destroy_all();
        throw;
    }
}

void svd_lowrank_cuda_release() {
    g_ctx.destroy_all();
}

void svd_lowrank_cuda(
    const float* d_A_row_major,
    float* d_U_row_major,
    float* d_S,
    float* d_V_row_major,
    unsigned long long seed) {
    if (!g_ctx.initialized) {
        throw std::logic_error("svd_lowrank_cuda is not initialized; call svd_lowrank_cuda_initialize(m, n, q, niter) first");
    }

    validate_run_args(d_A_row_major, d_U_row_major, d_S, d_V_row_major);
    run_svd_pipeline(&g_ctx, d_A_row_major, d_U_row_major, d_S, d_V_row_major, seed);
}
