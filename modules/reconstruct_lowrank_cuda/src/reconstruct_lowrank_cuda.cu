#include <cstddef>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <reconstruct_lowrank_cuda/reconstruct_lowrank_cuda.cuh>

namespace {

using reconstruct_lowrank_cuda_detail::check_cublas;
using reconstruct_lowrank_cuda_detail::check_cuda;

__global__ void fill_kernel(float* dst, std::size_t count, float value) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = value;
    }
}

void run_reconstruct_lowrank_cuda(
    ReconstructLowrankCUDAContext* ctx,
    const float* d_U_row_major,
    const float* d_S,
    const float* d_V_row_major,
    float* d_out_row_major) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // U is row-major (m x k). Treat it as U^T (k x m) in column-major and scale by diag(S).
    check_cublas(cublasSdgmm(
        ctx->cublas_handler,
        CUBLAS_SIDE_LEFT,
        ctx->k,
        ctx->m,
        d_U_row_major,
        ctx->k,
        d_S,
        1,
        ctx->cuda_ptrs.d_scaled_u,
        ctx->k),
        "cublasSdgmm failed");

    // Reconstruct A = U * diag(S) * V^T in row-major via A^T = V * (U*diag(S))^T.
    // V is row-major (n x k), interpreted as column-major (k x n), then transposed.
    check_cublas(cublasSgemm(
        ctx->cublas_handler,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        ctx->n,
        ctx->m,
        ctx->k,
        &alpha,
        d_V_row_major,
        ctx->k,
        ctx->cuda_ptrs.d_scaled_u,
        ctx->k,
        &beta,
        d_out_row_major,
        ctx->n),
        "cublasSgemm failed");
}

void warmup(ReconstructLowrankCUDAContext* ctx, unsigned long long warmup_seed) {
    (void) warmup_seed;

    float* d_U = nullptr;
    float* d_S = nullptr;
    float* d_V = nullptr;
    float* d_out = nullptr;

    try {
        const std::size_t u_count = static_cast<std::size_t>(ctx->m) * ctx->k;
        const std::size_t s_count = static_cast<std::size_t>(ctx->k);
        const std::size_t v_count = static_cast<std::size_t>(ctx->n) * ctx->k;
        const std::size_t out_count = static_cast<std::size_t>(ctx->m) * ctx->n;

        check_cuda(cudaMalloc(&d_U, u_count * sizeof(float)), "warmup cudaMalloc d_U failed");
        check_cuda(cudaMalloc(&d_S, s_count * sizeof(float)), "warmup cudaMalloc d_S failed");
        check_cuda(cudaMalloc(&d_V, v_count * sizeof(float)), "warmup cudaMalloc d_V failed");
        check_cuda(cudaMalloc(&d_out, out_count * sizeof(float)), "warmup cudaMalloc d_out failed");

        constexpr int kBlockSize = 256;
        const auto launch_fill = [&](float* ptr, std::size_t count) {
            const int grid = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
            fill_kernel<<<grid, kBlockSize, 0, ctx->stream>>>(ptr, count, 1.0f);
            check_cuda(cudaGetLastError(), "warmup fill kernel launch failed");
        };

        launch_fill(d_U, u_count);
        launch_fill(d_S, s_count);
        launch_fill(d_V, v_count);
        launch_fill(d_out, out_count);

        run_reconstruct_lowrank_cuda(ctx, d_U, d_S, d_V, d_out);
        check_cuda(cudaStreamSynchronize(ctx->stream), "warmup cudaStreamSynchronize failed");
    } catch (...) {
        cudaFree(d_U);
        cudaFree(d_S);
        cudaFree(d_V);
        cudaFree(d_out);
        throw;
    }

    cudaFree(d_U);
    cudaFree(d_S);
    cudaFree(d_V);
    cudaFree(d_out);
}

}  // namespace

void reconstruct_lowrank_cuda_release(ReconstructLowrankCUDAContext* ctx) noexcept {
    if (ctx == nullptr) {
        return;
    }
    ctx->destroy_all();
}

void reconstruct_lowrank_cuda_initialize(
    ReconstructLowrankCUDAContext* ctx,
    int m,
    int n,
    int k,
    unsigned long long warmup_seed) {
    if (ctx == nullptr) {
        throw std::invalid_argument("ctx must not be null");
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        throw std::invalid_argument("m, n and k must be positive");
    }

    reconstruct_lowrank_cuda_release(ctx);

    ctx->m = m;
    ctx->n = n;
    ctx->k = k;

    try {
        check_cuda(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed");
        check_cublas(cublasCreate(&ctx->cublas_handler), "cublasCreate failed");
        check_cublas(cublasSetStream(ctx->cublas_handler, ctx->stream), "cublasSetStream failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_scaled_u, static_cast<std::size_t>(m) * k * sizeof(float)),
                   "cudaMalloc d_scaled_u failed");
        warmup(ctx, warmup_seed);
        ctx->initialized = true;
    } catch (...) {
        ctx->destroy_all();
        throw;
    }
}

void reconstruct_lowrank_cuda(
    ReconstructLowrankCUDAContext* ctx,
    const float* d_U_row_major,
    const float* d_S,
    const float* d_V_row_major,
    float* d_out_row_major) {
    if (ctx == nullptr) {
        throw std::invalid_argument("ctx must not be null");
    }
    if (!ctx->initialized) {
        throw std::runtime_error("context is not initialized");
    }
    if (d_U_row_major == nullptr || d_S == nullptr || d_V_row_major == nullptr || d_out_row_major == nullptr) {
        throw std::invalid_argument("device pointers must not be null");
    }

    run_reconstruct_lowrank_cuda(ctx, d_U_row_major, d_S, d_V_row_major, d_out_row_major);
    check_cuda(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize failed");
}
