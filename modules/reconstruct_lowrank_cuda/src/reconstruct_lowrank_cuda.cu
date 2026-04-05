#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <reconstruct_lowrank_cuda/reconstruct_lowrank_cuda.hpp>

namespace {

using reconstruct_lowrank_cuda_detail::check_cublas;
using reconstruct_lowrank_cuda_detail::check_cuda;

void run_reconstruct_lowrank_cuda(ReconstructLowrankCUDAContext* ctx) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // U is row-major (m x k). Treat it as U^T (k x m) in column-major and scale by diag(S).
    check_cublas(cublasSdgmm(
        ctx->cublas_handler,
        CUBLAS_SIDE_LEFT,
        ctx->k,
        ctx->m,
        ctx->cuda_ptrs.d_U,
        ctx->k,
        ctx->cuda_ptrs.d_S,
        1,
        ctx->cuda_ptrs.d_U,
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
        ctx->cuda_ptrs.d_V,
        ctx->k,
        ctx->cuda_ptrs.d_U,
        ctx->k,
        &beta,
        ctx->cuda_ptrs.d_result,
        ctx->n),
        "cublasSgemm failed");
}

void warmup(ReconstructLowrankCUDAContext* ctx, unsigned long long warmup_seed) {
    (void) warmup_seed;

    std::vector<float> U(static_cast<std::size_t>(ctx->m) * ctx->k, 1.0f);
    std::vector<float> S(static_cast<std::size_t>(ctx->k), 1.0f);
    std::vector<float> V(static_cast<std::size_t>(ctx->n) * ctx->k, 1.0f);

    check_cuda(cudaMemcpyAsync(ctx->cuda_ptrs.d_U,
                          U.data(),
                          static_cast<std::size_t>(ctx->m) * ctx->k * sizeof(float),
                          cudaMemcpyHostToDevice),
               "warmup cudaMemcpy U failed");
    check_cuda(cudaMemcpyAsync(ctx->cuda_ptrs.d_S,
                          S.data(),
                          static_cast<std::size_t>(ctx->k) * sizeof(float),
                          cudaMemcpyHostToDevice),
               "warmup cudaMemcpy S failed");
    check_cuda(cudaMemcpyAsync(ctx->cuda_ptrs.d_V,
                          V.data(),
                          static_cast<std::size_t>(ctx->n) * ctx->k * sizeof(float),
                          cudaMemcpyHostToDevice),
               "warmup cudaMemcpy V failed");

    run_reconstruct_lowrank_cuda(ctx);
    check_cuda(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize failed");
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
    ctx->host_ptrs.h_result.resize(static_cast<std::size_t>(m) * n);

    try {
        check_cublas(cublasCreate(&ctx->cublas_handler), "cublasCreate failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_U, static_cast<std::size_t>(m) * k * sizeof(float)), "cudaMalloc d_U failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_S, static_cast<std::size_t>(k) * sizeof(float)), "cudaMalloc d_S failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_V, static_cast<std::size_t>(n) * k * sizeof(float)), "cudaMalloc d_V failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_result, static_cast<std::size_t>(m) * n * sizeof(float)), "cudaMalloc d_result failed");
        warmup(ctx, warmup_seed);
        ctx->initialized = true;
    } catch (...) {
        ctx->destroy_all();
        throw;
    }
}

void reconstruct_lowrank_cuda(
    ReconstructLowrankCUDAContext* ctx,
    const std::vector<float>& U_row_major,
    const std::vector<float>& S,
    const std::vector<float>& V_row_major,
    std::vector<float>* out_row_major) {
    if (ctx == nullptr) {
        throw std::invalid_argument("ctx must not be null");
    }
    if (!ctx->initialized) {
        throw std::runtime_error("context is not initialized");
    }
    if (out_row_major == nullptr) {
        throw std::invalid_argument("out_row_major must not be null");
    }

    const std::size_t m = static_cast<std::size_t>(ctx->m);
    const std::size_t n = static_cast<std::size_t>(ctx->n);
    const std::size_t k = static_cast<std::size_t>(ctx->k);

    if (U_row_major.size() != m * k) {
        throw std::invalid_argument("U_row_major size mismatch");
    }
    if (S.size() != k) {
        throw std::invalid_argument("S size mismatch");
    }
    if (V_row_major.size() != n * k) {
        throw std::invalid_argument("V_row_major size mismatch");
    }

    check_cuda(cudaMemcpyAsync(ctx->cuda_ptrs.d_U, U_row_major.data(), m * k * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy U failed");
    check_cuda(cudaMemcpyAsync(ctx->cuda_ptrs.d_S, S.data(), k * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy S failed");
    check_cuda(cudaMemcpyAsync(ctx->cuda_ptrs.d_V, V_row_major.data(), n * k * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy V failed");

    run_reconstruct_lowrank_cuda(ctx);

    out_row_major->resize(m * n);
    check_cuda(cudaMemcpyAsync(out_row_major->data(), ctx->cuda_ptrs.d_result, m * n * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy result failed");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}
