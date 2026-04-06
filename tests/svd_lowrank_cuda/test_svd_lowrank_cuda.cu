#include <svd_lowrank_cuda/svd_lowrank_cuda.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct HostSVDResult {
    int m = 0;
    int n = 0;
    int k = 0;
    std::vector<float> U_row_major;
    std::vector<float> S;
    std::vector<float> V_row_major;
};

struct ScopedDeviceBuffers {
    float *d_A = nullptr;
    float *d_U = nullptr;
    float *d_S = nullptr;
    float *d_V = nullptr;

    ~ScopedDeviceBuffers() {
        cudaFree(d_A);
        cudaFree(d_U);
        cudaFree(d_S);
        cudaFree(d_V);
    }
};

void check_cuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

std::vector<float> matmul_row_major(const std::vector<float>& A, int m, int k, const std::vector<float>& B, int n) {
    std::vector<float> C(static_cast<std::size_t>(m) * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            const float a = A[static_cast<std::size_t>(i) * k + p];
            for (int j = 0; j < n; ++j) {
                C[static_cast<std::size_t>(i) * n + j] += a * B[static_cast<std::size_t>(p) * n + j];
            }
        }
    }
    return C;
}

std::vector<float> transpose_row_major(const std::vector<float>& A, int rows, int cols) {
    std::vector<float> At(static_cast<std::size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            At[static_cast<std::size_t>(c) * rows + r] = A[static_cast<std::size_t>(r) * cols + c];
        }
    }
    return At;
}

std::vector<float> make_low_rank_matrix(int m, int n, int rank, unsigned long long seed) {
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> L(static_cast<std::size_t>(m) * rank);
    std::vector<float> R(static_cast<std::size_t>(rank) * n);
    for (float& x : L) {
        x = dist(gen);
    }
    for (float& x : R) {
        x = dist(gen);
    }

    return matmul_row_major(L, m, rank, R, n);
}

std::vector<float> reconstruct_from_svd(const HostSVDResult& svd) {
    std::vector<float> US(static_cast<std::size_t>(svd.m) * svd.k, 0.0f);
    for (int i = 0; i < svd.m; ++i) {
        for (int j = 0; j < svd.k; ++j) {
            US[static_cast<std::size_t>(i) * svd.k + j] = svd.U_row_major[static_cast<std::size_t>(i) * svd.k + j] * svd.S[j];
        }
    }

    const auto Vt = transpose_row_major(svd.V_row_major, svd.n, svd.k);
    return matmul_row_major(US, svd.m, svd.k, Vt, svd.n);
}

void compute_mae_mse(const std::vector<float>& A, const std::vector<float>& B, double* mae, double* mse) {
    const std::size_t size = A.size();
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        const double diff = static_cast<double>(A[i]) - static_cast<double>(B[i]);
        sum_abs += std::abs(diff);
        sum_sq += diff * diff;
    }
    *mae = sum_abs / static_cast<double>(size);
    *mse = sum_sq / static_cast<double>(size);
}

} // namespace

int main() {
    try {
        const std::vector<int> ms = {512, 1024, 4096};
        const std::vector<int> ns = {4096, 8192};

        constexpr int q = 64;
        constexpr int niter = 2;
        constexpr unsigned long long base_seed = 20260323ULL;

        std::cout << "svd_lowrank_cuda benchmark" << '\n';
        std::cout << "q=" << q << ", niter=" << niter << '\n';
        std::cout << std::fixed << std::setprecision(6);

        for (int m : ms) {
            for (int n : ns) {
                const int true_rank = std::min({q / 2, m, n});
                const unsigned long long seed = base_seed + static_cast<unsigned long long>(m) * 100000ULL + static_cast<unsigned long long>(n);

                std::cout << "\nCase m=" << m << ", n=" << n << ", true_rank=" << true_rank << '\n';

                auto A = make_low_rank_matrix(m, n, true_rank, seed);
                ScopedDeviceBuffers buffers;
                HostSVDResult svd;
                svd.m = m;
                svd.n = n;
                svd.k = q;
                svd.U_row_major.resize(static_cast<std::size_t>(m) * q);
                svd.S.resize(static_cast<std::size_t>(q));
                svd.V_row_major.resize(static_cast<std::size_t>(n) * q);

                check_cuda(cudaMalloc(&buffers.d_A, static_cast<std::size_t>(m) * n * sizeof(float)), "cudaMalloc d_A failed");
                check_cuda(cudaMalloc(&buffers.d_U, static_cast<std::size_t>(m) * q * sizeof(float)), "cudaMalloc d_U failed");
                check_cuda(cudaMalloc(&buffers.d_S, static_cast<std::size_t>(q) * sizeof(float)), "cudaMalloc d_S failed");
                check_cuda(cudaMalloc(&buffers.d_V, static_cast<std::size_t>(n) * q * sizeof(float)), "cudaMalloc d_V failed");
                check_cuda(cudaMemcpyAsync(
                               buffers.d_A,
                               A.data(),
                               static_cast<std::size_t>(m) * n * sizeof(float),
                               cudaMemcpyHostToDevice),
                           "cudaMemcpyAsync A H2D failed");

                svd_lowrank_cuda_initialize(m, n, q, niter, seed + 1ULL);

                const auto wall_start = std::chrono::high_resolution_clock::now();
                svd_lowrank_cuda(buffers.d_A, buffers.d_U, buffers.d_S, buffers.d_V, seed + 7ULL);
                const auto wall_end = std::chrono::high_resolution_clock::now();
                const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
                check_cuda(cudaMemcpyAsync(
                               svd.U_row_major.data(),
                               buffers.d_U,
                               svd.U_row_major.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                           "cudaMemcpyAsync U D2H failed");
                check_cuda(cudaMemcpyAsync(
                               svd.S.data(),
                               buffers.d_S,
                               svd.S.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                           "cudaMemcpyAsync S D2H failed");
                check_cuda(cudaMemcpyAsync(
                               svd.V_row_major.data(),
                               buffers.d_V,
                               svd.V_row_major.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                           "cudaMemcpyAsync V D2H failed");

                const auto A_hat = reconstruct_from_svd(svd);
                double mae = 0.0;
                double mse = 0.0;
                compute_mae_mse(A, A_hat, &mae, &mse);

                std::cout << "k=" << svd.k
                          << " | MAE=" << mae
                          << " | MSE=" << mse
                          << " | wall_ms=" << wall_ms
                          << '\n';

                svd_lowrank_cuda_release();
            }
        }

        return 0;
    } catch (const std::exception& e) {
        svd_lowrank_cuda_release();
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
