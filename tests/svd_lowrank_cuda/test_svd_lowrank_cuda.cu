#include <svd_lowrank_cuda/svd_lowrank_cuda.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

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

std::vector<float> reconstruct_from_svd(const SVDLowrankCPUResult& svd) {
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

                svd_lowrank_cuda_initialize(m, n, q, seed + 1ULL);

                SVDLowrankTimings device_timings;
                const auto wall_start = std::chrono::high_resolution_clock::now();
                const SVDLowrankCPUResult& svd = svd_lowrank_cuda(A.data(), niter, seed + 7ULL, &device_timings);
                const auto wall_end = std::chrono::high_resolution_clock::now();
                const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

                const auto A_hat = reconstruct_from_svd(svd);
                double mae = 0.0;
                double mse = 0.0;
                compute_mae_mse(A, A_hat, &mae, &mse);

                std::cout << "k=" << svd.k
                          << " | MAE=" << mae
                          << " | MSE=" << mse
                          << " | wall_ms=" << wall_ms
                          << " | h2d_ms=" << device_timings.h2d_ms
                          << " | transpose_ms=" << device_timings.transpose_ms
                          << " | compute_ms=" << device_timings.compute_ms
                          << " | d2h_ms=" << device_timings.d2h_ms
                          << " | total_ms=" << device_timings.total_ms
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
