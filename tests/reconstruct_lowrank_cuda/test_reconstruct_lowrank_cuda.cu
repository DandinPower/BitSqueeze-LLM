#include <reconstruct_lowrank_cuda/reconstruct_lowrank_cuda.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kM = 512;
constexpr int kK = 128;
constexpr int kN = 8192;
constexpr int kWarmupIters = 5;
constexpr int kMeasureIters = 30;
constexpr unsigned long long kSeed = 20260405ULL;

struct ScopedDeviceBuffers {
    float *d_U = nullptr;
    float *d_S = nullptr;
    float *d_V = nullptr;
    float *d_out = nullptr;

    ~ScopedDeviceBuffers() {
        cudaFree(d_U);
        cudaFree(d_S);
        cudaFree(d_V);
        cudaFree(d_out);
    }
};

void check_cuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

float compute_reference_entry(const std::vector<float>& U,
                              const std::vector<float>& S,
                              const std::vector<float>& V,
                              int row,
                              int col) {
    float acc = 0.0f;
    for (int p = 0; p < kK; ++p) {
        acc += U[static_cast<std::size_t>(row) * kK + p] * S[p] * V[static_cast<std::size_t>(col) * kK + p];
    }
    return acc;
}

} // namespace

int main() {
    ReconstructLowrankCUDAContext ctx;
    ScopedDeviceBuffers buffers;

    try {
        std::mt19937 gen(static_cast<std::mt19937::result_type>(kSeed));
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> row_dist(0, kM - 1);
        std::uniform_int_distribution<int> col_dist(0, kN - 1);

        std::vector<float> U(static_cast<std::size_t>(kM) * kK);
        std::vector<float> S(static_cast<std::size_t>(kK));
        std::vector<float> V(static_cast<std::size_t>(kN) * kK);

        for (float& x : U) {
            x = dist(gen);
        }
        for (float& x : S) {
            x = std::abs(dist(gen)) + 0.1f;
        }
        for (float& x : V) {
            x = dist(gen);
        }

        reconstruct_lowrank_cuda_initialize(&ctx, kM, kN, kK, kSeed + 1ULL);
        check_cuda(cudaMalloc(&buffers.d_U, static_cast<std::size_t>(kM) * kK * sizeof(float)), "cudaMalloc d_U failed");
        check_cuda(cudaMalloc(&buffers.d_S, static_cast<std::size_t>(kK) * sizeof(float)), "cudaMalloc d_S failed");
        check_cuda(cudaMalloc(&buffers.d_V, static_cast<std::size_t>(kN) * kK * sizeof(float)), "cudaMalloc d_V failed");
        check_cuda(cudaMalloc(&buffers.d_out, static_cast<std::size_t>(kM) * kN * sizeof(float)), "cudaMalloc d_out failed");
        check_cuda(cudaMemcpyAsync(
                       buffers.d_U,
                       U.data(),
                       static_cast<std::size_t>(kM) * kK * sizeof(float),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpyAsync U H2D failed");
        check_cuda(cudaMemcpyAsync(
                       buffers.d_S,
                       S.data(),
                       static_cast<std::size_t>(kK) * sizeof(float),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpyAsync S H2D failed");
        check_cuda(cudaMemcpyAsync(
                       buffers.d_V,
                       V.data(),
                       static_cast<std::size_t>(kN) * kK * sizeof(float),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpyAsync V H2D failed");

        std::vector<float> out(static_cast<std::size_t>(kM) * kN, 0.0f);
        for (int i = 0; i < kWarmupIters; ++i) {
            reconstruct_lowrank_cuda(&ctx, buffers.d_U, buffers.d_S, buffers.d_V, buffers.d_out);
        }

        std::vector<double> latencies_ms;
        latencies_ms.reserve(kMeasureIters);
        for (int i = 0; i < kMeasureIters; ++i) {
            const auto begin = std::chrono::high_resolution_clock::now();
            reconstruct_lowrank_cuda(&ctx, buffers.d_U, buffers.d_S, buffers.d_V, buffers.d_out);
            const auto end = std::chrono::high_resolution_clock::now();
            latencies_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        }
        check_cuda(cudaMemcpyAsync(
                       out.data(),
                       buffers.d_out,
                       out.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync out D2H failed");

        if (out.size() != static_cast<std::size_t>(kM) * kN) {
            throw std::runtime_error("output size mismatch");
        }

        std::vector<float> U_after(U.size(), 0.0f);
        std::vector<float> S_after(S.size(), 0.0f);
        std::vector<float> V_after(V.size(), 0.0f);
        check_cuda(cudaMemcpyAsync(
                       U_after.data(),
                       buffers.d_U,
                       U_after.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync U D2H failed");
        check_cuda(cudaMemcpyAsync(
                       S_after.data(),
                       buffers.d_S,
                       S_after.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync S D2H failed");
        check_cuda(cudaMemcpyAsync(
                       V_after.data(),
                       buffers.d_V,
                       V_after.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpyAsync V D2H failed");

        double max_input_drift = 0.0;
        for (std::size_t i = 0; i < U.size(); ++i) {
            max_input_drift = std::max(max_input_drift, std::abs(static_cast<double>(U[i]) - static_cast<double>(U_after[i])));
        }
        for (std::size_t i = 0; i < S.size(); ++i) {
            max_input_drift = std::max(max_input_drift, std::abs(static_cast<double>(S[i]) - static_cast<double>(S_after[i])));
        }
        for (std::size_t i = 0; i < V.size(); ++i) {
            max_input_drift = std::max(max_input_drift, std::abs(static_cast<double>(V[i]) - static_cast<double>(V_after[i])));
        }
        if (!std::isfinite(max_input_drift) || max_input_drift > 0.0) {
            throw std::runtime_error("caller-owned input buffers were modified");
        }

        double max_abs_err = 0.0;
        for (int i = 0; i < 16; ++i) {
            const int r = row_dist(gen);
            const int c = col_dist(gen);
            const float ref = compute_reference_entry(U, S, V, r, c);
            const float got = out[static_cast<std::size_t>(r) * kN + c];
            const double abs_err = std::abs(static_cast<double>(ref) - static_cast<double>(got));
            max_abs_err = std::max(max_abs_err, abs_err);
        }

        if (!std::isfinite(max_abs_err) || max_abs_err > 1e-2) {
            throw std::runtime_error("sanity check failed: max_abs_err too large");
        }

        const auto [min_it, max_it] = std::minmax_element(latencies_ms.begin(), latencies_ms.end());
        const double avg_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
                              static_cast<double>(latencies_ms.size());

        std::vector<double> sorted = latencies_ms;
        std::sort(sorted.begin(), sorted.end());
        const double p50_ms = sorted[sorted.size() / 2];
        const double p95_ms = sorted[static_cast<std::size_t>(std::floor(0.95 * (sorted.size() - 1)))];

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "reconstruct_lowrank_cuda latency benchmark\n";
        std::cout << "shape: U(" << kM << "x" << kK << "), S(" << kK << "), V(" << kN << "x" << kK << ")\n";
        std::cout << "iters: warmup=" << kWarmupIters << ", measure=" << kMeasureIters << '\n';
        std::cout << "latency_ms: avg=" << avg_ms
                  << " min=" << *min_it
                  << " p50=" << p50_ms
                  << " p95=" << p95_ms
                  << " max=" << *max_it << '\n';
        std::cout << "sanity_max_abs_err=" << max_abs_err << '\n';

        reconstruct_lowrank_cuda_release(&ctx);
        return 0;
    } catch (const std::exception& e) {
        reconstruct_lowrank_cuda_release(&ctx);
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
