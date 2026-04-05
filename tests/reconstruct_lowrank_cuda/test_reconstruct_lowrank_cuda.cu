#include <reconstruct_lowrank_cuda/reconstruct_lowrank_cuda.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int kM = 512;
constexpr int kK = 128;
constexpr int kN = 8192;
constexpr int kWarmupIters = 5;
constexpr int kMeasureIters = 30;
constexpr unsigned long long kSeed = 20260405ULL;

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

        std::vector<float> out;
        for (int i = 0; i < kWarmupIters; ++i) {
            reconstruct_lowrank_cuda(&ctx, U, S, V, &out);
        }

        std::vector<double> latencies_ms;
        latencies_ms.reserve(kMeasureIters);
        for (int i = 0; i < kMeasureIters; ++i) {
            const auto begin = std::chrono::high_resolution_clock::now();
            reconstruct_lowrank_cuda(&ctx, U, S, V, &out);
            const auto end = std::chrono::high_resolution_clock::now();
            latencies_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        }

        if (out.size() != static_cast<std::size_t>(kM) * kN) {
            throw std::runtime_error("output size mismatch");
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
