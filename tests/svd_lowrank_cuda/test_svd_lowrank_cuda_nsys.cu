#include <svd_lowrank_cuda/svd_lowrank_cuda.cuh>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kM = 512;
constexpr int kN = 8192;
constexpr int kQ = 128;
constexpr int kNiter = 2;

constexpr unsigned long long kWarmupSeed = 2026040601ULL;
constexpr unsigned long long kReplaySeed = 2026040602ULL;

struct ScopedDeviceBuffers {
    float* d_A = nullptr;
    float* d_U = nullptr;
    float* d_S = nullptr;
    float* d_V = nullptr;

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

std::vector<float> make_input_matrix(int m, int n, std::uint32_t seed) {
    std::vector<float> A(static_cast<std::size_t>(m) * n);
    std::uint32_t state = seed;

    for (float& x : A) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        x = static_cast<float>(state & 0x00ffffffu) / 8388608.0f - 1.0f;
    }

    return A;
}

void run_profiled_replay(
    ScopedDeviceBuffers* buffers,
    unsigned long long seed,
    const char* range_name,
    const char* before_mark,
    const char* after_mark) {
    nvtxMarkA(before_mark);
    nvtxRangePushA(range_name);
    svd_lowrank_cuda(buffers->d_A, buffers->d_U, buffers->d_S, buffers->d_V, seed);
    nvtxRangePop();
    nvtxMarkA(after_mark);
}

} // namespace

int main() {
    try {
        ScopedDeviceBuffers buffers;

        const std::vector<float> A = make_input_matrix(kM, kN, 0x13579BDFu);

        check_cuda(cudaMalloc(&buffers.d_A, static_cast<std::size_t>(kM) * kN * sizeof(float)), "cudaMalloc d_A failed");
        check_cuda(cudaMalloc(&buffers.d_U, static_cast<std::size_t>(kM) * kQ * sizeof(float)), "cudaMalloc d_U failed");
        check_cuda(cudaMalloc(&buffers.d_S, static_cast<std::size_t>(kQ) * sizeof(float)), "cudaMalloc d_S failed");
        check_cuda(cudaMalloc(&buffers.d_V, static_cast<std::size_t>(kN) * kQ * sizeof(float)), "cudaMalloc d_V failed");
        check_cuda(
            cudaMemcpyAsync(
                buffers.d_A,
                A.data(),
                static_cast<std::size_t>(kM) * kN * sizeof(float),
                cudaMemcpyHostToDevice),
            "cudaMemcpyAsync A H2D failed");

        std::cout << "svd_lowrank_cuda nsys profile case" << '\n';
        std::cout << "m=" << kM << ", n=" << kN << ", q=" << kQ << ", niter=" << kNiter << '\n';
        std::cout << "1. initialize + warmup + graph capture" << '\n';
        svd_lowrank_cuda_initialize(kM, kN, kQ, kNiter, kWarmupSeed);
        std::cout << "filterable NVTX subranges: "
                  << "svd_lowrank_cuda_graph_replay, "
                  << "svd_lowrank_cuda_postsvd, "
                  << "svd_lowrank_cuda_cusolver_xgesvdp, "
                  << "svd_lowrank_cuda_recover_u_gemm, "
                  << "svd_lowrank_cuda_output_materialization" << '\n';

        std::cout << "2. replay once on 512x8192" << '\n';
        run_profiled_replay(
            &buffers,
            kReplaySeed,
            "svd_lowrank_cuda_replay_1_512x8192",
            "before_svd_lowrank_cuda_replay_1_512x8192",
            "after_svd_lowrank_cuda_replay_1_512x8192");

        std::cout << "3. replay again on 512x8192" << '\n';
        run_profiled_replay(
            &buffers,
            kReplaySeed,
            "svd_lowrank_cuda_replay_2_512x8192",
            "before_svd_lowrank_cuda_replay_2_512x8192",
            "after_svd_lowrank_cuda_replay_2_512x8192");

        svd_lowrank_cuda_release();
        return 0;
    } catch (const std::exception& e) {
        svd_lowrank_cuda_release();
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
