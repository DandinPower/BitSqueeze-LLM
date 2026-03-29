#pragma once

#include <cstdint>
#include <vector>

struct SVDLowrankTimings {
    double h2d_ms = 0.0;
    double transpose_ms = 0.0;
    double compute_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
};

struct SVDLowrankCPUResult {
    int m = 0;
    int n = 0;
    int k = 0;
    std::vector<float> U_row_major;
    std::vector<float> S;
    std::vector<float> V_row_major;
};

const SVDLowrankCPUResult& svd_lowrank_cuda(
    const float* A_row_major,
    int niter = 2,
    unsigned long long seed = 1234ULL,
    SVDLowrankTimings* timings = nullptr);

void svd_lowrank_cuda_initialize(
    int m,
    int n,
    int q,
    unsigned long long warmup_seed = 1234ULL);

void svd_lowrank_cuda_release();
