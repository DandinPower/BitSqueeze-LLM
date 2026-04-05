#pragma once

#include <cstdint>
#include <vector>

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
    unsigned long long seed = 1234ULL);

void svd_lowrank_cuda_initialize(
    int m,
    int n,
    int q,
    int niter,
    unsigned long long warmup_seed = 1234ULL);

void svd_lowrank_cuda_release();
