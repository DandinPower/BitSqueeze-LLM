#pragma once

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace quantization_cuda_detail {

inline void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

} // namespace quantization_cuda_detail
