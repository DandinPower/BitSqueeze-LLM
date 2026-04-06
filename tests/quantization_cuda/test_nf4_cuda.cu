#include <quantization/quantization.hpp>
#include <quantization/utils/evaluation.hpp>
#include <quantization/utils/random.hpp>
#include <quantization_cuda/quantization_cuda.cuh>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceBuffers {
    float *d_input = nullptr;
    float *d_output = nullptr;
    float *d_block_scales = nullptr;
    uint8_t *d_data = nullptr;

    ~DeviceBuffers() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_block_scales);
        cudaFree(d_data);
    }
};

int run_case(const float *input, unsigned long long count) {
    DeviceBuffers buffers;
    quantization_buffer_t *legacy = nullptr;
    QuantizationCUDAContext ctx{};

    try {
        const auto num_blocks = quantization_cuda_compute_num_blocks(count);
        const auto packed_bytes = static_cast<std::size_t>((count + 1ULL) / 2ULL);
        check_cuda(cudaMalloc(&buffers.d_input, static_cast<std::size_t>(count) * sizeof(float)),
                   "cudaMalloc d_input failed");
        check_cuda(cudaMalloc(&buffers.d_output, static_cast<std::size_t>(count) * sizeof(float)),
                   "cudaMalloc d_output failed");
        check_cuda(cudaMalloc(&buffers.d_block_scales, static_cast<std::size_t>(num_blocks) * sizeof(float)),
                   "cudaMalloc d_block_scales failed");
        check_cuda(cudaMalloc(&buffers.d_data, packed_bytes), "cudaMalloc d_data failed");
        check_cuda(cudaMemcpy(
                       buffers.d_input,
                       input,
                       static_cast<std::size_t>(count) * sizeof(float),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpy input H2D failed");

        quantization_cuda_initialize(&ctx, count, 1234ULL);

        quantization_cuda_array_t quant_array{};
        quantization_cuda_bind_nf4_array(
            &quant_array,
            count,
            buffers.d_block_scales,
            buffers.d_data);

        quantization_cuda_compress(&ctx, buffers.d_input, &quant_array);

        const std::uint64_t packed_size = quantization_cuda_get_packed_size(&quant_array);
        std::vector<uint8_t> packed(static_cast<std::size_t>(packed_size), 0);
        if (quantization_cuda_pack_to_buffer(&quant_array, packed.data(), packed_size) != 0) {
            throw std::runtime_error("quantization_cuda_pack_to_buffer failed");
        }

        quantization_cuda_packed_header_t header{};
        if (quantization_cuda_validate_packed_buffer(packed.data(), packed_size, &header) != 0) {
            throw std::runtime_error("quantization_cuda_validate_packed_buffer failed");
        }
        if (header.method != NF4 || header.num_elements != count) {
            throw std::runtime_error("unexpected quantization_cuda packed header");
        }

        check_cuda(cudaMemset(buffers.d_block_scales, 0, static_cast<std::size_t>(num_blocks) * sizeof(float)),
                   "cudaMemset d_block_scales failed");
        check_cuda(cudaMemset(buffers.d_data, 0, packed_bytes), "cudaMemset d_data failed");

        if (quantization_cuda_unpack_from_buffer(packed.data(), packed_size, &quant_array) != 0) {
            throw std::runtime_error("quantization_cuda_unpack_from_buffer failed");
        }
        quantization_cuda_decompress(&ctx, &quant_array, buffers.d_output);

        std::vector<float> cuda_deq(static_cast<std::size_t>(count), 0.0f);
        check_cuda(cudaMemcpy(
                       cuda_deq.data(),
                       buffers.d_output,
                       static_cast<std::size_t>(count) * sizeof(float),
                       cudaMemcpyDeviceToHost),
                   "cudaMemcpy output D2H failed");

        if (quantization_compress(input, count, NF4, &legacy) != 0 || legacy == nullptr) {
            throw std::runtime_error("legacy NF4 compress failed");
        }
        std::vector<float> legacy_deq(static_cast<std::size_t>(count), 0.0f);
        if (quantization_decompress(legacy, legacy_deq.data(), count) != 0) {
            throw std::runtime_error("legacy NF4 decompress failed");
        }

        for (unsigned long long i = 0; i < count; ++i) {
            if (cuda_deq[static_cast<std::size_t>(i)] != legacy_deq[static_cast<std::size_t>(i)]) {
                std::fprintf(stderr, "NF4 CUDA mismatch at %llu\n", i);
                return 1;
            }
        }

        double mae = 0.0;
        double mse = 0.0;
        double maxabs = 0.0;
        measure_metrics(input, cuda_deq.data(), count, &mae, &mse, &maxabs);
        std::printf(
            "NF4 CUDA roundtrip: packed=%llu bytes, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
            static_cast<unsigned long long>(packed_size),
            mae,
            mse,
            maxabs);
        quantization_free(legacy);
        quantization_cuda_release(&ctx);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "NF4 CUDA test failed: %s\n", e.what());
        quantization_free(legacy);
        quantization_cuda_release(&ctx);
        return 1;
    }
}

} // namespace

int main(void) {
    const unsigned long long count = 1ULL << 20;
    float **inputs = gen_random_float_arrays(2, count, -10.0f, 10.0f, 12345);
    if (inputs == nullptr) {
        return EXIT_FAILURE;
    }

    int result = EXIT_SUCCESS;
    for (unsigned long long i = 0; i < 2; ++i) {
        if (run_case(inputs[i], count) != 0) {
            result = EXIT_FAILURE;
            break;
        }
    }

    free_random_float_arrays(inputs, 2);
    return result;
}
