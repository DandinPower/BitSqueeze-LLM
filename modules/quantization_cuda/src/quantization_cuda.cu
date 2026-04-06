#include <quantization_cuda/quantization_cuda.cuh>

#include "quantization_cuda_utils.cuh"

#include <cub/cub.cuh>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

using quantization_cuda_detail::check_cuda;

namespace {

constexpr int kQuantizationBlockThreads = 64;
constexpr int kDequantizationBlockThreads = 256;

__device__ __constant__ float kNF4Levels[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

__device__ __forceinline__ float sanitize_input_value(float value) {
    return isfinite(value) ? value : 0.0f;
}

__device__ __forceinline__ std::uint8_t float_to_nf4_code(float value) {
    if (!isfinite(value)) {
        value = 0.0f;
    }

    int best_index = 0;
    float best_error = fabsf(value - kNF4Levels[0]);
    for (int i = 1; i < 16; ++i) {
        const float error = fabsf(value - kNF4Levels[i]);
        if (error < best_error) {
            best_error = error;
            best_index = i;
        }
    }
    return static_cast<std::uint8_t>(best_index);
}

__device__ __forceinline__ float nf4_code_to_fp32(std::uint8_t code) {
    return kNF4Levels[code & 0xF];
}

__device__ __forceinline__ std::uint8_t fp32_to_e4m3(float value) {
    constexpr int kFp8ExponentBias = 7;
    constexpr int kFp8ExponentBits = 4;

    if (!isfinite(value)) {
        value =
            (value < 0.0f)
                ? -QUANTIZATION_CUDA_NF4_DQ_FP8_MAX_NORM_VALUE
                : QUANTIZATION_CUDA_NF4_DQ_FP8_MAX_NORM_VALUE;
    }

    const int sign = signbit(value) ? 1 : 0;
    float abs_value = fabsf(value);
    if (abs_value == 0.0f) {
        return static_cast<std::uint8_t>(sign << 7);
    }
    if (abs_value > QUANTIZATION_CUDA_NF4_DQ_FP8_MAX_NORM_VALUE) {
        abs_value = QUANTIZATION_CUDA_NF4_DQ_FP8_MAX_NORM_VALUE;
    }

    int exp2 = 0;
    const float mantissa = frexpf(abs_value, &exp2);
    int exponent_field = exp2 - 1 + kFp8ExponentBias;

    if (exponent_field <= 0) {
        int mantissa_field = static_cast<int>(lrintf(abs_value * 512.0f));
        if (mantissa_field > 7) {
            mantissa_field = 7;
        }
        return static_cast<std::uint8_t>((sign << 7) | (mantissa_field & 0x7));
    }

    int mantissa_field = static_cast<int>(lrintf(((mantissa * 2.0f) - 1.0f) * 8.0f));
    if (mantissa_field > 7) {
        mantissa_field = 0;
        exponent_field += 1;
    }

    if (exponent_field >= (1 << kFp8ExponentBits)) {
        exponent_field = (1 << kFp8ExponentBits) - 1;
        mantissa_field = 7;
    }

    return static_cast<std::uint8_t>(
        (sign << 7) | ((exponent_field & 0xF) << 3) | (mantissa_field & 0x7));
}

__device__ __forceinline__ float e4m3_to_fp32(std::uint8_t value) {
    constexpr int kFp8ExponentBias = 7;

    const int sign = (value >> 7) & 0x1;
    const int exponent_field = (value >> 3) & 0xF;
    const int mantissa_field = value & 0x7;

    float result = 0.0f;
    if (exponent_field == 0) {
        const float mantissa = static_cast<float>(mantissa_field) / 8.0f;
        result = mantissa * ldexpf(1.0f, 1 - kFp8ExponentBias);
    } else {
        const int exponent = exponent_field - kFp8ExponentBias;
        const float mantissa = 1.0f + static_cast<float>(mantissa_field) / 8.0f;
        result = mantissa * ldexpf(1.0f, exponent);
    }
    return sign ? -result : result;
}

__global__ void nf4_compress_kernel(
    const float *d_input,
    float *d_block_scales,
    std::uint8_t *d_data,
    std::uint64_t num_elements,
    std::uint64_t block_size) {
    const std::uint64_t block_index = static_cast<std::uint64_t>(blockIdx.x);
    const std::uint64_t start = block_index * block_size;
    if (start >= num_elements) {
        return;
    }

    const unsigned int tid = threadIdx.x;
    const std::uint64_t remain = min(block_size, num_elements - start);

    __shared__ float shared_absmax[kQuantizationBlockThreads];

    float abs_value = 0.0f;
    if (tid < remain) {
        const float value = sanitize_input_value(d_input[start + tid]);
        abs_value = fabsf(value);
    }
    shared_absmax[tid] = abs_value;
    __syncthreads();

    for (unsigned int stride = kQuantizationBlockThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_absmax[tid] = fmaxf(shared_absmax[tid], shared_absmax[tid + stride]);
        }
        __syncthreads();
    }

    const float block_scale = (shared_absmax[0] > 0.0f) ? shared_absmax[0] : 1.0f;
    if (tid == 0) {
        d_block_scales[block_index] = block_scale;
    }
    __syncthreads();

    if (tid < ((remain + 1ULL) / 2ULL)) {
        const std::uint64_t local_index = static_cast<std::uint64_t>(tid) * 2ULL;
        const std::uint64_t packed_index = (start + local_index) / 2ULL;
        const float inverse_scale = 1.0f / block_scale;

        std::uint8_t packed_value = 0;
        if (local_index < remain) {
            const float value = sanitize_input_value(d_input[start + local_index]) * inverse_scale;
            packed_value = static_cast<std::uint8_t>((float_to_nf4_code(value) & 0xF) << 4);
        }
        if ((local_index + 1ULL) < remain) {
            const float value = sanitize_input_value(d_input[start + local_index + 1ULL]) * inverse_scale;
            packed_value |= static_cast<std::uint8_t>(float_to_nf4_code(value) & 0xF);
        }
        d_data[packed_index] = packed_value;
    }
}

__global__ void compute_block_absmax_kernel(
    const float *d_input,
    float *d_block_absmax,
    std::uint64_t num_elements,
    std::uint64_t block_size) {
    const std::uint64_t block_index = static_cast<std::uint64_t>(blockIdx.x);
    const std::uint64_t start = block_index * block_size;
    if (start >= num_elements) {
        return;
    }

    const unsigned int tid = threadIdx.x;
    const std::uint64_t remain = min(block_size, num_elements - start);

    __shared__ float shared_absmax[kQuantizationBlockThreads];

    float abs_value = 0.0f;
    if (tid < remain) {
        const float value = sanitize_input_value(d_input[start + tid]);
        abs_value = fabsf(value);
    }
    shared_absmax[tid] = abs_value;
    __syncthreads();

    for (unsigned int stride = kQuantizationBlockThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_absmax[tid] = fmaxf(shared_absmax[tid], shared_absmax[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_block_absmax[block_index] = (shared_absmax[0] > 0.0f) ? shared_absmax[0] : 1.0f;
    }
}

__global__ void store_dq_scale_kernel(const float *d_reduce_out, float *d_dq_scale) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    const float abs_max = d_reduce_out[0];
    if (abs_max == 0.0f || !isfinite(abs_max)) {
        d_dq_scale[0] = 1.0f;
        return;
    }
    d_dq_scale[0] = abs_max / QUANTIZATION_CUDA_NF4_DQ_FP8_MAX_NORM_VALUE;
}

__global__ void nf4_dq_compress_kernel(
    const float *d_input,
    const float *d_block_absmax,
    const float *d_dq_scale,
    std::uint8_t *d_block_scales,
    std::uint8_t *d_data,
    std::uint64_t num_elements,
    std::uint64_t block_size) {
    const std::uint64_t block_index = static_cast<std::uint64_t>(blockIdx.x);
    const std::uint64_t start = block_index * block_size;
    if (start >= num_elements) {
        return;
    }

    const unsigned int tid = threadIdx.x;
    const std::uint64_t remain = min(block_size, num_elements - start);

    float dq_scale = d_dq_scale[0];
    if (dq_scale == 0.0f || !isfinite(dq_scale)) {
        dq_scale = 1.0f;
    }

    const std::uint8_t block_scale_code =
        fp32_to_e4m3(d_block_absmax[block_index] / dq_scale);
    if (tid == 0) {
        d_block_scales[block_index] = block_scale_code;
    }
    __syncthreads();

    float block_scale = dq_scale * e4m3_to_fp32(block_scale_code);
    if (block_scale == 0.0f || !isfinite(block_scale)) {
        block_scale = 1.0f;
    }
    const float inverse_scale = 1.0f / block_scale;

    if (tid < ((remain + 1ULL) / 2ULL)) {
        const std::uint64_t local_index = static_cast<std::uint64_t>(tid) * 2ULL;
        const std::uint64_t packed_index = (start + local_index) / 2ULL;

        std::uint8_t packed_value = 0;
        if (local_index < remain) {
            const float value = sanitize_input_value(d_input[start + local_index]) * inverse_scale;
            packed_value = static_cast<std::uint8_t>((float_to_nf4_code(value) & 0xF) << 4);
        }
        if ((local_index + 1ULL) < remain) {
            const float value = sanitize_input_value(d_input[start + local_index + 1ULL]) * inverse_scale;
            packed_value |= static_cast<std::uint8_t>(float_to_nf4_code(value) & 0xF);
        }
        d_data[packed_index] = packed_value;
    }
}

__global__ void nf4_decompress_kernel(
    const float *d_block_scales,
    const std::uint8_t *d_data,
    float *d_output,
    std::uint64_t num_elements,
    std::uint64_t block_size) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_elements) {
        return;
    }

    float block_scale = d_block_scales[index / block_size];
    if (block_scale == 0.0f || !isfinite(block_scale)) {
        block_scale = 1.0f;
    }

    const std::uint8_t packed_value = d_data[index / 2ULL];
    const std::uint8_t code =
        ((index % 2ULL) == 0ULL)
            ? static_cast<std::uint8_t>(packed_value >> 4)
            : static_cast<std::uint8_t>(packed_value & 0xF);
    d_output[index] = block_scale * nf4_code_to_fp32(code);
}

__global__ void nf4_dq_decompress_kernel(
    const float *d_dq_scale,
    const std::uint8_t *d_block_scales,
    const std::uint8_t *d_data,
    float *d_output,
    std::uint64_t num_elements,
    std::uint64_t block_size) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_elements) {
        return;
    }

    float dq_scale = d_dq_scale[0];
    if (dq_scale == 0.0f || !isfinite(dq_scale)) {
        dq_scale = 1.0f;
    }

    float block_scale = dq_scale * e4m3_to_fp32(d_block_scales[index / block_size]);
    if (block_scale == 0.0f || !isfinite(block_scale)) {
        block_scale = 1.0f;
    }

    const std::uint8_t packed_value = d_data[index / 2ULL];
    const std::uint8_t code =
        ((index % 2ULL) == 0ULL)
            ? static_cast<std::uint8_t>(packed_value >> 4)
            : static_cast<std::uint8_t>(packed_value & 0xF);
    d_output[index] = block_scale * nf4_code_to_fp32(code);
}

bool quantization_cuda_method_is_supported(quantization_method_t method) {
    return method == NF4 || method == NF4_DQ;
}

std::uint64_t quantization_cuda_get_data_size_bytes(std::uint64_t num_elements) {
    return (num_elements + 1ULL) / 2ULL;
}

std::uint64_t quantization_cuda_get_scale_size_bytes(
    quantization_method_t method,
    std::uint64_t num_blocks) {
    switch (method) {
        case NF4:
            return num_blocks * sizeof(float);
        case NF4_DQ:
            return num_blocks * sizeof(std::uint8_t);
        default:
            return 0;
    }
}

void validate_context(const QuantizationCUDAContext *ctx) {
    if (ctx == nullptr) {
        throw std::invalid_argument("ctx must not be null");
    }
    if (!ctx->initialized) {
        throw std::invalid_argument("quantization cuda context is not initialized");
    }
}

void validate_array_descriptor(const quantization_cuda_array_t *quant_array) {
    if (quant_array == nullptr) {
        throw std::invalid_argument("quant_array must not be null");
    }
    if (!quantization_cuda_method_is_supported(quant_array->method)) {
        throw std::invalid_argument("unsupported quantization method");
    }
    if (quant_array->num_elements == 0 || quant_array->num_blocks == 0) {
        throw std::invalid_argument("quant_array shape must be positive");
    }
    if (quant_array->block_size != QUANTIZATION_CUDA_NF4_BLOCK_SIZE) {
        throw std::invalid_argument("block_size must match the CUDA NF4 block size");
    }
    if (quant_array->num_blocks !=
        quantization_cuda_compute_num_blocks(quant_array->num_elements, quant_array->block_size)) {
        throw std::invalid_argument("quant_array num_blocks does not match num_elements");
    }
    if (quant_array->d_data == nullptr) {
        throw std::invalid_argument("quant_array data buffer must not be null");
    }

    switch (quant_array->method) {
        case NF4:
            if (quant_array->d_block_scales_fp32 == nullptr) {
                throw std::invalid_argument("NF4 block scales buffer must not be null");
            }
            break;
        case NF4_DQ:
            if (quant_array->d_block_scales_fp8 == nullptr || quant_array->d_dq_scale == nullptr) {
                throw std::invalid_argument("NF4_DQ buffers must not be null");
            }
            break;
        default:
            throw std::invalid_argument("unsupported quantization method");
    }
}

void validate_context_capacity(
    const QuantizationCUDAContext *ctx,
    const quantization_cuda_array_t *quant_array) {
    validate_context(ctx);
    validate_array_descriptor(quant_array);

    if (quant_array->num_elements > ctx->num_elements_capacity ||
        quant_array->num_blocks > ctx->num_blocks_capacity) {
        throw std::invalid_argument("quant_array exceeds context capacity");
    }
}

} // namespace

void QuantizationCUDADevicePtrs::free_all() noexcept {
    if (d_block_absmax) cudaFree(d_block_absmax);
    if (d_reduce_out) cudaFree(d_reduce_out);
    if (d_reduce_temp_storage) cudaFree(d_reduce_temp_storage);

    d_block_absmax = nullptr;
    d_reduce_out = nullptr;
    d_reduce_temp_storage = nullptr;
    reduce_temp_storage_bytes = 0;
}

void QuantizationCUDAContext::destroy_all() noexcept {
    cuda_ptrs.free_all();

    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }

    initialized = false;
    num_elements_capacity = 0;
    num_blocks_capacity = 0;
}

std::uint64_t quantization_cuda_compute_num_blocks(
    std::uint64_t num_elements,
    std::uint64_t block_size) {
    if (num_elements == 0 || block_size == 0) {
        return 0;
    }
    return (num_elements + block_size - 1ULL) / block_size;
}

void quantization_cuda_bind_nf4_array(
    quantization_cuda_array_t *quant_array,
    std::uint64_t num_elements,
    float *d_block_scales,
    std::uint8_t *d_data) {
    if (quant_array == nullptr) {
        throw std::invalid_argument("quant_array must not be null");
    }

    quant_array->method = NF4;
    quant_array->num_elements = num_elements;
    quant_array->num_blocks =
        quantization_cuda_compute_num_blocks(num_elements, QUANTIZATION_CUDA_NF4_BLOCK_SIZE);
    quant_array->block_size = QUANTIZATION_CUDA_NF4_BLOCK_SIZE;
    quant_array->d_block_scales_fp32 = d_block_scales;
    quant_array->d_block_scales_fp8 = nullptr;
    quant_array->d_dq_scale = nullptr;
    quant_array->d_data = d_data;
}

void quantization_cuda_bind_nf4_dq_array(
    quantization_cuda_array_t *quant_array,
    std::uint64_t num_elements,
    float *d_dq_scale,
    std::uint8_t *d_block_scales,
    std::uint8_t *d_data) {
    if (quant_array == nullptr) {
        throw std::invalid_argument("quant_array must not be null");
    }

    quant_array->method = NF4_DQ;
    quant_array->num_elements = num_elements;
    quant_array->num_blocks =
        quantization_cuda_compute_num_blocks(num_elements, QUANTIZATION_CUDA_NF4_BLOCK_SIZE);
    quant_array->block_size = QUANTIZATION_CUDA_NF4_BLOCK_SIZE;
    quant_array->d_block_scales_fp32 = nullptr;
    quant_array->d_block_scales_fp8 = d_block_scales;
    quant_array->d_dq_scale = d_dq_scale;
    quant_array->d_data = d_data;
}

void quantization_cuda_release(QuantizationCUDAContext *ctx) noexcept {
    if (ctx == nullptr) {
        return;
    }
    ctx->destroy_all();
}

void quantization_cuda_initialize(
    QuantizationCUDAContext *ctx,
    std::uint64_t num_elements_capacity,
    unsigned long long warmup_seed) {
    if (ctx == nullptr) {
        throw std::invalid_argument("ctx must not be null");
    }
    if (num_elements_capacity == 0) {
        throw std::invalid_argument("num_elements_capacity must be positive");
    }
    (void)warmup_seed;

    const std::uint64_t num_blocks_capacity =
        quantization_cuda_compute_num_blocks(num_elements_capacity, QUANTIZATION_CUDA_NF4_BLOCK_SIZE);
    if (num_blocks_capacity == 0 ||
        num_blocks_capacity >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("num_blocks_capacity exceeds supported CUDA reduction range");
    }

    quantization_cuda_release(ctx);

    try {
        ctx->num_elements_capacity = num_elements_capacity;
        ctx->num_blocks_capacity = num_blocks_capacity;

        check_cuda(
            cudaMalloc(
                &ctx->cuda_ptrs.d_block_absmax,
                static_cast<std::size_t>(num_blocks_capacity) * sizeof(float)),
            "cudaMalloc d_block_absmax failed");
        check_cuda(cudaMalloc(&ctx->cuda_ptrs.d_reduce_out, sizeof(float)),
                   "cudaMalloc d_reduce_out failed");

        check_cuda(
            cub::DeviceReduce::Max(
                nullptr,
                ctx->cuda_ptrs.reduce_temp_storage_bytes,
                ctx->cuda_ptrs.d_block_absmax,
                ctx->cuda_ptrs.d_reduce_out,
                static_cast<int>(num_blocks_capacity),
                ctx->stream),
            "cub reduce size query failed");

        if (ctx->cuda_ptrs.reduce_temp_storage_bytes > 0) {
            check_cuda(
                cudaMalloc(
                    &ctx->cuda_ptrs.d_reduce_temp_storage,
                    ctx->cuda_ptrs.reduce_temp_storage_bytes),
                "cudaMalloc d_reduce_temp_storage failed");
        }

        ctx->initialized = true;
    } catch (...) {
        ctx->destroy_all();
        throw;
    }
}

void quantization_cuda_compress(
    QuantizationCUDAContext *ctx,
    const float *d_float_array,
    quantization_cuda_array_t *quant_array) {
    validate_context_capacity(ctx, quant_array);
    if (d_float_array == nullptr) {
        throw std::invalid_argument("d_float_array must not be null");
    }

    switch (quant_array->method) {
        case NF4:
            nf4_compress_kernel<<<
                static_cast<unsigned int>(quant_array->num_blocks),
                kQuantizationBlockThreads>>>(
                d_float_array,
                quant_array->d_block_scales_fp32,
                quant_array->d_data,
                quant_array->num_elements,
                quant_array->block_size);
            break;
        case NF4_DQ:
            compute_block_absmax_kernel<<<
                static_cast<unsigned int>(quant_array->num_blocks),
                kQuantizationBlockThreads>>>(
                d_float_array,
                ctx->cuda_ptrs.d_block_absmax,
                quant_array->num_elements,
                quant_array->block_size);
            check_cuda(cudaGetLastError(), "compute_block_absmax_kernel launch failed");

            check_cuda(
                cub::DeviceReduce::Max(
                    ctx->cuda_ptrs.d_reduce_temp_storage,
                    ctx->cuda_ptrs.reduce_temp_storage_bytes,
                    ctx->cuda_ptrs.d_block_absmax,
                    ctx->cuda_ptrs.d_reduce_out,
                    static_cast<int>(quant_array->num_blocks),
                    ctx->stream),
                "cub reduce max failed");

            store_dq_scale_kernel<<<1, 1>>>(ctx->cuda_ptrs.d_reduce_out, quant_array->d_dq_scale);
            check_cuda(cudaGetLastError(), "store_dq_scale_kernel launch failed");

            nf4_dq_compress_kernel<<<
                static_cast<unsigned int>(quant_array->num_blocks),
                kQuantizationBlockThreads>>>(
                d_float_array,
                ctx->cuda_ptrs.d_block_absmax,
                quant_array->d_dq_scale,
                quant_array->d_block_scales_fp8,
                quant_array->d_data,
                quant_array->num_elements,
                quant_array->block_size);
            break;
        default:
            throw std::invalid_argument("unsupported quantization method");
    }

    check_cuda(cudaGetLastError(), "quantization_cuda_compress launch failed");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after quantization_cuda_compress failed");
}

void quantization_cuda_decompress(
    QuantizationCUDAContext *ctx,
    const quantization_cuda_array_t *quant_array,
    float *d_float_array) {
    validate_context_capacity(ctx, quant_array);
    if (d_float_array == nullptr) {
        throw std::invalid_argument("d_float_array must not be null");
    }

    const int grid = static_cast<int>(
        (quant_array->num_elements + kDequantizationBlockThreads - 1ULL) /
        static_cast<std::uint64_t>(kDequantizationBlockThreads));

    switch (quant_array->method) {
        case NF4:
            nf4_decompress_kernel<<<grid, kDequantizationBlockThreads>>>(
                quant_array->d_block_scales_fp32,
                quant_array->d_data,
                d_float_array,
                quant_array->num_elements,
                quant_array->block_size);
            break;
        case NF4_DQ:
            nf4_dq_decompress_kernel<<<grid, kDequantizationBlockThreads>>>(
                quant_array->d_dq_scale,
                quant_array->d_block_scales_fp8,
                quant_array->d_data,
                d_float_array,
                quant_array->num_elements,
                quant_array->block_size);
            break;
        default:
            throw std::invalid_argument("unsupported quantization method");
    }

    check_cuda(cudaGetLastError(), "quantization_cuda_decompress launch failed");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after quantization_cuda_decompress failed");
}

std::uint64_t quantization_cuda_get_packed_size_for_shape(
    quantization_method_t method,
    std::uint64_t num_elements,
    std::uint64_t block_size) {
    if (!quantization_cuda_method_is_supported(method) ||
        num_elements == 0 ||
        block_size != QUANTIZATION_CUDA_NF4_BLOCK_SIZE) {
        return 0;
    }

    const std::uint64_t num_blocks = quantization_cuda_compute_num_blocks(num_elements, block_size);
    return sizeof(quantization_cuda_packed_header_t) +
           quantization_cuda_get_scale_size_bytes(method, num_blocks) +
           quantization_cuda_get_data_size_bytes(num_elements);
}

std::uint64_t quantization_cuda_get_packed_size(const quantization_cuda_array_t *quant_array) {
    if (quant_array == nullptr) {
        return 0;
    }
    return quantization_cuda_get_packed_size_for_shape(
        quant_array->method,
        quant_array->num_elements,
        quant_array->block_size);
}

int quantization_cuda_validate_packed_buffer(
    const void *buffer,
    std::uint64_t buffer_size,
    quantization_cuda_packed_header_t *header_out) noexcept {
    try {
        if (buffer == nullptr || buffer_size < sizeof(quantization_cuda_packed_header_t)) {
            return 1;
        }

        const auto *header = static_cast<const quantization_cuda_packed_header_t *>(buffer);
        if (header->magic != QUANTIZATION_CUDA_PACKED_MAGIC ||
            header->version != QUANTIZATION_CUDA_PACKED_VERSION) {
            return 1;
        }

        const quantization_method_t method =
            static_cast<quantization_method_t>(header->method);
        if (!quantization_cuda_method_is_supported(method)) {
            return 1;
        }
        if (header->num_elements == 0 || header->block_size != QUANTIZATION_CUDA_NF4_BLOCK_SIZE) {
            return 1;
        }
        if (header->num_blocks !=
            quantization_cuda_compute_num_blocks(header->num_elements, header->block_size)) {
            return 1;
        }

        const std::uint64_t expected_size = quantization_cuda_get_packed_size_for_shape(
            method,
            header->num_elements,
            header->block_size);
        if (buffer_size != expected_size) {
            return 1;
        }

        if (header_out != nullptr) {
            *header_out = *header;
        }
        return 0;
    } catch (...) {
        return 1;
    }
}

int quantization_cuda_pack_to_buffer(
    const quantization_cuda_array_t *quant_array,
    void *buffer,
    std::uint64_t buffer_size) noexcept {
    try {
        validate_array_descriptor(quant_array);
        if (buffer == nullptr) {
            return 1;
        }

        quantization_cuda_packed_header_t header{};
        header.magic = QUANTIZATION_CUDA_PACKED_MAGIC;
        header.version = QUANTIZATION_CUDA_PACKED_VERSION;
        header.method = static_cast<std::int32_t>(quant_array->method);
        header.num_elements = quant_array->num_elements;
        header.num_blocks = quant_array->num_blocks;
        header.block_size = quant_array->block_size;

        const std::uint64_t expected_size = quantization_cuda_get_packed_size(quant_array);
        if (buffer_size != expected_size) {
            return 1;
        }

        auto *bytes = static_cast<std::uint8_t *>(buffer);
        std::uint8_t *scales_dst = bytes + sizeof(header);
        std::uint8_t *data_dst =
            scales_dst +
            quantization_cuda_get_scale_size_bytes(quant_array->method, quant_array->num_blocks);

        switch (quant_array->method) {
            case NF4:
                check_cuda(
                    cudaMemcpy(
                        scales_dst,
                        quant_array->d_block_scales_fp32,
                        static_cast<std::size_t>(quant_array->num_blocks) * sizeof(float),
                        cudaMemcpyDeviceToHost),
                    "cudaMemcpy NF4 block scales D2H failed");
                break;
            case NF4_DQ:
                check_cuda(
                    cudaMemcpy(
                        &header.dq_scale,
                        quant_array->d_dq_scale,
                        sizeof(float),
                        cudaMemcpyDeviceToHost),
                    "cudaMemcpy NF4_DQ dq_scale D2H failed");
                check_cuda(
                    cudaMemcpy(
                        scales_dst,
                        quant_array->d_block_scales_fp8,
                        static_cast<std::size_t>(quant_array->num_blocks) * sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost),
                    "cudaMemcpy NF4_DQ block scales D2H failed");
                break;
            default:
                return 1;
        }

        std::memcpy(bytes, &header, sizeof(header));
        check_cuda(
            cudaMemcpy(
                data_dst,
                quant_array->d_data,
                static_cast<std::size_t>(quantization_cuda_get_data_size_bytes(quant_array->num_elements)),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy quantized data D2H failed");
        return 0;
    } catch (...) {
        return 1;
    }
}

int quantization_cuda_unpack_from_buffer(
    const void *buffer,
    std::uint64_t buffer_size,
    quantization_cuda_array_t *quant_array) noexcept {
    try {
        validate_array_descriptor(quant_array);

        quantization_cuda_packed_header_t header{};
        if (quantization_cuda_validate_packed_buffer(buffer, buffer_size, &header) != 0) {
            return 1;
        }

        if (quant_array->method != static_cast<quantization_method_t>(header.method) ||
            quant_array->num_elements != header.num_elements ||
            quant_array->num_blocks != header.num_blocks ||
            quant_array->block_size != header.block_size) {
            return 1;
        }

        const auto *bytes = static_cast<const std::uint8_t *>(buffer);
        const std::uint8_t *scales_src = bytes + sizeof(header);
        const std::uint8_t *data_src =
            scales_src +
            quantization_cuda_get_scale_size_bytes(quant_array->method, quant_array->num_blocks);

        switch (quant_array->method) {
            case NF4:
                check_cuda(
                    cudaMemcpy(
                        quant_array->d_block_scales_fp32,
                        scales_src,
                        static_cast<std::size_t>(quant_array->num_blocks) * sizeof(float),
                        cudaMemcpyHostToDevice),
                    "cudaMemcpy NF4 block scales H2D failed");
                break;
            case NF4_DQ:
                check_cuda(
                    cudaMemcpy(
                        quant_array->d_dq_scale,
                        &header.dq_scale,
                        sizeof(float),
                        cudaMemcpyHostToDevice),
                    "cudaMemcpy NF4_DQ dq_scale H2D failed");
                check_cuda(
                    cudaMemcpy(
                        quant_array->d_block_scales_fp8,
                        scales_src,
                        static_cast<std::size_t>(quant_array->num_blocks) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
                    "cudaMemcpy NF4_DQ block scales H2D failed");
                break;
            default:
                return 1;
        }

        check_cuda(
            cudaMemcpy(
                quant_array->d_data,
                data_src,
                static_cast<std::size_t>(quantization_cuda_get_data_size_bytes(quant_array->num_elements)),
                cudaMemcpyHostToDevice),
            "cudaMemcpy quantized data H2D failed");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after quantization unpack failed");
        return 0;
    } catch (...) {
        return 1;
    }
}
