# bitsqz_llm

`bitsqz_llm` is a CUDA-backed matrix compression library for LLM activations. The supported public integration surface is the installed CMake target `bitsqz_llm::bitsqz_llm` and the single public header `bitsqz_llm.hpp`.

## Features

- Shared or static library builds through standard CMake `BUILD_SHARED_LIBS`.
- A single public project header: `#include <bitsqz_llm.hpp>`.
- Compression and decompression for device-resident FP32 matrices.
- SVD-based low-rank compression plus direct quantization-only mode.
- Installable CMake package for reuse in downstream projects.

## Requirements

- CMake 3.23 or newer
- CUDA Toolkit
- A C++ compiler supported by your CUDA toolchain
- OpenMP, if available, is linked automatically

## Build

Build a shared library and the example:

```bash
cmake -S . -B build -DBUILD_SHARED_LIBS=ON -DBITSQZ_LLM_BUILD_EXAMPLES=ON
cmake --build build -j
```

Build a static library:

```bash
cmake -S . -B build-static -DBUILD_SHARED_LIBS=OFF
cmake --build build-static -j
```

Install into a prefix:

```bash
cmake --install build --prefix "$HOME/.local"
```

The installed package layout is:

- `include/bitsqz_llm.hpp`
- `lib/libbitsqz_llm*.so` or the static equivalent
- `lib/cmake/bitsqz_llm/`

## Public API

### `quantization_method_t`

This enum selects the quantized representation used by the library. `bitsqz_llm` currently accepts:

- `quantization_INVALID`
- `NF4`
- `NF4_DQ`

For `bitsqz_llm_initialize`:

- `svd_uv_format` selects the quantized format for the low-rank `U` and `V` matrices.
- `svd_s_format` selects the quantized format for the singular-value vector.
- `quantization_only_format` selects the format used when no SVD path is enabled.

### `bitsqz_llm_array_t`

This struct owns the packed representation returned by `bitsqz_llm_compress`. Important fields:

- `num_rows`, `num_columns`, `num_elements`: original matrix shape
- `outlier_topk_ratio`, `error_correction_topk_ratio`: top-k settings used during compression
- `svd_ranks`, `svd_niters`, `effective_rank`: SVD configuration and result metadata
- `svd_uv_format`, `svd_s_format`, `quantization_only_format`: format choices used by compression
- `section_*`: offsets and storage kinds for each packed section
- `payload`: pointer to the packed payload owned by the returned allocation

The object returned by `bitsqz_llm_compress` must be released with `bitsqz_llm_free`.

### `bitsqz_llm_compress_profile_t`

This struct reports coarse timing buckets collected during compression, including top-k separation, low-rank factorization, quantization, reconstruction, and other work.

### `int bitsqz_llm_initialize(...)`

Initializes the runtime for a fixed matrix shape and compression configuration.

- `num_rows`, `num_columns`: shape of the FP32 matrix
- `outlier_topk_ratio`, `error_correction_topk_ratio`: optional top-k retention ratios in `[0, 1]`
- `svd_ranks`: target rank, or a negative value to disable the SVD path
- `svd_niters`: SVD iteration count
- `svd_uv_format`, `svd_s_format`: quantized formats for the low-rank path
- `quantization_only_format`: quantized format for the direct quantization path

Returns `0` on success and non-zero on invalid configuration or initialization failure.

### `void bitsqz_llm_release()`

Releases the runtime state created by `bitsqz_llm_initialize`. Call this when the configured workload is no longer needed.

### `int bitsqz_llm_compress(...)`

Compresses a device-resident FP32 matrix.

- `d_row_major_matrix_float_data`: pointer to device memory containing `num_rows * num_columns` FP32 values
- `out`: receives a heap allocation containing the packed result
- `profile`: optional timing output

Returns `0` on success.

### `int bitsqz_llm_decompress(...)`

Decompresses a packed object back into device memory.

- `compressed`: object returned by `bitsqz_llm_compress` or `load_bitsqz_llm_from_buffer`
- `d_dst`: destination device pointer
- `dst_num_elements`: number of FP32 elements available at `d_dst`

Returns `0` on success.

### `uint64_t bitsqz_llm_get_packed_size(...)`

Returns the total number of bytes required to store the packed object, including metadata and payload.

### `bitsqz_llm_array_t *load_bitsqz_llm_from_buffer(...)`

Copies an existing packed buffer into a new owned `bitsqz_llm_array_t` allocation. This is the entry point for loading previously serialized data.

### `void bitsqz_llm_free(...)`

Releases a packed allocation returned by `bitsqz_llm_compress` or `load_bitsqz_llm_from_buffer`.

## Example

See [`examples/bitsqz_llm_walkthrough.cpp`](/net/home/liaw/Gata-DI/BitSqueeze-LLM/examples/bitsqz_llm_walkthrough.cpp) for a complete initialize/compress/decompress walkthrough.

Minimal usage:

```cpp
#include <bitsqz_llm.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

int main() {
    constexpr uint16_t rows = 8;
    constexpr uint16_t cols = 8;

    std::vector<float> host_matrix(rows * cols, 1.0f);
    float *d_source = nullptr;
    float *d_restored = nullptr;
    bitsqz_llm_array_t *compressed = nullptr;

    cudaMalloc(&d_source, host_matrix.size() * sizeof(float));
    cudaMalloc(&d_restored, host_matrix.size() * sizeof(float));
    cudaMemcpy(d_source, host_matrix.data(), host_matrix.size() * sizeof(float), cudaMemcpyHostToDevice);

    bitsqz_llm_initialize(rows, cols, 0.0f, 0.0f, 4, 2, NF4, NF4, quantization_INVALID);
    bitsqz_llm_compress(d_source, &compressed, nullptr);
    bitsqz_llm_decompress(compressed, d_restored, static_cast<uint32_t>(host_matrix.size()));

    bitsqz_llm_free(compressed);
    bitsqz_llm_release();
    cudaFree(d_source);
    cudaFree(d_restored);
    return 0;
}
```

## Downstream CMake Usage

After installing `bitsqz_llm`, a downstream project can consume it with:

```cmake
cmake_minimum_required(VERSION 3.23)
project(my_bitsqz_app LANGUAGES CXX CUDA)

find_package(bitsqz_llm CONFIG REQUIRED)

add_executable(my_bitsqz_app main.cpp)
target_link_libraries(my_bitsqz_app PRIVATE bitsqz_llm::bitsqz_llm)
```

If `main.cpp` performs its own CUDA memory management, it can include `cuda_runtime.h` directly. No internal `bitsqz_llm` module headers are required; downstream source code only needs to include:

```cpp
#include <bitsqz_llm.hpp>
```

## Running the Example

Build and run the in-tree walkthrough:

```bash
cmake -S . -B build -DBUILD_SHARED_LIBS=ON -DBITSQZ_LLM_BUILD_EXAMPLES=ON
cmake --build build --target bitsqz_llm_walkthrough -j
./build/examples/bitsqz_llm_walkthrough
```
