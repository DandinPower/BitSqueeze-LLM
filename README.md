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

- `quantization_NONE`
- `NF4`
- `NF4_DQ`

`quantization_NONE` disables quantization for that path and keeps the corresponding section in FP32 form.

For the split runtime APIs:

- `svd_uv_format` selects the quantized format for the low-rank `U` and `V` matrices.
- `svd_s_format` selects the quantized format for the singular-value vector.
- `quantization_only_format` selects the format used when no SVD path is enabled.

### `bitsqz_llm_input_location_t`

This enum selects how `bitsqz_llm_compress` interprets its input pointer.

- `BITSQZ_LLM_INPUT_HOST = 0`: the input pointer is host memory
- `BITSQZ_LLM_INPUT_DEVICE = 1`: the input pointer is CUDA device memory

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

### `int bitsqz_llm_compress_initialize(...)`

Initializes the compression runtime for a fixed matrix shape and compression configuration.

- `num_rows`, `num_columns`: shape of the FP32 matrix
- `outlier_topk_ratio`, `error_correction_topk_ratio`: optional top-k retention ratios in `[0, 1]`
- `svd_ranks`: target rank, or a negative value to disable the SVD path
- `svd_niters`: SVD iteration count
- `svd_uv_format`, `svd_s_format`: quantized formats for the low-rank path
- `quantization_only_format`: quantized format for the direct quantization path
- `input_location`: whether `bitsqz_llm_compress` will receive host or device pointers

Returns `0` on success and non-zero on invalid configuration or initialization failure.

Practical configuration rules:

- Set `outlier_topk_ratio = 0.0f` to disable outlier extraction entirely.
- Set `error_correction_topk_ratio = 0.0f` to disable error-correction top-k storage entirely.
- Set `svd_ranks >= 1` to enable the SVD low-rank path.
- Set `svd_ranks < 1` to disable SVD and use the direct path only.
- When SVD is enabled, `svd_uv_format` and `svd_s_format` control whether `U`, `S`, and `V` are stored as quantized sections or as FP32 sections.
- When SVD is disabled, `quantization_only_format` controls how the full matrix is stored on the direct path.
- Use `quantization_NONE` for any format argument when you want that path to stay in FP32 instead of being quantized.
- For `svd_s_format`, `quantization_NONE` is usually recommended because the singular-value vector is small, so quantizing it typically has negligible impact on total packed size.

In other words:

- `svd_ranks >= 1` means "compress with low-rank SVD, then optionally quantize the SVD factors."
- `svd_ranks < 1` means "skip SVD and store the matrix through the direct path."
- `quantization_NONE` means "do not quantize this path."
- `NF4` or `NF4_DQ` mean "quantize this path with that format."

Common setups:

1. Quantization-only, with no outlier path and no error correction.

```cpp
bitsqz_llm_compress_initialize(
    rows,
    cols,
    0.0f,               // outlier_topk_ratio: disabled
    0.0f,               // error_correction_topk_ratio: disabled
    -1,                 // svd_ranks: disable SVD
    1,                  // svd_niters: ignored when SVD is disabled
    quantization_NONE,  // svd_uv_format: unused because SVD is disabled
    quantization_NONE,  // svd_s_format: unused because SVD is disabled
    NF4,                // quantization_only_format: direct quantization path
    BITSQZ_LLM_INPUT_DEVICE);
```

2. SVD-only, with no quantization.

```cpp
bitsqz_llm_compress_initialize(
    rows,
    cols,
    0.0f,
    0.0f,
    128,                // enable SVD with rank 128
    2,                  // SVD iterations
    quantization_NONE,  // keep U/V in FP32
    quantization_NONE,  // keep S in FP32
    quantization_NONE,  // direct path format is unused while SVD is enabled
    BITSQZ_LLM_INPUT_DEVICE);
```

3. SVD plus quantized factors.

```cpp
bitsqz_llm_compress_initialize(
    rows,
    cols,
    0.0f,
    0.0f,
    128,
    2,
    NF4,                // quantize U/V
    NF4,                // quantize S
    quantization_NONE,  // direct path format is unused while SVD is enabled
    BITSQZ_LLM_INPUT_DEVICE);
```

In practice, many configurations use `NF4` for `svd_uv_format` and `quantization_NONE` for `svd_s_format`, because the `S` vector is much smaller than `U` and `V`.

4. SVD plus outlier and error-correction paths.

```cpp
bitsqz_llm_compress_initialize(
    rows,
    cols,
    0.01f,              // keep top 1% outlier columns
    0.02f,              // keep top 2% residual/error columns
    128,
    2,
    NF4,
    quantization_NONE,
    quantization_NONE,
    BITSQZ_LLM_INPUT_DEVICE);
```

Notes:

- Ratios must be in `[0, 1]`.
- `svd_niters` must be at least `1`.
- The current implementation accepts `quantization_NONE`, `NF4`, and `NF4_DQ` for format arguments.
- If you pass `BITSQZ_LLM_INPUT_HOST`, `bitsqz_llm_compress` expects a host pointer. If you pass `BITSQZ_LLM_INPUT_DEVICE`, it expects a CUDA device pointer.

### `void bitsqz_llm_compress_release()`

Releases the compression runtime state created by `bitsqz_llm_compress_initialize`.

### `int bitsqz_llm_decompress_initialize(...)`

Initializes the decompression runtime for a fixed matrix shape and deterministic decompression configuration.

- `num_rows`, `num_columns`: shape of the FP32 matrix
- `outlier_topk_ratio`, `error_correction_topk_ratio`: top-k capacities to support during decompression
- `svd_ranks`: exact packed SVD rank to decode; use a value less than `1` when only non-SVD payloads need to be decompressed
- `svd_uv_format`, `svd_s_format`: quantized formats supported for the low-rank path
- `quantization_only_format`: quantized format supported for the direct quantization path

Returns `0` on success and non-zero on invalid configuration or initialization failure.

### `void bitsqz_llm_decompress_release()`

Releases the decompression runtime state created by `bitsqz_llm_decompress_initialize`.

### `int bitsqz_llm_compress(...)`

Compresses an FP32 matrix. The input pointer is interpreted according to the `input_location` selected during `bitsqz_llm_compress_initialize`.

- `row_major_matrix_float_data`: pointer to host or device memory containing `num_rows * num_columns` FP32 values
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
    float *d_restored = nullptr;
    bitsqz_llm_array_t *compressed = nullptr;

    cudaMalloc(&d_restored, host_matrix.size() * sizeof(float));
    bitsqz_llm_compress_initialize(
        rows, cols, 0.0f, 0.0f, 4, 2, NF4, NF4, quantization_NONE, BITSQZ_LLM_INPUT_HOST);
    bitsqz_llm_decompress_initialize(
        rows, cols, 0.0f, 0.0f, 4, NF4, NF4, quantization_NONE);
    bitsqz_llm_compress(host_matrix.data(), &compressed, nullptr);
    bitsqz_llm_decompress(compressed, d_restored, static_cast<uint32_t>(host_matrix.size()));

    bitsqz_llm_free(compressed);
    bitsqz_llm_compress_release();
    bitsqz_llm_decompress_release();
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

## Citation

If you use `bitsqz_llm` in your research or project, please cite it as follows:

### BibTeX

```bibtex
@software{bitsqz_llm,
  author = {Yong-Cheng Liaw and Gata.xyz},
  title = {BitSqueeze-LLM: CUDA-backed Matrix Compression for LLM Activations},
  year = {2026},
  url = {https://github.com/DandinPower/BitSqueeze-LLM}
}
