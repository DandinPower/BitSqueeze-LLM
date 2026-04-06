find_package(CUDAToolkit REQUIRED)

add_library(quantization_cuda_lib)

target_sources(quantization_cuda_lib
    PRIVATE
        src/quantization_cuda.cu
)

target_include_directories(quantization_cuda_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quantization_cuda_lib
    PUBLIC
        CUDA::cudart
)
