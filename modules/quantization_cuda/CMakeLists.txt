find_package(CUDAToolkit REQUIRED)

add_library(quantization_cuda_lib)

target_sources(quantization_cuda_lib
    PRIVATE
        src/quantization_cuda.cu
)

target_include_directories(quantization_cuda_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(quantization_cuda_lib
    PUBLIC
        CUDA::cudart
)
