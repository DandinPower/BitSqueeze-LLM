find_package(CUDAToolkit REQUIRED)

add_library(topk_cuda_lib)

target_sources(topk_cuda_lib
    PRIVATE
        src/topk_cuda.cu
)

target_include_directories(topk_cuda_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(topk_cuda_lib
    PUBLIC
        CUDA::cudart
)
