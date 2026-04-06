find_package(CUDAToolkit REQUIRED)

add_library(topk_cuda_lib)

target_sources(topk_cuda_lib
    PRIVATE
        src/topk_cuda.cu
)

target_include_directories(topk_cuda_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(topk_cuda_lib
    PUBLIC
        CUDA::cudart
)
