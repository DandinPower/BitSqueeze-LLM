find_package(CUDAToolkit REQUIRED)

add_library(svd_lowrank_cuda_lib)

target_sources(svd_lowrank_cuda_lib
    PRIVATE
        src/svd_lowrank_cuda.cu
)

target_include_directories(svd_lowrank_cuda_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(svd_lowrank_cuda_lib 
    PRIVATE
        CUDA::curand
        CUDA::cublas
        CUDA::cusolver
)