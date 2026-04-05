find_package(CUDAToolkit REQUIRED)

add_library(reconstruct_lowrank_cuda_lib)

target_sources(reconstruct_lowrank_cuda_lib
    PRIVATE
        src/reconstruct_lowrank_cuda.cu
)

target_include_directories(reconstruct_lowrank_cuda_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(reconstruct_lowrank_cuda_lib 
    PUBLIC
        CUDA::cublas
)
