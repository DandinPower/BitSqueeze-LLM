cmake_minimum_required(VERSION 3.23)

project(svd_lowrank_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)    # Compile the generation after A100 (https://developer.nvidia.com/cuda/gpus)

add_subdirectory(tests)

add_subdirectory(modules/quantization)

add_subdirectory(modules/svd_lowrank_cuda)

add_subdirectory(modules/topk)

add_subdirectory(modules/reconstruct_lowrank_cuda)

add_library(bitsqz_llm_lib)

find_package(OpenMP)

target_sources(bitsqz_llm_lib
    PRIVATE 
        src/bitsqz_llm.cpp
)
target_include_directories(bitsqz_llm_lib
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(bitsqz_llm_lib
    PUBLIC
        quantization_lib
        topk_lib
        svd_lowrank_cuda_lib
        reconstruct_lowrank_cuda_lib
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bitsqz_llm_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
