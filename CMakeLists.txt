cmake_minimum_required(VERSION 3.23)

project(svd_lowrank_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)    # Compile the generation after A100 (https://developer.nvidia.com/cuda/gpus)

add_subdirectory(tests)

add_subdirectory(modules/quantization)

add_subdirectory(modules/svd_lowrank_cuda)

add_subdirectory(modules/topk)