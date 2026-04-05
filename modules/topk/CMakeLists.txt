find_package(CUDAToolkit REQUIRED)

add_library(topk_lib)

target_sources(topk_lib
    PRIVATE
        src/topk.cu
)

target_include_directories(topk_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(topk_lib
    PUBLIC
        CUDA::cudart
)
