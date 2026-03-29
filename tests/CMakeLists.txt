add_executable(svd_lowrank_cuda_tests)

target_sources(svd_lowrank_cuda_tests
    PRIVATE
        svd_lowrank_cuda/test_svd_lowrank_cuda.cu
)

target_link_libraries(svd_lowrank_cuda_tests
    PRIVATE
        svd_lowrank_cuda_lib
)