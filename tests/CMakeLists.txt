add_executable(svd_lowrank_cuda_tests)

target_sources(svd_lowrank_cuda_tests
    PRIVATE
        svd_lowrank_cuda/test_svd_lowrank_cuda.cu
)

target_link_libraries(svd_lowrank_cuda_tests
    PRIVATE
        svd_lowrank_cuda_lib
)

add_executable(quantization_nf4_dq_tests)

target_sources(quantization_nf4_dq_tests
    PRIVATE
        quantization/test_nf4_dq_impl.cpp
)

target_link_libraries(quantization_nf4_dq_tests
    PRIVATE
        quantization_lib
)