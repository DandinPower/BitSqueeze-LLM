add_executable(svd_lowrank_cuda_tests)

target_sources(svd_lowrank_cuda_tests
    PRIVATE
        svd_lowrank_cuda/test_svd_lowrank_cuda.cu
)

target_link_libraries(svd_lowrank_cuda_tests
    PRIVATE
        svd_lowrank_cuda_lib
)

function(add_quantization_test target_name source_file)
    add_executable(${target_name})
    target_sources(${target_name}
        PRIVATE
            ${source_file}
    )
    target_link_libraries(${target_name}
        PRIVATE
            quantization_lib
    )
endfunction()

function(add_topk_test target_name source_file)
    add_executable(${target_name})
    target_sources(${target_name}
        PRIVATE
            ${source_file}
    )
    target_link_libraries(${target_name}
        PRIVATE
            topk_lib
    )
endfunction()

add_quantization_test(quantization_q8_0_tests quantization/test_q8_0_impl.cpp)
add_quantization_test(quantization_q4_0_tests quantization/test_q4_0_impl.cpp)
add_quantization_test(quantization_q2_k_tests quantization/test_q2_k_impl.cpp)
add_quantization_test(quantization_q2_k_fast_tests quantization/test_q2_k_fast_impl.cpp)
add_quantization_test(quantization_bf16_tests quantization/test_bf16_impl.cpp)
add_quantization_test(quantization_fp16_tests quantization/test_fp16_impl.cpp)
add_quantization_test(quantization_fp8_tests quantization/test_fp8_impl.cpp)
add_quantization_test(quantization_fp4_tests quantization/test_fp4_impl.cpp)
add_quantization_test(quantization_mxfp8_tests quantization/test_mxfp8_impl.cpp)
add_quantization_test(quantization_mxfp4_tests quantization/test_mxfp4_impl.cpp)
add_quantization_test(quantization_nvfp4_tests quantization/test_nvfp4_impl.cpp)
add_quantization_test(quantization_nf4_tests quantization/test_nf4_impl.cpp)
add_quantization_test(quantization_nf4_dq_tests quantization/test_nf4_dq_impl.cpp)

add_topk_test(topk_extraction_tests topk/test_topk_extraction.cpp)
add_topk_test(topk_separation_tests topk/test_topk_separation.cpp)
