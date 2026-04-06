add_library(quantization_lib)

find_package(OpenMP)

target_sources(quantization_lib
    PRIVATE
        src/quantization.cpp
        src/impl/q8_0_impl.cpp
        src/impl/q4_0_impl.cpp
        src/impl/q2_k_impl.cpp
        src/impl/q2_k_fast_impl.cpp
        src/impl/bf16_impl.cpp
        src/impl/fp16_impl.cpp
        src/impl/fp8_impl.cpp
        src/impl/fp4_impl.cpp
        src/impl/mxfp8_impl.cpp
        src/impl/mxfp4_impl.cpp
        src/impl/nvfp4_impl.cpp
        src/impl/nf4_impl.cpp
        src/impl/nf4_dq_impl.cpp
)

target_include_directories(quantization_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${CMAKE_SOURCE_DIR}/modules/quantization_cuda/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(quantization_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
