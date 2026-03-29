add_library(quantization_lib)

target_sources(quantization_lib
    PRIVATE
        src/quantization.cpp
        src/impl/nf4_dq_impl.cpp
)

target_include_directories(quantization_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
)