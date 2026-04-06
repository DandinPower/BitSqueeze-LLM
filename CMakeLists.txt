cmake_minimum_required(VERSION 3.23)

project(bitsqz_llm VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CMakePackageConfigHelpers)
include(CMakeFindDependencyMacro)
include(GNUInstallDirs)
include(CTest)

# Compile the generation after A100 (https://developer.nvidia.com/cuda/gpus)
set(CMAKE_CUDA_ARCHITECTURES 80)    #

option(BITSQZ_LLM_BUILD_EXAMPLES "Build bitsqz_llm examples" ON)

find_package(CUDAToolkit REQUIRED)

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

add_subdirectory(legacy_modules/quantization)
add_subdirectory(modules/quantization_cuda)

add_subdirectory(modules/svd_lowrank_cuda)

add_subdirectory(modules/topk_cuda)

add_subdirectory(modules/reconstruct_lowrank_cuda)

add_library(bitsqz_llm_lib)
add_library(bitsqz_llm::bitsqz_llm ALIAS bitsqz_llm_lib)

find_package(OpenMP)

target_sources(bitsqz_llm_lib
    PRIVATE 
        src/bitsqz_llm.cu
)
target_include_directories(bitsqz_llm_lib
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(bitsqz_llm_lib
    PUBLIC
        CUDA::cudart
    PRIVATE
        quantization_lib
        quantization_cuda_lib
        topk_cuda_lib
        svd_lowrank_cuda_lib
        reconstruct_lowrank_cuda_lib
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bitsqz_llm_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(bitsqz_llm_lib PRIVATE BITSQZ_LLM_EXPORTS)
set_target_properties(bitsqz_llm_lib PROPERTIES
    EXPORT_NAME bitsqz_llm
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(BITSQZ_LLM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

set(bitsqz_llm_install_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/bitsqz_llm")

install(
    TARGETS
        quantization_lib
        quantization_cuda_lib
        svd_lowrank_cuda_lib
        topk_cuda_lib
        reconstruct_lowrank_cuda_lib
        bitsqz_llm_lib
    EXPORT bitsqz_llmTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/legacy_modules/quantization/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/quantization_cuda/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/svd_lowrank_cuda/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/reconstruct_lowrank_cuda/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/topk_cuda/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bitsqz_llmConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/bitsqz_llmConfig.cmake
    INSTALL_DESTINATION ${bitsqz_llm_install_cmakedir}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/bitsqz_llmConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/bitsqz_llmConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/bitsqz_llmConfigVersion.cmake
    DESTINATION ${bitsqz_llm_install_cmakedir}
)

install(
    EXPORT bitsqz_llmTargets
    NAMESPACE bitsqz_llm::
    DESTINATION ${bitsqz_llm_install_cmakedir}
)
