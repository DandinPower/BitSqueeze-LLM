add_library(topk_lib)

find_package(OpenMP)

target_sources(topk_lib
    PRIVATE
        src/topk.cpp
)

target_include_directories(topk_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(topk_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
