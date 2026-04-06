add_executable(bitsqz_llm_walkthrough)

target_sources(bitsqz_llm_walkthrough
    PRIVATE
        bitsqz_llm_walkthrough.cpp
)

target_link_libraries(bitsqz_llm_walkthrough
    PRIVATE
        bitsqz_llm::bitsqz_llm
)
