# Embed .tiktoken files as C arrays
set(TIKTOKEN_FILES
    ${CMAKE_SOURCE_DIR}/data/cl100k_base.tiktoken
    ${CMAKE_SOURCE_DIR}/data/o200k_base.tiktoken
)

foreach(tik_file ${TIKTOKEN_FILES})
    get_filename_component(tik_name ${tik_file} NAME_WE)
    get_filename_component(tik_fname ${tik_file} NAME)
    set(embed_file ${CMAKE_CURRENT_BINARY_DIR}/${tik_name}_data.c)
    add_custom_command(
        OUTPUT ${embed_file}
        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/data xxd -i ${tik_fname} ${embed_file}
        DEPENDS ${tik_file}
        COMMENT "Embedding ${tik_name}.tiktoken"
    )
    list(APPEND EMBED_SOURCES ${embed_file})
endforeach()

add_library(zazaki_tokenizer_core STATIC
    tokenizer.cpp
    registry.cpp
    openai/tiktoken_file.cpp
    openai/encoder.cpp
    ${EMBED_SOURCES}
)

target_include_directories(zazaki_tokenizer_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(zazaki_tokenizer_core PUBLIC re2::re2)
set_target_properties(zazaki_tokenizer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
