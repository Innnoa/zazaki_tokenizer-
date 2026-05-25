# Embed .tiktoken files as C arrays
set(TIKTOKEN_FILES
    ${CMAKE_SOURCE_DIR}/data/cl100k_base.tiktoken
    ${CMAKE_SOURCE_DIR}/data/o200k_base.tiktoken
)

foreach(tik_file ${TIKTOKEN_FILES})
    get_filename_component(tik_name ${tik_file} NAME_WE)
    set(embed_file ${CMAKE_CURRENT_BINARY_DIR}/${tik_name}_data.c)
    add_custom_command(
        OUTPUT ${embed_file}
        COMMAND xxd -i ${tik_file} ${embed_file}
        DEPENDS ${tik_file}
        COMMENT "Embedding ${tik_name}.tiktoken"
    )
    list(APPEND EMBED_SOURCES ${embed_file})
endforeach()

add_library(zazaki_tokenizer STATIC
    tokenizer.cpp
    registry.cpp
    openai/tiktoken_file.cpp
    openai/encoder.cpp
    ${EMBED_SOURCES}
)

target_include_directories(zazaki_tokenizer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(zazaki_tokenizer PUBLIC re2::re2)
