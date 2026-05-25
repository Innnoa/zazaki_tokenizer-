pybind11_add_module(_zazaki_tokenizer python_bindings.cpp)
target_link_libraries(_zazaki_tokenizer PRIVATE zazaki_tokenizer_core)
set_target_properties(_zazaki_tokenizer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/zazaki_tokenizer")
