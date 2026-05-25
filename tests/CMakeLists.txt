add_executable(test_zazaki_tokenizer
    test_encoder.cpp
    test_tokenizer.cpp
    test_registry.cpp
)
target_link_libraries(test_zazaki_tokenizer PRIVATE zazaki_tokenizer GTest::gtest GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_zazaki_tokenizer)
