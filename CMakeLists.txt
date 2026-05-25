cmake_minimum_required(VERSION 3.20)
project(zazaki_tokenizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_CLI "Build CLI tool" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" OFF)

# Fetch RE2 for regex pre-tokenization
include(FetchContent)
FetchContent_Declare(
    re2
    GIT_REPOSITORY https://github.com/google/re2.git
    GIT_TAG 2024-07-02
)
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(re2)

# Fetch pybind11 if Python bindings enabled
if(BUILD_PYTHON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Fetch GoogleTest if tests enabled
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_subdirectory(src/core)
if(BUILD_CLI)
    add_subdirectory(src/cli)
endif()
if(BUILD_PYTHON)
    add_subdirectory(src/bindings)
endif()
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
