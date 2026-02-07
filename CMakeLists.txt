cmake_minimum_required(VERSION 3.16)
project(NTT3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific flags
if(MSVC)
    add_compile_options(/arch:AVX2 /O2 /Oi /GL /fp:fast)
    add_link_options(/LTCG)
else()
    add_compile_options(-mavx2 -O2 -march=native)
endif()

# Header-only library
add_library(ntt_lib INTERFACE)
target_include_directories(ntt_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Correctness test
add_executable(test_correctness test_correctness.cpp)
target_link_libraries(test_correctness ntt_lib)

# Benchmark
add_executable(bench_sizes bench_sizes.cpp)
target_link_libraries(bench_sizes ntt_lib)
