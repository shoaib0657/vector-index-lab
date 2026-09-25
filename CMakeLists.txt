cmake_minimum_required(VERSION 3.20)
project(hnsw_vector_search LANGUAGES CXX)

enable_testing()

add_executable(toolchain_smoke tests/toolchain_smoke.cpp)
target_compile_features(toolchain_smoke PRIVATE cxx_std_20)
set_target_properties(toolchain_smoke PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME toolchain_smoke COMMAND toolchain_smoke)
