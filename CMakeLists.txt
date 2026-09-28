cmake_minimum_required(VERSION 3.20)
project(hnsw_vector_search LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

add_library(vector_search src/exact_search.cpp)
target_compile_features(vector_search PUBLIC cxx_std_20)
target_include_directories(vector_search PUBLIC include)

add_executable(toolchain_smoke tests/toolchain_smoke.cpp)
target_compile_features(toolchain_smoke PRIVATE cxx_std_20)
add_test(NAME toolchain_smoke COMMAND toolchain_smoke)

function(add_vector_search_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE vector_search)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_vector_search_test(exact_search_test tests/exact_search_test.cpp)
add_vector_search_test(exact_search_edge_test tests/exact_search_edge_test.cpp)
