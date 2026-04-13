cmake_minimum_required(VERSION 3.16)
project(bench_graphblas CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_path(GRAPHBLAS_INCLUDE_DIR GraphBLAS.h
    PATHS
        /usr/include/suitesparse
        /usr/local/include/suitesparse
        ${SUITESPARSE_ROOT}/include/suitesparse
)

find_library(GRAPHBLAS_LIB NAMES graphblas
    PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
        ${SUITESPARSE_ROOT}/lib
)

find_library(CONFIG_LIB NAMES suitesparseconfig
    PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
        ${SUITESPARSE_ROOT}/lib
)

add_executable(bench_graphblas bench_graphblas.cpp)
target_include_directories(bench_graphblas PRIVATE ${GRAPHBLAS_INCLUDE_DIR})
target_link_libraries(bench_graphblas PRIVATE
    ${GRAPHBLAS_LIB}
    ${CONFIG_LIB}
    m pthread
)
