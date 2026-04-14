cmake_minimum_required(VERSION 3.16)
project(bench_graphblas CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Detect system library path based on architecture
if(NOT DEFINED SUITESPARSE_ROOT)
    execute_process(
        COMMAND dpkg-architecture -qDEB_HOST_MULTIARCH
        OUTPUT_VARIABLE DEB_HOST_MULTIARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(DEB_HOST_MULTIARCH)
        set(SYS_LIB_PATH "/usr/lib/${DEB_HOST_MULTIARCH}")
    else()
        set(SYS_LIB_PATH "/usr/lib")
    endif()
endif()

find_path(GRAPHBLAS_INCLUDE_DIR GraphBLAS.h
    PATHS
        /usr/include/suitesparse
        /usr/local/include/suitesparse
        ${SUITESPARSE_ROOT}/include/suitesparse
)

find_library(GRAPHBLAS_LIB NAMES graphblas
    PATHS
        ${SYS_LIB_PATH}
        /usr/lib
        /usr/local/lib
        ${SUITESPARSE_ROOT}/lib
)

find_library(CONFIG_LIB NAMES suitesparseconfig
    PATHS
        ${SYS_LIB_PATH}
        /usr/lib
        /usr/local/lib
        ${SUITESPARSE_ROOT}/lib
)

if(NOT GRAPHBLAS_INCLUDE_DIR OR NOT GRAPHBLAS_LIB OR NOT CONFIG_LIB)
    message(FATAL_ERROR "GraphBLAS not found. Pass -DSUITESPARSE_ROOT=<path> or install libgraphblas-dev.")
endif()

add_executable(bench_graphblas bench_graphblas.cpp)
target_include_directories(bench_graphblas PRIVATE ${GRAPHBLAS_INCLUDE_DIR})
target_link_libraries(bench_graphblas PRIVATE
    ${GRAPHBLAS_LIB}
    ${CONFIG_LIB}
    m pthread
)
