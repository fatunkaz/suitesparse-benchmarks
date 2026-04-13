cmake_minimum_required(VERSION 3.10)
project(bench_umfpack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Search paths: system + optional SUITESPARSE_ROOT for cross-compilation
set(SS_SEARCH_PATHS
    /usr/include/suitesparse
    /usr/local/include/suitesparse
)
if(DEFINED SUITESPARSE_ROOT)
    list(PREPEND SS_SEARCH_PATHS "${SUITESPARSE_ROOT}/include/suitesparse")
endif()

set(SS_LIB_PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/lib
)
if(DEFINED SUITESPARSE_ROOT)
    list(PREPEND SS_LIB_PATHS "${SUITESPARSE_ROOT}/lib")
endif()

find_path(UMFPACK_INCLUDE_DIR NAMES umfpack.h PATHS ${SS_SEARCH_PATHS})
find_library(UMFPACK_LIB NAMES umfpack PATHS ${SS_LIB_PATHS})
find_library(AMD_LIB NAMES amd PATHS ${SS_LIB_PATHS})
find_library(COLAMD_LIB NAMES colamd PATHS ${SS_LIB_PATHS})
find_library(SUITESPARSECONFIG_LIB NAMES suitesparseconfig PATHS ${SS_LIB_PATHS})

if(NOT UMFPACK_INCLUDE_DIR OR NOT UMFPACK_LIB OR NOT AMD_LIB
        OR NOT COLAMD_LIB OR NOT SUITESPARSECONFIG_LIB)
    message(FATAL_ERROR "SuiteSparse components not found. "
        "Pass -DSUITESPARSE_ROOT=<path> for cross-compilation.")
endif()

add_executable(bench_umfpack bench_umfpack.cpp)
target_include_directories(bench_umfpack PRIVATE ${UMFPACK_INCLUDE_DIR})
target_compile_options(bench_umfpack PRIVATE -fopenmp)
target_link_libraries(bench_umfpack PRIVATE
    ${UMFPACK_LIB}
    ${AMD_LIB}
    ${COLAMD_LIB}
    ${SUITESPARSECONFIG_LIB}
    -fopenmp m
)
