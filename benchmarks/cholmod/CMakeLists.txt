cmake_minimum_required(VERSION 3.16)
project(bench_cholmod CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Search paths: system + Homebrew (macOS) + optional SUITESPARSE_ROOT
set(SS_SEARCH_PATHS
    /usr/include/suitesparse
    /usr/local/include/suitesparse
    /opt/homebrew/include/suitesparse
    /opt/homebrew/opt/suite-sparse/include/suitesparse
)
set(SS_LIB_PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /usr/lib/riscv64-linux-gnu
    /usr/lib
    /usr/local/lib
    /opt/homebrew/lib
    /opt/homebrew/opt/suite-sparse/lib
)
if(DEFINED SUITESPARSE_ROOT)
    list(PREPEND SS_SEARCH_PATHS "${SUITESPARSE_ROOT}/include/suitesparse")
    list(PREPEND SS_LIB_PATHS "${SUITESPARSE_ROOT}/lib")
endif()

find_path(CHOLMOD_INCLUDE_DIR cholmod.h PATHS ${SS_SEARCH_PATHS})

foreach(LIB cholmod amd camd colamd ccolamd suitesparseconfig)
    find_library(${LIB}_LIB NAMES ${LIB} PATHS ${SS_LIB_PATHS})
endforeach()

find_library(BLAS_LIB NAMES openblas blas PATHS ${SS_LIB_PATHS})

if(NOT CHOLMOD_INCLUDE_DIR OR NOT cholmod_LIB)
    message(FATAL_ERROR "CHOLMOD not found. Install libsuitesparse-dev or suite-sparse via Homebrew.")
endif()

add_executable(bench_cholmod bench_cholmod.cpp)
target_include_directories(bench_cholmod PRIVATE ${CHOLMOD_INCLUDE_DIR})

# OpenMP is optional on macOS
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_compile_options(bench_cholmod PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(bench_cholmod PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(bench_cholmod PRIVATE
    ${cholmod_LIB} ${amd_LIB} ${camd_LIB}
    ${colamd_LIB} ${ccolamd_LIB}
    ${suitesparseconfig_LIB} ${BLAS_LIB}
    m
)
