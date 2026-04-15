cmake_minimum_required(VERSION 3.16)
project(bench_umfpack CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Platform-specific optimization flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    # T-Head C910: RV64GCV — enable vector extension explicitly
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=rv64gcv -mtune=thead-c910")
else()
    # x86-64 and AArch64: use native detection
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

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

find_path(UMFPACK_INCLUDE_DIR umfpack.h PATHS ${SS_SEARCH_PATHS})
find_library(UMFPACK_LIB NAMES umfpack PATHS ${SS_LIB_PATHS})
find_library(AMD_LIB NAMES amd PATHS ${SS_LIB_PATHS})
find_library(COLAMD_LIB NAMES colamd PATHS ${SS_LIB_PATHS})
find_library(SUITESPARSECONFIG_LIB NAMES suitesparseconfig PATHS ${SS_LIB_PATHS})
find_library(BLAS_LIB NAMES openblas blas PATHS ${SS_LIB_PATHS})

if(NOT UMFPACK_INCLUDE_DIR OR NOT UMFPACK_LIB)
    message(FATAL_ERROR "UMFPACK not found. Install libsuitesparse-dev or suite-sparse via Homebrew.")
endif()

add_executable(bench_umfpack bench_umfpack.cpp)
target_include_directories(bench_umfpack PRIVATE ${UMFPACK_INCLUDE_DIR})
target_link_libraries(bench_umfpack PRIVATE
    ${UMFPACK_LIB} ${AMD_LIB} ${COLAMD_LIB}
    ${SUITESPARSECONFIG_LIB} ${BLAS_LIB}
    m
)
