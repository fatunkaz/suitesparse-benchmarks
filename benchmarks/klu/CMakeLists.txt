cmake_minimum_required(VERSION 3.16)
project(bench_klu CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
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

find_path(KLU_INCLUDE_DIR klu.h PATHS ${SS_SEARCH_PATHS})

foreach(LIB klu amd btf colamd suitesparseconfig)
    find_library(${LIB}_LIB NAMES ${LIB} PATHS ${SS_LIB_PATHS})
endforeach()

if(NOT KLU_INCLUDE_DIR OR NOT klu_LIB)
    message(FATAL_ERROR "KLU not found. Install libsuitesparse-dev or suite-sparse via Homebrew.")
endif()

add_executable(bench_klu bench_klu.cpp)
target_include_directories(bench_klu PRIVATE ${KLU_INCLUDE_DIR})
target_link_libraries(bench_klu PRIVATE
    ${klu_LIB} ${amd_LIB} ${btf_LIB}
    ${colamd_LIB} ${suitesparseconfig_LIB}
    m
)
