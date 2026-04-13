cmake_minimum_required(VERSION 3.16)
project(bench_klu CXX)

set(CMAKE_CXX_STANDARD 11)

find_path(KLU_INCLUDE_DIR klu.h
    PATHS
        /usr/include/suitesparse
        /usr/local/include/suitesparse
        ${SUITESPARSE_ROOT}/include/suitesparse
)

foreach(LIB klu amd btf colamd suitesparseconfig)
    find_library(${LIB}_LIB NAMES ${LIB}
        PATHS
            /usr/lib/x86_64-linux-gnu
            /usr/lib
            /usr/local/lib
            ${SUITESPARSE_ROOT}/lib
    )
endforeach()

add_executable(bench_klu bench_klu.cpp)
target_include_directories(bench_klu PRIVATE ${KLU_INCLUDE_DIR})
target_compile_options(bench_klu PRIVATE -fopenmp)
target_link_libraries(bench_klu PRIVATE
    ${klu_LIB} ${amd_LIB} ${btf_LIB}
    ${colamd_LIB} ${suitesparseconfig_LIB}
    -fopenmp m
)
