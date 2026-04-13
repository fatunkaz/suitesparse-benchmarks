cmake_minimum_required(VERSION 3.16)
project(bench_cholmod CXX)

set(CMAKE_CXX_STANDARD 11)

find_path(CHOLMOD_INCLUDE_DIR cholmod.h
    PATHS
        /usr/include/suitesparse
        /usr/local/include/suitesparse
        ${SUITESPARSE_ROOT}/include/suitesparse
)

foreach(LIB cholmod amd camd colamd ccolamd suitesparseconfig)
    find_library(${LIB}_LIB NAMES ${LIB}
        PATHS
            /usr/lib/x86_64-linux-gnu
            /usr/lib
            /usr/local/lib
            ${SUITESPARSE_ROOT}/lib
    )
endforeach()

find_library(BLAS_LIB NAMES openblas blas
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib ${SUITESPARSE_ROOT}/lib
)

add_executable(bench_cholmod bench_cholmod.cpp)
target_include_directories(bench_cholmod PRIVATE ${CHOLMOD_INCLUDE_DIR})
target_compile_options(bench_cholmod PRIVATE -fopenmp)
target_link_libraries(bench_cholmod PRIVATE
    ${cholmod_LIB} ${amd_LIB} ${camd_LIB}
    ${colamd_LIB} ${ccolamd_LIB}
    ${suitesparseconfig_LIB} ${BLAS_LIB}
    -fopenmp m pthread
)