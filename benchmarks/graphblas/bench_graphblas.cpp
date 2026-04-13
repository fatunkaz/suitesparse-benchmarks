// SPDX-License-Identifier: MIT
/**
 * @file bench_graphblas.cpp
 * @brief GraphBLAS benchmark reproducing the NetworkX graph algorithm scenario.
 *
 * Loads a real-world directed graph from a Matrix Market file and performs
 * sparse matrix-matrix multiplication (SpGEMM): C = A * A, computing weighted
 * length-2 paths in the citation graph. Uses the GrB_PLUS_TIMES semiring over FP64.
 * Intended for cross-architecture performance comparison: x86-64, RISC-V, AArch64.
 */

#include "GraphBLAS.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

/** Number of SpGEMM iterations for averaging (after one warm-up run). */
#define N_ITER 10

/**
 * @brief Returns the current time in seconds (monotonic clock).
 * @return Time in seconds as a double.
 */
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/**
 * @brief Returns peak resident set size (RSS) of the process in kilobytes.
 *
 * On macOS, ru_maxrss is in bytes; on Linux, it is in kilobytes.
 * @return Peak RSS in KB, or -1 on error.
 */
static long peak_rss_kb(void) {
#ifdef __APPLE__
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
        return usage.ru_maxrss / 1024;
    return -1;
#else
    long rss = 0;
    FILE *f = fopen("/proc/self/status", "r");
    if (!f)
        return -1;
    char line[128];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmPeak:", 7) == 0) {
            sscanf(line + 7, "%ld", &rss);
            break;
        }
    }
    fclose(f);
    return rss;
#endif
}

/**
 * @brief Reads a sparse matrix from a Matrix Market (.mtx) file into a GrB_Matrix.
 *
 * Supports general and symmetric patterns, as well as binary (pattern-only) and
 * real-valued entries. MTX indices are 1-based and converted to 0-based internally.
 * For symmetric matrices, both (i,j) and (j,i) entries are inserted.
 *
 * @param filename Path to the .mtx file.
 * @return Allocated GrB_Matrix (FP64), or NULL on failure.
 */
static GrB_Matrix read_mtx(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", filename);
        return NULL;
    }

    char line[256];
    int is_symmetric = 0;
    int is_pattern = 0;

    /* Parse Matrix Market header comments */
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '%') {
            if (strstr(line, "symmetric"))
                is_symmetric = 1;
            if (strstr(line, "pattern"))
                is_pattern = 1;
            continue;
        }
        break;
    }

    /* Read matrix dimensions and number of stored entries */
    long nrows, ncols, nnz;
    sscanf(line, "%ld %ld %ld", &nrows, &ncols, &nnz);

    GrB_Matrix A = NULL;
    GrB_Matrix_new(&A, GrB_FP64, (GrB_Index)nrows, (GrB_Index)ncols);

    long cnt = 0;
    while (fgets(line, sizeof(line), f) && cnt < nnz) {
        if (line[0] == '%')
            continue;
        long r, c;
        double v = 1.0;
        if (is_pattern)
            sscanf(line, "%ld %ld", &r, &c);
        else
            sscanf(line, "%ld %ld %lf", &r, &c, &v);

        /* Convert 1-based MTX indices to 0-based GraphBLAS indices */
        GrB_Matrix_setElement_FP64(A, v, (GrB_Index)(r - 1), (GrB_Index)(c - 1));
        if (is_symmetric && r != c)
            GrB_Matrix_setElement_FP64(A, v, (GrB_Index)(c - 1), (GrB_Index)(r - 1));
        cnt++;
    }
    fclose(f);

    /* Finalise internal GraphBLAS data structures */
    GrB_Matrix_wait(A, GrB_MATERIALIZE);
    return A;
}

/**
 * @brief Benchmark entry point.
 *
 * Usage: bench_graphblas [path/to/matrix.mtx]
 * Default matrix: ../../matrices/HEP-th-new/HEP-th-new.mtx
 */
int main(int argc, char **argv) {
    const char *mtx_file = "../../matrices/HEP-th-new/HEP-th-new.mtx";
    if (argc >= 2)
        mtx_file = argv[1];

    GrB_init(GrB_NONBLOCKING);

    printf("=== GraphBLAS benchmark (NetworkX SpGEMM scenario) ===\n");
    printf("Matrix file: %s\n", mtx_file);

    /* --- Load matrix from file --- */
    double t0 = now();
    GrB_Matrix A = read_mtx(mtx_file);
    double t_load = now() - t0;

    if (A == NULL) {
        fprintf(stderr, "Failed to load matrix\n");
        GrB_finalize();
        return 1;
    }

    GrB_Index nrows, ncols, nnz;
    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_ncols(&ncols, A);
    GrB_Matrix_nvals(&nnz, A);

    printf("Matrix: %llu x %llu, nnz: %llu\n", (unsigned long long)nrows, (unsigned long long)ncols,
           (unsigned long long)nnz);
    printf("Load time: %.6f s\n", t_load);

    /* --- SpGEMM: C = A * A on PLUS_TIMES semiring ---
     * Computes weighted length-2 paths in the citation graph.
     * One warm-up run initialises the GraphBLAS JIT cache. */
    GrB_Matrix C = NULL;
    GrB_Matrix_new(&C, GrB_FP64, nrows, ncols);

    GrB_mxm(C, GrB_NULL, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, A, GrB_NULL);
    GrB_Matrix_wait(C, GrB_MATERIALIZE);

    /* Timed iterations */
    double t_spgemm_total = 0.0;
    for (int iter = 0; iter < N_ITER; iter++) {
        GrB_Matrix_clear(C);
        double t1 = now();
        GrB_mxm(C, GrB_NULL, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, A, GrB_NULL);
        GrB_Matrix_wait(C, GrB_MATERIALIZE);
        t_spgemm_total += now() - t1;
    }
    double t_spgemm_avg = t_spgemm_total / N_ITER;

    GrB_Index nnz_C;
    GrB_Matrix_nvals(&nnz_C, C);

    long rss = peak_rss_kb();

    printf("SpGEMM A*A (%d iters): total %.6f s, avg %.6f s\n", N_ITER, t_spgemm_total,
           t_spgemm_avg);
    printf("nnz(C) = %llu\n", (unsigned long long)nnz_C);
    printf("Peak RSS: %ld KB\n", rss);
    printf("--- Summary ---\n");
    printf("load:     %.6f s\n", t_load);
    printf("spgemm:   %.6f s (avg over %d iters)\n", t_spgemm_avg, N_ITER);
    printf("nnz(C):   %llu\n", (unsigned long long)nnz_C);
    printf("peak RSS: %ld KB\n", rss);

    GrB_Matrix_free(&A);
    GrB_Matrix_free(&C);
    GrB_finalize();
    return 0;
}
