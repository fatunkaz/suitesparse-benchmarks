// SPDX-License-Identifier: MIT
/**
 * @file bench_graphblas.cpp
 * @brief GraphBLAS benchmark: SpGEMM A*A on a sparse citation graph.
 *
 * Scenario: NetworkX graph algorithm using GraphBLAS SpGEMM to compute
 * length-2 paths in the citation graph. Uses the GrB_PLUS_TIMES semiring
 * over FP64. Intended for cross-architecture performance comparison:
 * x86-64, RISC-V, AArch64.
 *
 * Compatible with SuiteSparse:GraphBLAS >= 7.x and >= 10.x
 */

#ifdef __cplusplus
extern "C" {
#endif
#include <GraphBLAS.h>
#ifdef __cplusplus
}
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#define N_ITER 10

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long peak_rss_kb(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        return usage.ru_maxrss / 1024;
#else
        return usage.ru_maxrss;
#endif
    }
    return -1;
}

/**
 * @brief Reads a sparse matrix from a Matrix Market (.mtx) file into a
 * GrB_Matrix. Supports coordinate format with real or pattern (binary)
 * real-valued entries. MTX indices are 1-based and converted to 0-based
 * internally. For symmetric matrices, both (i,j) and (j,i) entries are
 * inserted.
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

        GrB_Matrix_setElement_FP64(A, v, (GrB_Index)(r - 1), (GrB_Index)(c - 1));
        if (is_symmetric && r != c)
            GrB_Matrix_setElement_FP64(A, v, (GrB_Index)(c - 1), (GrB_Index)(r - 1));
        cnt++;
    }
    fclose(f);

    GrB_Matrix_wait(A, GrB_MATERIALIZE);
    return A;
}

/**
 * @brief Main benchmark entry point.
 */
int main(int argc, char **argv) {
    const char *mtx_file = "../../matrices/HEP-th-new/HEP-th-new.mtx";
    if (argc >= 2)
        mtx_file = argv[1];

    GrB_init(GrB_NONBLOCKING);

    printf("=== GraphBLAS benchmark (NetworkX SpGEMM scenario) ===\n");
    printf("Matrix file: %s\n", mtx_file);

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

    GrB_Matrix C = NULL;
    GrB_Matrix_new(&C, GrB_FP64, nrows, ncols);

    /* Warm-up run */
    GrB_mxm(C, GrB_NULL, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, A, GrB_NULL);
    GrB_Matrix_wait(C, GrB_MATERIALIZE);

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
