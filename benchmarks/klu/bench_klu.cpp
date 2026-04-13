// SPDX-License-Identifier: MIT
/**
 * @file bench_klu.cpp
 * @brief KLU benchmark reproducing the SUNDIALS ODE integrator scenario.
 *
 * Measures BTF symbolic analysis (once), first LU factorization, repeated
 * refactorization (numeric only, structure fixed), and triangular solve on a
 * block-diagonal tridiagonal matrix typical for spatially discretised ODE systems.
 * Intended for cross-architecture performance comparison: x86-64, RISC-V, AArch64.
 */

#define _DARWIN_C_SOURCE
#include "klu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

/**
 * @brief Returns the current time in seconds (monotonic clock).
 * @return Time in seconds as a double.
 */
static double get_time_sec() {
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
 * @brief Computes the relative residual norm ||Ax - b||_2 / ||b||_2.
 *
 * @param n   Matrix dimension.
 * @param Ap  Column pointers (CSC format).
 * @param Ai  Row indices (CSC format).
 * @param Ax  Non-zero values (CSC format).
 * @param x   Solution vector.
 * @param b   Right-hand side vector.
 * @return Relative residual norm.
 */
static double residual_norm(int n, int *Ap, int *Ai, double *Ax, double *x, double *b) {
    double *r = (double *)malloc(n * sizeof(double));
    memcpy(r, b, n * sizeof(double));
    for (int j = 0; j < n; j++) {
        for (int p = Ap[j]; p < Ap[j + 1]; p++)
            r[Ai[p]] -= Ax[p] * x[j];
    }
    double norm_r = 0.0, norm_b = 0.0;
    for (int i = 0; i < n; i++) {
        norm_r += r[i] * r[i];
        norm_b += b[i] * b[i];
    }
    free(r);
    return (norm_b > 0.0) ? sqrt(norm_r) / sqrt(norm_b) : sqrt(norm_r);
}

/**
 * @brief Builds a block-diagonal tridiagonal matrix in CSC format.
 *
 * The matrix consists of @p nb blocks of size @p bs x @p bs, each being
 * a tridiagonal matrix with values depending on @p scale. Diagonal entries
 * are scale*(2 + 0.05*cos(row)), off-diagonal entries are -scale*(1 + 0.1*sin(row)).
 * This structure is typical for spatially discretised ODE Jacobians in SUNDIALS.
 *
 * @param nb    Number of diagonal blocks.
 * @param bs    Block size.
 * @param Ap    Output column pointers (must be pre-allocated, size N+1).
 * @param Ai    Output row indices (must be pre-allocated, size NNZ).
 * @param Ax    Output values (must be pre-allocated, size NNZ).
 * @param scale Scaling factor for matrix entries.
 */
static void build_block_diagonal(int nb, int bs, int *Ap, int *Ai, double *Ax, double scale) {
    int nnz = 0;
    Ap[0] = 0;
    for (int b = 0; b < nb; b++) {
        for (int i = 0; i < bs; i++) {
            int row = b * bs + i;
            if (i > 0) {
                Ai[nnz] = row - 1;
                Ax[nnz] = -scale * (1.0 + 0.1 * sin(row));
                nnz++;
            }
            Ai[nnz] = row;
            Ax[nnz] = scale * (2.0 + 0.05 * cos(row));
            nnz++;
            if (i < bs - 1) {
                Ai[nnz] = row + 1;
                Ax[nnz] = -scale * (1.0 + 0.1 * sin(row + 1));
                nnz++;
            }
            Ap[row + 1] = nnz;
        }
    }
}

/**
 * @brief Updates non-zero values of the block-diagonal matrix in place.
 *
 * Simulates the Jacobian update at each integrator step: structure is fixed,
 * only numerical values change. Corresponds to klu_refactor usage in SUNDIALS.
 *
 * @param nb    Number of diagonal blocks.
 * @param bs    Block size.
 * @param Ax    Values array to update in place.
 * @param scale New scaling factor.
 */
static void update_values(int nb, int bs, double *Ax, double scale) {
    int nnz = 0;
    for (int b = 0; b < nb; b++) {
        for (int i = 0; i < bs; i++) {
            int row = b * bs + i;
            if (i > 0)
                Ax[nnz++] = -scale * (1.0 + 0.1 * sin(row + scale));
            Ax[nnz++] = scale * (2.0 + 0.05 * cos(row + scale));
            if (i < bs - 1)
                Ax[nnz++] = -scale * (1.0 + 0.1 * sin(row + 1 + scale));
        }
    }
}

/**
 * @brief Benchmark entry point.
 *
 * Matrix: 50 blocks x 200 (N = 10000, NNZ = 29900).
 */
int main(void) {
    const int NB = 50, BS = 200, N = NB * BS, N_ITER = 20;
    const int NNZ = NB * (3 * BS - 2);

    printf("=== KLU benchmark (SUNDIALS scenario) ===\n");
    printf("Blocks: %d x %d, matrix size: %d x %d\n", NB, BS, N, N);
    printf("nnz: %d\n", NNZ);

    int *Ap = (int *)malloc((N + 1) * sizeof(int));
    int *Ai = (int *)malloc(NNZ * sizeof(int));
    double *Ax = (double *)malloc(NNZ * sizeof(double));
    double *b = (double *)malloc(N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));

    build_block_diagonal(NB, BS, Ap, Ai, Ax, 1.0);
    for (int i = 0; i < N; i++)
        b[i] = 1.0;

    klu_common common;
    klu_defaults(&common);

    /* --- Symbolic analysis with BTF reordering (once) --- */
    double t0 = get_time_sec();
    klu_symbolic *Symbolic = klu_analyze(N, Ap, Ai, &common);
    double t_symbolic = get_time_sec() - t0;

    /* --- First full LU factorization --- */
    t0 = get_time_sec();
    klu_numeric *Numeric = klu_factor(Ap, Ai, Ax, Symbolic, &common);
    double t_factor_first = get_time_sec() - t0;

    /* Fill-in ratio: (nnz(L) + nnz(U)) / nnz(A) */
    long lnz = (long)Numeric->lnz;
    long unz = (long)Numeric->unz;
    double fill_ratio = (NNZ > 0) ? (double)(lnz + unz) / NNZ : 0.0;

    /* --- Refactorization loop (structure fixed, values change each step) --- */
    double t_refactor_total = 0.0;
    for (int iter = 0; iter < N_ITER; iter++) {
        update_values(NB, BS, Ax, 1.0 + 0.01 * (iter + 1));
        double t1 = get_time_sec();
        klu_refactor(Ap, Ai, Ax, Symbolic, Numeric, &common);
        t_refactor_total += get_time_sec() - t1;
    }
    double t_refactor_avg = t_refactor_total / N_ITER;

    /* --- Triangular solve --- */
    memcpy(x, b, N * sizeof(double));
    t0 = get_time_sec();
    klu_solve(Symbolic, Numeric, N, 1, x, &common);
    double t_solve = get_time_sec() - t0;

    double rel_res = residual_norm(N, Ap, Ai, Ax, x, b);
    long rss = peak_rss_kb();

    printf("Symbolic analysis (BTF):     %f s\n", t_symbolic);
    printf("First factorization:         %f s\n", t_factor_first);
    printf("Refactorization (%d iters): total %f s, avg %f s\n", N_ITER, t_refactor_total,
           t_refactor_avg);
    printf("Solve:                       %f s\n", t_solve);
    printf("Fill-in: lnz=%ld, unz=%ld, ratio=%.2f\n", lnz, unz, fill_ratio);
    printf("Residual ||Ax-b||/||b||:     %.2e\n", rel_res);
    printf("Peak RSS:                    %ld KB\n", rss);
    printf("--- Summary ---\n");
    printf("symbolic:  %f s\n", t_symbolic);
    printf("factor:    %f s\n", t_factor_first);
    printf("refactor:  %f s (avg over %d iters)\n", t_refactor_avg, N_ITER);
    printf("solve:     %f s\n", t_solve);
    printf("fill-in:   %.2f\n", fill_ratio);
    printf("residual:  %.2e\n", rel_res);
    printf("peak RSS:  %ld KB\n", rss);

    klu_free_numeric(&Numeric, &common);
    klu_free_symbolic(&Symbolic, &common);
    free(Ap);
    free(Ai);
    free(Ax);
    free(b);
    free(x);
    return 0;
}
