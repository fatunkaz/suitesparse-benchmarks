// SPDX-License-Identifier: MIT
/**
 * @file bench_cholmod.cpp
 * @brief CHOLMOD benchmark reproducing the Ceres Solver factorization scenario.
 *
 * Measures symbolic analysis (once), repeated Cholesky numeric factorization,
 * and triangular solve on a discrete Laplacian matrix (SPD, lower-triangular CSC).
 * Intended for cross-architecture performance comparison: x86-64, RISC-V, AArch64.
 */

#include "cholmod.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

/** Number of numeric factorization iterations for averaging. */
#define N_ITER 20

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
 * @brief Computes the relative residual norm ||Ax - b||_2 / ||b||_2.
 *
 * Uses CHOLMOD sparse-dense multiply: r = b - A*x, then computes norms.
 * @param A  Sparse matrix (full symmetric storage).
 * @param x  Solution vector (dense).
 * @param b  Right-hand side vector (dense).
 * @param cc CHOLMOD common workspace.
 * @return Relative residual norm.
 */
static double residual_norm(cholmod_sparse *A, cholmod_dense *x, cholmod_dense *b,
                            cholmod_common *cc) {
    cholmod_dense *r = cholmod_copy_dense(b, cc);
    double alpha[2] = {-1.0, 0.0};
    double beta[2] = {1.0, 0.0};
    /* r = beta*b + alpha*A*x  =>  r = b - A*x */
    cholmod_sdmult(A, 0, alpha, beta, x, r, cc);

    double *rv = (double *)r->x;
    double *bv = (double *)b->x;
    int n = (int)b->nrow;

    double norm_r = 0.0, norm_b = 0.0;
    for (int i = 0; i < n; i++) {
        norm_r += rv[i] * rv[i];
        norm_b += bv[i] * bv[i];
    }
    cholmod_free_dense(&r, cc);
    return (norm_b > 0.0) ? sqrt(norm_r) / sqrt(norm_b) : sqrt(norm_r);
}

/**
 * @brief Builds a discrete Laplacian matrix on an nx x ny grid.
 *
 * The resulting matrix is symmetric positive definite (SPD) stored in
 * CHOLMOD lower-triangular format. Reproduces the structure of normal
 * equations arising in Ceres Solver bundle adjustment problems.
 *
 * @param nx  Grid width.
 * @param ny  Grid height.
 * @param cc  CHOLMOD common workspace.
 * @return    Pointer to CHOLMOD sparse matrix, or NULL on failure.
 */
static cholmod_sparse *build_laplacian(int nx, int ny, cholmod_common *cc) {
    int n = nx * ny;
    int nnz = 3 * n - nx - ny; /* lower triangle: diagonal + left + bottom */

    cholmod_triplet *T = cholmod_allocate_triplet(n, n, nnz, 1, CHOLMOD_REAL, cc);
    if (!T)
        return NULL;

    int *Ti = (int *)T->i;
    int *Tj = (int *)T->j;
    double *Tx = (double *)T->x;
    int cnt = 0;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            int k = iy * nx + ix;

            /* Diagonal entry: degree + shift to ensure positive definiteness */
            double diag = 4.0;
            if (ix > 0)
                diag += 1.0;
            if (ix < nx - 1)
                diag += 1.0;
            if (iy > 0)
                diag += 1.0;
            if (iy < ny - 1)
                diag += 1.0;

            Ti[cnt] = k;
            Tj[cnt] = k;
            Tx[cnt] = diag;
            cnt++;

            /* Left neighbour (lower triangle only) */
            if (ix > 0) {
                Ti[cnt] = k;
                Tj[cnt] = k - 1;
                Tx[cnt] = -1.0;
                cnt++;
            }
            /* Bottom neighbour (lower triangle only) */
            if (iy > 0) {
                Ti[cnt] = k;
                Tj[cnt] = k - nx;
                Tx[cnt] = -1.0;
                cnt++;
            }
        }
    }

    T->nnz = cnt;
    cholmod_sparse *A = cholmod_triplet_to_sparse(T, cnt, cc);
    cholmod_free_triplet(&T, cc);
    return A;
}

/**
 * @brief Benchmark entry point.
 *
 * Usage: bench_cholmod [nx ny]
 * Default grid size: 316 x 316 (n = 99856).
 */
int main(int argc, char **argv) {
    int nx = 316, ny = 316;
    if (argc >= 3) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
    }

    cholmod_common cc;
    cholmod_start(&cc);
    cc.print = 0; /* suppress CHOLMOD diagnostic output */

    printf("=== CHOLMOD benchmark (Ceres scenario) ===\n");
    printf("Grid: %d x %d, matrix size: %d x %d\n", nx, ny, nx * ny, nx * ny);

    cholmod_sparse *A = build_laplacian(nx, ny, &cc);
    if (!A) {
        fprintf(stderr, "Failed to build matrix\n");
        return 1;
    }
    long nnz_A = (long)A->nzmax;
    printf("nnz (lower triangle): %ld\n", nnz_A);

    /* --- Symbolic analysis (once, result cached for all numeric iterations) --- */
    double t0 = now();
    cholmod_factor *L = cholmod_analyze(A, &cc);
    double t_symbolic = now() - t0;
    if (!L) {
        fprintf(stderr, "cholmod_analyze failed\n");
        return 1;
    }
    printf("Symbolic analysis:    %.6f s\n", t_symbolic);

    /* --- Numeric factorization (N_ITER times for stable average) --- */
    double t_numeric_total = 0.0;
    for (int iter = 0; iter < N_ITER; iter++) {
        double t1 = now();
        int ok = cholmod_factorize(A, L, &cc);
        t_numeric_total += now() - t1;
        if (!ok || cc.status != CHOLMOD_OK) {
            fprintf(stderr, "cholmod_factorize failed at iter %d\n", iter);
            return 1;
        }
    }
    double t_numeric_avg = t_numeric_total / N_ITER;
    printf("Numeric factorization (%d iters): total %.6f s, avg %.6f s\n", N_ITER, t_numeric_total,
           t_numeric_avg);

    /* Fill-in ratio: nnz(L) / nnz(A_full) */
    long nnz_L = (long)cc.lnz;
    long nnz_full = 2 * nnz_A - nx * ny; /* symmetrise lower triangle */
    double fill_ratio = (nnz_full > 0) ? (double)nnz_L / (double)nnz_full : 0.0;
    printf("Fill-in: nnz(L)=%ld, nnz(A_full)=%ld, ratio=%.2f\n", nnz_L, nnz_full, fill_ratio);

    /* --- Triangular solve --- */
    int n = nx * ny;
    cholmod_dense *b = cholmod_ones(n, 1, CHOLMOD_REAL, &cc);
    double t2 = now();
    cholmod_dense *x = cholmod_solve(CHOLMOD_A, L, b, &cc);
    double t_solve = now() - t2;
    printf("Solve:                %.6f s\n", t_solve);

    /* --- Residual check --- */
    cholmod_sparse *A_full = cholmod_copy(A, 0, 1, &cc); /* full symmetric copy */
    double rel_res = residual_norm(A_full, x, b, &cc);
    printf("Residual ||Ax-b||/||b||: %.2e\n", rel_res);
    cholmod_free_sparse(&A_full, &cc);

    long rss = peak_rss_kb();
    printf("Peak RSS:             %ld KB\n", rss);

    /* --- Performance metrics --- */
    /* Cholesky factorization: approx 2*nnz(L) floating-point operations */
    double flop_numeric = 2.0 * (double)nnz_L;
    double mflops_numeric = (t_numeric_avg > 0.0) ? (flop_numeric / t_numeric_avg) / 1e6 : 0.0;

    /* Triangular solve: approx 2*nnz(L) floating-point operations */
    double flop_solve = 2.0 * (double)nnz_L;
    double mflops_solve = (t_solve > 0.0) ? (flop_solve / t_solve) / 1e6 : 0.0;

    /* Memory bandwidth estimate for numeric factorization:
     * reads L (nnz_L doubles) + reads A (nnz_A doubles) + writes L (nnz_L doubles) */
    double bytes_numeric = (double)(2 * nnz_L + nnz_A) * sizeof(double);
    double bandwidth_numeric_gbs =
        (t_numeric_avg > 0.0) ? (bytes_numeric / t_numeric_avg) / 1e9 : 0.0;

    printf("--- Summary ---\n");
    printf("symbolic:  %.6f s\n", t_symbolic);
    printf("numeric:   %.6f s (avg over %d iters)\n", t_numeric_avg, N_ITER);
    printf("solve:     %.6f s\n", t_solve);
    printf("fill-in:   %.2f\n", fill_ratio);
    printf("residual:  %.2e\n", rel_res);
    printf("peak RSS:  %ld KB\n", rss);
    printf("numeric:   %.2f MFLOP/s\n", mflops_numeric);
    printf("solve:     %.2f MFLOP/s\n", mflops_solve);
    printf("bandwidth: %.2f GB/s (numeric factorization)\n", bandwidth_numeric_gbs);
    double ai_numeric =
        (bandwidth_numeric_gbs > 0.0) ? (mflops_numeric / 1000.0) / bandwidth_numeric_gbs : 0.0;
    printf("arith. intensity: %.4f FLOP/byte (numeric)\n", ai_numeric);

    cholmod_free_sparse(&A, &cc);
    cholmod_free_factor(&L, &cc);
    cholmod_free_dense(&b, &cc);
    cholmod_free_dense(&x, &cc);
    cholmod_finish(&cc);
    return 0;
}
