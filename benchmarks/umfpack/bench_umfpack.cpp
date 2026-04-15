// SPDX-License-Identifier: MIT
/**
 * @file bench_umfpack.cpp
 * @brief UMFPACK benchmark reproducing the SciPy spsolve scenario.
 *
 * Measures symbolic analysis (once) and repeated numeric LU factorization
 * without caching (mimicking scipy.sparse.linalg.spsolve without factorized()),
 * followed by a single triangular solve. Matrix is an unsymmetric convection-
 * diffusion operator on a regular grid, typical for SciPy sparse linear algebra.
 * Intended for cross-architecture performance comparison: x86-64, RISC-V, AArch64.
 */

#include "umfpack.h"
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
 * @brief Computes y = A*x for a CSC sparse matrix A.
 *
 * @param n   Matrix dimension.
 * @param Ap  Column pointers.
 * @param Ai  Row indices.
 * @param Ax  Non-zero values.
 * @param x   Input vector.
 * @param y   Output vector (overwritten).
 */
static void csc_matvec(int n, const int *Ap, const int *Ai, const double *Ax, const double *x,
                       double *y) {
    memset(y, 0, n * sizeof(double));
    for (int j = 0; j < n; j++) {
        double xj = x[j];
        for (int idx = Ap[j]; idx < Ap[j + 1]; idx++)
            y[Ai[idx]] += Ax[idx] * xj;
    }
}

/**
 * @brief Builds an unsymmetric sparse matrix in CSC format.
 *
 * Discretises a 2D convection-diffusion equation on an nx x ny grid using
 * a 5-point stencil with asymmetric convective coefficients (-1.0 left/bottom,
 * -1.5 right/top). The matrix is diagonally dominant, ensuring a well-posed
 * linear system. Typical for SciPy sparse solver benchmarks.
 *
 * @param nx      Grid width.
 * @param ny      Grid height.
 * @param Ap_out  Output column pointers.
 * @param Ai_out  Output row indices.
 * @param Ax_out  Output non-zero values.
 * @param nnz_out Output number of non-zeros.
 */
static void build_unsymmetric_csc(int nx, int ny, int **Ap_out, int **Ai_out, double **Ax_out,
                                  int *nnz_out) {
    int n = nx * ny;
    int max_nnz = 5 * n;

    /* Step 1: collect entries in coordinate (triplet) format */
    int *Ti = (int *)malloc(max_nnz * sizeof(int));
    int *Tj = (int *)malloc(max_nnz * sizeof(int));
    double *Tx = (double *)malloc(max_nnz * sizeof(double));
    int cnt = 0;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            int k = iy * nx + ix;

            /* Diagonal entry: base value + contributions from neighbours */
            double diag = 4.0;
            if (ix > 0)
                diag += 0.5; /* left neighbour contribution  */
            if (ix < nx - 1)
                diag += 0.5; /* right neighbour contribution */
            if (iy > 0)
                diag += 0.5; /* bottom neighbour contribution */
            if (iy < ny - 1)
                diag += 0.5; /* top neighbour contribution   */

            Ti[cnt] = k;
            Tj[cnt] = k;
            Tx[cnt] = diag;
            cnt++;

            /* Off-diagonal entries with asymmetric convective coefficients */
            if (ix > 0) {
                Ti[cnt] = k;
                Tj[cnt] = k - 1;
                Tx[cnt] = -1.0;
                cnt++;
            }
            if (ix < nx - 1) {
                Ti[cnt] = k;
                Tj[cnt] = k + 1;
                Tx[cnt] = -1.5;
                cnt++;
            }
            if (iy > 0) {
                Ti[cnt] = k;
                Tj[cnt] = k - nx;
                Tx[cnt] = -1.0;
                cnt++;
            }
            if (iy < ny - 1) {
                Ti[cnt] = k;
                Tj[cnt] = k + nx;
                Tx[cnt] = -1.5;
                cnt++;
            }
        }
    }

    /* Step 2: convert triplet to CSC by counting per-column entries */
    int *Ap = (int *)calloc(n + 1, sizeof(int));
    int *Ai = (int *)malloc(cnt * sizeof(int));
    double *Ax = (double *)malloc(cnt * sizeof(double));
    int *pos = (int *)malloc((n + 1) * sizeof(int));

    for (int i = 0; i < cnt; i++)
        Ap[Tj[i] + 1]++;
    for (int j = 0; j < n; j++)
        Ap[j + 1] += Ap[j];
    memcpy(pos, Ap, (n + 1) * sizeof(int));

    for (int i = 0; i < cnt; i++) {
        int j = Tj[i], p = pos[j]++;
        Ai[p] = Ti[i];
        Ax[p] = Tx[i];
    }

    /* Step 3: sort row indices within each column (insertion sort).
     * Acceptable cost: stencil has at most 5 entries per column. */
    for (int j = 0; j < n; j++) {
        int start = Ap[j], end = Ap[j + 1];
        for (int i = start + 1; i < end; i++) {
            int key_i = Ai[i];
            double key_x = Ax[i];
            int m = i - 1;
            while (m >= start && Ai[m] > key_i) {
                Ai[m + 1] = Ai[m];
                Ax[m + 1] = Ax[m];
                m--;
            }
            Ai[m + 1] = key_i;
            Ax[m + 1] = key_x;
        }
    }

    *Ap_out = Ap;
    *Ai_out = Ai;
    *Ax_out = Ax;
    *nnz_out = cnt;
    free(Ti);
    free(Tj);
    free(Tx);
    free(pos);
}

/**
 * @brief Benchmark entry point.
 *
 * Usage: bench_umfpack [nx ny]
 * Default grid size: 100 x 100 (n = 10000).
 */
int main(int argc, char **argv) {
    int nx = 100, ny = 100;
    if (argc >= 3) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
    }

    int *Ap = NULL, *Ai = NULL, nnz = 0;
    double *Ax = NULL;
    build_unsymmetric_csc(nx, ny, &Ap, &Ai, &Ax, &nnz);
    int n = nx * ny;

    printf("=== UMFPACK benchmark (SciPy spsolve scenario) ===\n");
    printf("Grid: %d x %d, matrix size: %d x %d\n", nx, ny, n, n);
    printf("nnz: %d\n", nnz);

    double control[UMFPACK_CONTROL];
    double info[UMFPACK_INFO];
    umfpack_di_defaults(control);
    control[UMFPACK_PRL] = 0; /* suppress UMFPACK diagnostic output */

    void *Symbolic = NULL;
    void *Numeric = NULL;

    /* --- Symbolic analysis (once) --- */
    double t0 = now();
    int status = umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, control, info);
    double t_symbolic = now() - t0;
    if (status != UMFPACK_OK) {
        fprintf(stderr, "umfpack_di_symbolic failed (status %d)\n", status);
        return 1;
    }
    printf("Symbolic analysis:    %.6f s\n", t_symbolic);

    /* --- Numeric factorization (N_ITER times, Numeric recreated each time) ---
     * Mimics scipy.sparse.linalg.spsolve() without factorized() caching. */
    double t_numeric_total = 0.0;
    for (int iter = 0; iter < N_ITER; iter++) {
        if (Numeric)
            umfpack_di_free_numeric(&Numeric);
        double t1 = now();
        status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, control, info);
        t_numeric_total += now() - t1;
        if (status != UMFPACK_OK) {
            fprintf(stderr, "umfpack_di_numeric failed at iter %d\n", iter);
            return 1;
        }
    }
    double t_numeric_avg = t_numeric_total / N_ITER;
    printf("Numeric factorization (%d iters): total %.6f s, avg %.6f s\n", N_ITER, t_numeric_total,
           t_numeric_avg);

    /* Fill-in ratio: (nnz(L) + nnz(U)) / nnz(A) */
    int lnz = 0, unz = 0, nr = 0, nc = 0, nz_diag = 0;
    umfpack_di_get_lunz(&lnz, &unz, &nr, &nc, &nz_diag, Numeric);
    double fill_ratio = (nnz > 0) ? (double)(lnz + unz) / (double)nnz : 0.0;
    printf("Fill-in: nnz(L+U)=%d, nnz(A)=%d, ratio=%.2f\n", lnz + unz, nnz, fill_ratio);

    /* --- Triangular solve --- */
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    double *r = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        b[i] = 1.0;

    double t2 = now();
    status = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, control, info);
    double t_solve = now() - t2;
    printf("Solve:                %.6f s\n", t_solve);
    if (status != UMFPACK_OK)
        fprintf(stderr, "umfpack_di_solve failed\n");

    /* --- Residual check --- */
    csc_matvec(n, Ap, Ai, Ax, x, r);
    double norm_r = 0.0, norm_b = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = r[i] - b[i];
        norm_r += diff * diff;
        norm_b += b[i] * b[i];
    }
    double rel_res = (norm_b > 0.0) ? sqrt(norm_r) / sqrt(norm_b) : sqrt(norm_r);
    printf("Residual ||Ax-b||/||b||: %.2e\n", rel_res);

    long rss = peak_rss_kb();
    printf("Peak RSS:             %ld KB\n", rss);

    /* --- Performance metrics --- */
    /* LU factorization: approx 2*nnz(L+U) floating-point operations */
    double flop_numeric = 2.0 * (double)(lnz + unz);
    double mflops_numeric = (t_numeric_avg > 0.0) ? (flop_numeric / t_numeric_avg) / 1e6 : 0.0;

    /* Triangular solve: approx 2*nnz(L+U) floating-point operations */
    double flop_solve = 2.0 * (double)(lnz + unz);
    double mflops_solve = (t_solve > 0.0) ? (flop_solve / t_solve) / 1e6 : 0.0;

    /* Memory bandwidth estimate for numeric factorization */
    double bytes_numeric = (double)(2 * (lnz + unz) + nnz) * sizeof(double);
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

    umfpack_di_free_symbolic(&Symbolic);
    umfpack_di_free_numeric(&Numeric);
    free(Ap);
    free(Ai);
    free(Ax);
    free(b);
    free(x);
    free(r);
    return 0;
}
