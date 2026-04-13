#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Build and run all SuiteSparse benchmarks.
#
# Host requirement: x86-64 machine running Linux (Ubuntu 24.04 recommended)
#
# Before running this script, complete the following steps:
#
# 1. Install system packages:
#
#   sudo apt install -y build-essential cmake git wget m4 \
#       gcc-riscv64-linux-gnu g++-riscv64-linux-gnu gfortran-riscv64-linux-gnu \
#       gcc-aarch64-linux-gnu g++-aarch64-linux-gnu gfortran-aarch64-linux-gnu \
#       qemu-user-static libsuitesparse-dev
#
# 2. Build OpenBLAS, GMP, MPFR and SuiteSparse for RISC-V -> $HOME/riscv-libs
#    Build OpenBLAS, GMP, MPFR and SuiteSparse for AArch64 -> $HOME/arm64-libs
#    Build GraphBLAS for x86-64 -> $HOME/x86-libs
#    See README.md for detailed build instructions.
#
# 3. Download the test matrix:
#
#   cd matrices
#   wget "https://suitesparse-collection-website.herokuapp.com/MM/Pajek/HEP-th-new.tar.gz"
#   tar -xzf HEP-th-new.tar.gz && rm HEP-th-new.tar.gz
#
# Usage:
#   ./scripts/run_all.sh
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO_ROOT/results"
RISCV_LIBS="${RISCV_LIBS:-$HOME/riscv-libs}"
ARM64_LIBS="${ARM64_LIBS:-$HOME/arm64-libs}"
X86_LIBS="${X86_LIBS:-$HOME/x86-libs}"
MATRIX="$REPO_ROOT/matrices/HEP-th-new/HEP-th-new.mtx"

mkdir -p "$RESULTS"

echo "=== SuiteSparse benchmark suite ==="
echo "Results will be saved to: $RESULTS"
echo ""

# ── Build ──────────────────────────────────────────────────────────────────────

build() {
    local bench=$1 arch=$2
    local build_dir="$REPO_ROOT/benchmarks/$bench/build_${arch}"
    mkdir -p "$build_dir" && cd "$build_dir"

    local libs cross_cxx ss_include ss_lib ext blas_lib
    case "$arch" in
        x86_64)
            libs="/usr"
            cross_cxx=""
            ss_include="/usr/include/suitesparse"
            ss_lib="/usr/lib/x86_64-linux-gnu"
            ext="so"
            blas_lib="$ss_lib/libblas.so"
            ;;
        riscv64)
            libs="$RISCV_LIBS"
            cross_cxx="-DCMAKE_CXX_COMPILER=riscv64-linux-gnu-g++"
            ss_include="$libs/include/suitesparse"
            ss_lib="$libs/lib"
            ext="a"
            blas_lib="$ss_lib/libopenblas.a"
            ;;
        aarch64)
            libs="$ARM64_LIBS"
            cross_cxx="-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++"
            ss_include="$libs/include/suitesparse"
            ss_lib="$libs/lib"
            ext="a"
            blas_lib="$ss_lib/libopenblas.a"
            ;;
    esac

    case "$bench" in
        cholmod)
            cmake .. -DCMAKE_BUILD_TYPE=Release $cross_cxx \
                -DCHOLMOD_INCLUDE_DIR="$ss_include" \
                -Dcholmod_LIB="$ss_lib/libcholmod.$ext" \
                -Damd_LIB="$ss_lib/libamd.$ext" \
                -Dcamd_LIB="$ss_lib/libcamd.$ext" \
                -Dcolamd_LIB="$ss_lib/libcolamd.$ext" \
                -Dccolamd_LIB="$ss_lib/libccolamd.$ext" \
                -Dsuitesparseconfig_LIB="$ss_lib/libsuitesparseconfig.$ext" \
                -DBLAS_LIB="$blas_lib" > /dev/null
            ;;
        klu)
            cmake .. -DCMAKE_BUILD_TYPE=Release $cross_cxx \
                -DKLU_INCLUDE_DIR="$ss_include" \
                -Dklu_LIB="$ss_lib/libklu.$ext" \
                -Damd_LIB="$ss_lib/libamd.$ext" \
                -Dbtf_LIB="$ss_lib/libbtf.$ext" \
                -Dcolamd_LIB="$ss_lib/libcolamd.$ext" \
                -Dsuitesparseconfig_LIB="$ss_lib/libsuitesparseconfig.$ext" \
                -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" > /dev/null
            ;;
        umfpack)
            local ss_root
            [ "$arch" = "x86_64" ] && ss_root="/usr" || ss_root="$libs"
            cmake .. -DCMAKE_BUILD_TYPE=Release $cross_cxx \
                -DSUITESPARSE_ROOT="$ss_root" \
                -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" > /dev/null
            ;;
        graphblas)
            local gb_inc gb_lib gb_cfg
            if [ "$arch" = "x86_64" ]; then
                gb_inc="$X86_LIBS/include/suitesparse"
                gb_lib="$X86_LIBS/lib/libgraphblas.so"
                gb_cfg="$X86_LIBS/lib/libsuitesparseconfig.so"
            else
                gb_inc="$ss_include"
                gb_lib="$ss_lib/libgraphblas.so"
                gb_cfg="$ss_lib/libsuitesparseconfig.so"
            fi
            cmake .. -DCMAKE_BUILD_TYPE=Release $cross_cxx \
                -DGRAPHBLAS_INCLUDE_DIR="$gb_inc" \
                -DGRAPHBLAS_LIB="$gb_lib" \
                -DCONFIG_LIB="$gb_cfg" > /dev/null
            ;;
    esac

    make -j"$(nproc)" > /dev/null
    cd "$REPO_ROOT"
}

# ── Run ────────────────────────────────────────────────────────────────────────

run() {
    local name=$1 binary=$2 arch=$3 args=${4:-}
    local out="$RESULTS/${name}_${arch}.txt"
    echo ">>> Running $name ($arch)..."
    case "$arch" in
        x86_64)
            LD_LIBRARY_PATH="$X86_LIBS/lib" "$binary" $args > "$out" 2>&1
            ;;
        riscv64)
            qemu-riscv64-static -L /usr/riscv64-linux-gnu \
                -E LD_LIBRARY_PATH="$RISCV_LIBS/lib" \
                "$binary" $args > "$out" 2>&1
            ;;
        aarch64)
            qemu-aarch64-static -L /usr/aarch64-linux-gnu \
                -E LD_LIBRARY_PATH="$ARM64_LIBS/lib" \
                "$binary" $args > "$out" 2>&1
            ;;
    esac
    echo "    saved to $out"
}

# ── Build all benchmarks ───────────────────────────────────────────────────────

for bench in cholmod klu umfpack graphblas; do
    for arch in x86_64 riscv64 aarch64; do
        echo "Building $bench ($arch)..."
        build "$bench" "$arch"
    done
done

echo ""

# ── Run all benchmarks ─────────────────────────────────────────────────────────

for arch in x86_64 riscv64 aarch64; do
    run cholmod   "$REPO_ROOT/benchmarks/cholmod/build_${arch}/bench_cholmod"     "$arch"
    run klu       "$REPO_ROOT/benchmarks/klu/build_${arch}/bench_klu"             "$arch"
    run umfpack   "$REPO_ROOT/benchmarks/umfpack/build_${arch}/bench_umfpack"     "$arch" "100 100"
    run graphblas "$REPO_ROOT/benchmarks/graphblas/build_${arch}/bench_graphblas" "$arch" "$MATRIX"
done

echo ""
echo "=== All benchmarks complete. Results in $RESULTS ==="