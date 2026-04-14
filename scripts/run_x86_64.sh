#!/usr/bin/env bash
# =============================================================================
# run_x86_64.sh — Build and run all SuiteSparse benchmarks natively on x86-64.
#
# Target architecture: x86-64
# Host requirement:    x86-64 machine running Linux (Ubuntu 24.04 recommended)
#
# Before running this script:
#
# 1. Install system packages:
#
#   sudo apt install -y build-essential cmake libsuitesparse-dev
#
# 2. Build GraphBLAS 10.x for x86-64 (required for GraphBLAS benchmark):
#
#   git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
#   cd SuiteSparse && mkdir build_x86 && cd build_x86
#   cmake .. -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_INSTALL_PREFIX=$HOME/x86-libs \
#       -DSUITESPARSE_ENABLE_PROJECTS="graphblas" \
#       -DSUITESPARSE_ENABLE_CUDA=OFF
#   make -j$(nproc) GraphBLAS && make install
#   cp /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so* $HOME/x86-libs/lib/
#   cp /usr/include/suitesparse/SuiteSparse_config.h \
#       $HOME/x86-libs/include/suitesparse/
#
# 3. Download the test matrix:
#
#   cd matrices
#   wget "https://suitesparse-collection-website.herokuapp.com/MM/Pajek/HEP-th-new.tar.gz"
#   tar -xzf HEP-th-new.tar.gz && rm HEP-th-new.tar.gz
#
# Usage:
#   ./scripts/run_x86_64.sh
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO_ROOT/results"
X86_LIBS="${X86_LIBS:-$HOME/x86-libs}"
MATRIX="$REPO_ROOT/matrices/HEP-th-new/HEP-th-new.mtx"

mkdir -p "$RESULTS"

echo "=== SuiteSparse benchmark suite (x86-64 native) ==="
echo "Results will be saved to: $RESULTS"
echo ""

build() {
    local bench=$1
    local build_dir="$REPO_ROOT/benchmarks/$bench/build_x86_64"
    mkdir -p "$build_dir" && cd "$build_dir"

    case "$bench" in
        cholmod)
            cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null ;;
        klu)
            cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null ;;
        umfpack)
            cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null ;;
        graphblas)
            cmake .. -DCMAKE_BUILD_TYPE=Release \
                -DGRAPHBLAS_INCLUDE_DIR="$X86_LIBS/include/suitesparse" \
                -DGRAPHBLAS_LIB="$X86_LIBS/lib/libgraphblas.so" \
                -DCONFIG_LIB="$X86_LIBS/lib/libsuitesparseconfig.so" > /dev/null ;;
    esac

    make -j"$(nproc)" > /dev/null
    cd "$REPO_ROOT"
}

run() {
    local name=$1 binary=$2 args=${3:-}
    local out="$RESULTS/${name}_x86_64.txt"
    echo ">>> Running $name..."
    LD_LIBRARY_PATH="$X86_LIBS/lib" "$binary" $args > "$out" 2>&1
    echo "    saved to $out"
}

for bench in cholmod klu umfpack graphblas; do
    echo "Building $bench (x86-64)..."
    build "$bench"
done

echo ""

run cholmod   "$REPO_ROOT/benchmarks/cholmod/build_x86_64/bench_cholmod"
run klu       "$REPO_ROOT/benchmarks/klu/build_x86_64/bench_klu"
run umfpack   "$REPO_ROOT/benchmarks/umfpack/build_x86_64/bench_umfpack" "100 100"
run graphblas "$REPO_ROOT/benchmarks/graphblas/build_x86_64/bench_graphblas" "$MATRIX"

echo ""
echo "=== Done. Results in $RESULTS ==="
