#!/usr/bin/env bash
# =============================================================================
# run_riscv64.sh — Build and run all SuiteSparse benchmarks natively on RISC-V.
#
# Target architecture: RISC-V 64-bit (RV64GC)
# Host requirement:    RISC-V 64-bit machine running Linux (Ubuntu 24.04)
#
# Before running this script:
#
# 1. Install system packages:
#
#   sudo apt install -y build-essential cmake libsuitesparse-dev
#
# 2. Download the test matrix:
#
#   cd matrices
#   curl -L "https://suitesparse-collection-website.herokuapp.com/MM/Pajek/HEP-th-new.tar.gz" \
#       -o HEP-th-new.tar.gz
#   tar -xzf HEP-th-new.tar.gz && rm HEP-th-new.tar.gz
#
# Note: On RISC-V, libsuitesparse-dev provides GraphBLAS 7.4.0.
#       The GraphBLAS benchmark is compatible with both 7.x and 10.x.
#
# Usage:
#   ./scripts/run_riscv64.sh
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO_ROOT/results"
MATRIX="$REPO_ROOT/matrices/HEP-th-new/HEP-th-new.mtx"

# Detect GraphBLAS library path
GB_LIB=$(find /usr -name "libgraphblas.so*" -not -name "*.7*" -not -name "*.4*" \
    2>/dev/null | head -1)
if [ -z "$GB_LIB" ]; then
    GB_LIB=$(find /usr -name "libgraphblas.so*" 2>/dev/null | head -1)
fi
GB_INCLUDE=$(find /usr/include -name "GraphBLAS.h" 2>/dev/null | \
    xargs -I{} dirname {} | head -1)
CFG_LIB=$(find /usr -name "libsuitesparseconfig.so" 2>/dev/null | head -1)

mkdir -p "$RESULTS"

echo "=== SuiteSparse benchmark suite (RISC-V native) ==="
echo "Results will be saved to: $RESULTS"
echo ""

build() {
    local bench=$1
    local build_dir="$REPO_ROOT/benchmarks/$bench/build_riscv64"
    mkdir -p "$build_dir" && cd "$build_dir"

    case "$bench" in
        cholmod)
            cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null ;;
        klu)
            cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null ;;
        umfpack)
            cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null ;;
        graphblas)
            g++ -O3 -DNDEBUG -std=c++11 \
                -I/usr/include -I/usr/include/suitesparse \
                "$REPO_ROOT/benchmarks/graphblas/bench_graphblas.cpp" \
                -o bench_graphblas \
                "$GB_LIB" "$CFG_LIB" -lm -lpthread ;;
    esac

    [ "$bench" != "graphblas" ] && make -j"$(nproc)" > /dev/null
    cd "$REPO_ROOT"
}

run() {
    local name=$1 binary=$2 args=${3:-}
    local out="$RESULTS/${name}_riscv64.txt"
    echo ">>> Running $name..."
    "$binary" $args > "$out" 2>&1
    echo "    saved to $out"
}

for bench in cholmod klu umfpack graphblas; do
    echo "Building $bench (riscv64)..."
    build "$bench"
done

echo ""

run cholmod   "$REPO_ROOT/benchmarks/cholmod/build_riscv64/bench_cholmod"
run klu       "$REPO_ROOT/benchmarks/klu/build_riscv64/bench_klu"
run umfpack   "$REPO_ROOT/benchmarks/umfpack/build_riscv64/bench_umfpack" "100 100"
run graphblas "$REPO_ROOT/benchmarks/graphblas/build_riscv64/bench_graphblas" "$MATRIX"

echo ""
echo "=== Done. Results in $RESULTS ==="
