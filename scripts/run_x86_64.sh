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
#   sudo apt install -y build-essential cmake git curl tar libsuitesparse-dev
#
# 2. Download the test matrix:
#
#   cd matrices
#   curl -L "https://suitesparse-collection-website.herokuapp.com/MM/Pajek/HEP-th-new.tar.gz" \
#       -o HEP-th-new.tar.gz
#   tar -xzf HEP-th-new.tar.gz && rm HEP-th-new.tar.gz
#
# Usage:
#   ./scripts/run_x86_64.sh
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO_ROOT/results"
MATRIX="$REPO_ROOT/matrices/HEP-th-new/HEP-th-new.mtx"

mkdir -p "$RESULTS"

echo "=== SuiteSparse benchmark suite (x86-64 native) ==="
echo "Results will be saved to: $RESULTS"
echo ""

build() {
    local bench=$1
    local build_dir="$REPO_ROOT/benchmarks/$bench/build_x86_64"
    mkdir -p "$build_dir" && cd "$build_dir"
    cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null
    make -j"$(nproc)" > /dev/null
    cd "$REPO_ROOT"
}

run() {
    local name=$1 binary=$2 args=${3:-}
    local out="$RESULTS/${name}_x86_64.txt"
    echo ">>> Running $name..."
    "$binary" $args > "$out" 2>&1
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
