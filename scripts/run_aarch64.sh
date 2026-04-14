#!/usr/bin/env bash
# =============================================================================
# run_aarch64.sh — Build and run all SuiteSparse benchmarks natively on AArch64.
#
# Target architecture: AArch64 (ARM64)
# Host requirement:    AArch64 machine running Linux, or Apple Silicon (macOS)
#
# === Linux (Ubuntu/Debian on AArch64) ===
#
# 1. Install system packages:
#
#   sudo apt install -y build-essential cmake libsuitesparse-dev
#
# 2. Build GraphBLAS 10.x:
#
#   git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
#   cd SuiteSparse && mkdir build_arm64 && cd build_arm64
#   cmake .. -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_INSTALL_PREFIX=$HOME/arm64-libs \
#       -DSUITESPARSE_ENABLE_PROJECTS="graphblas" \
#       -DSUITESPARSE_ENABLE_CUDA=OFF
#   make -j$(nproc) GraphBLAS && make install
#
# === macOS (Apple Silicon) ===
#
# 1. Install via Homebrew:
#
#   brew install suite-sparse cmake
#
# 2. Download the test matrix:
#
#   cd matrices
#   curl -L "https://suitesparse-collection-website.herokuapp.com/MM/Pajek/HEP-th-new.tar.gz" \
#       -o HEP-th-new.tar.gz
#   tar -xzf HEP-th-new.tar.gz && rm HEP-th-new.tar.gz
#
# Usage:
#   ./scripts/run_aarch64.sh
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO_ROOT/results"
MATRIX="$REPO_ROOT/matrices/HEP-th-new/HEP-th-new.mtx"

# Detect platform and set library paths
if [ "$(uname)" = "Darwin" ]; then
    # macOS with Homebrew
    SS_PREFIX="$(brew --prefix suite-sparse 2>/dev/null || echo /opt/homebrew)"
    SS_INCLUDE="$SS_PREFIX/include/suitesparse"
    SS_LIB="$SS_PREFIX/lib"
    GB_LIB="$SS_LIB/libgraphblas.dylib"
    CFG_LIB="$SS_LIB/libsuitesparseconfig.dylib"
    BLAS_LIB="$SS_LIB/libopenblas.dylib"
else
    # Linux AArch64
    ARM64_LIBS="${ARM64_LIBS:-$HOME/arm64-libs}"
    SS_INCLUDE="/usr/include/suitesparse"
    SS_LIB="/usr/lib/aarch64-linux-gnu"
    GB_LIB="$ARM64_LIBS/lib/libgraphblas.so"
    CFG_LIB="$SS_LIB/libsuitesparseconfig.so"
    BLAS_LIB="$SS_LIB/libblas.so"
fi

mkdir -p "$RESULTS"

echo "=== SuiteSparse benchmark suite (AArch64 native) ==="
echo "Platform: $(uname -s) $(uname -m)"
echo "Results will be saved to: $RESULTS"
echo ""

build() {
    local bench=$1
    local build_dir="$REPO_ROOT/benchmarks/$bench/build_aarch64"
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
                -DGRAPHBLAS_INCLUDE_DIR="$SS_INCLUDE" \
                -DGRAPHBLAS_LIB="$GB_LIB" \
                -DCONFIG_LIB="$CFG_LIB" > /dev/null ;;
    esac

    NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
    make -j"$NPROC" > /dev/null
    cd "$REPO_ROOT"
}

run() {
    local name=$1 binary=$2 args=${3:-}
    local out="$RESULTS/${name}_aarch64.txt"
    echo ">>> Running $name..."
    "$binary" $args > "$out" 2>&1
    echo "    saved to $out"
}

for bench in cholmod klu umfpack graphblas; do
    echo "Building $bench (aarch64)..."
    build "$bench"
done

echo ""

run cholmod   "$REPO_ROOT/benchmarks/cholmod/build_aarch64/bench_cholmod"
run klu       "$REPO_ROOT/benchmarks/klu/build_aarch64/bench_klu"
run umfpack   "$REPO_ROOT/benchmarks/umfpack/build_aarch64/bench_umfpack" "100 100"
run graphblas "$REPO_ROOT/benchmarks/graphblas/build_aarch64/bench_graphblas" "$MATRIX"

echo ""
echo "=== Done. Results in $RESULTS ==="
