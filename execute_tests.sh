#!/bin/bash
set -e

export RUST_BACKTRACE=1
export RUST_MIN_STACK=33554432
export GALLIVM_PERF=no_opt
export LP_DEBUG=no_opt

# TODO: We will remove this when we figure out why it acts out
export AETHERVK_DISABLE_GPU_AV=1

# Check if running in a headless environment and Vulkan ICD is not already overridden
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ] && [ -z "$VK_ICD_FILENAMES" ]; then
    echo "No display server detected. Configuring Vulkan for headless environment (lavapipe)..."
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
fi

SINGLE_THREAD=0

# Parse arguments
for arg in "$@"; do
    if [[ "$arg" == "--single-thread" ]] || [[ "$arg" == "-s" ]]; then
        SINGLE_THREAD=1
    fi
done

echo "========================================"
echo "         Running Rust Tests             "
echo "========================================"
# Keep going on a Rust failure so the C# suite still runs; the exit status reports both
RUST_STATUS=0
if [ $SINGLE_THREAD -eq 1 ]; then
    echo "Mode: Single-threaded, No Capture"
    cargo llvm-cov nextest --locked --no-capture --test-threads 1 || RUST_STATUS=$?
else
    echo "Mode: Default"
    cargo llvm-cov nextest --locked || RUST_STATUS=$?
fi

# Records the first failing exit status and carries on
run_rust() {
    local status=0
    "$@" || status=$?
    if [ $status -ne 0 ] && [ $RUST_STATUS -eq 0 ]; then
        RUST_STATUS=$status
    fi
}

NEXTEST_EXTRA=()
if [ $SINGLE_THREAD -eq 1 ]; then
    NEXTEST_EXTRA=(--no-capture --test-threads 1)
fi

echo ""
echo "========================================"
echo "     FITS tests: CFITSIO and fallback   "
echo "========================================"
# The run above builds the default members together, so the core gets the cdylib's default
# `cfitsio` feature. These runs pin each FITS reader explicitly: CFITSIO (--features cfitsio, plus
# the wrapper crate's own tests) and the pure-Rust fallback (no features), reusing the coverage build.
run_rust cargo llvm-cov nextest --locked --no-clean -p aethervk-cfitsio-rlib "${NEXTEST_EXTRA[@]}"
run_rust cargo llvm-cov nextest --locked --no-clean -p aethervk-core-rlib --features cfitsio -E 'test(/fits/)' "${NEXTEST_EXTRA[@]}"
run_rust cargo llvm-cov nextest --locked --no-clean -p aethervk-core-rlib -E 'test(/fits/)' "${NEXTEST_EXTRA[@]}"

echo ""
echo "========================================"
echo "          Running C# Tests              "
echo "========================================"
if [ $SINGLE_THREAD -eq 1 ]; then
    echo "Mode: Single-threaded, No Capture (Verbose)"
    # -m:1 disables MSBuild node parallelization. 
    # Console logger with normal verbosity prevents the runner from capturing/hiding standard output.
    dotnet test -m:1 --logger "console;verbosity=normal" --collect:"XPlat Code Coverage"
else
    echo "Mode: Default"
    dotnet test --collect:"XPlat Code Coverage"
fi

echo ""
echo "========================================"
echo "  Tests and Coverage Generation Done!   "
echo "========================================"

if [ $RUST_STATUS -ne 0 ]; then
    echo "Rust tests failed (exit $RUST_STATUS)"
    exit $RUST_STATUS
fi
