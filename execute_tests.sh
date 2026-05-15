#!/bin/bash
set -e

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
if [ $SINGLE_THREAD -eq 1 ]; then
    echo "Mode: Single-threaded, No Capture"
    cargo llvm-cov nextest --no-capture --test-threads 1
else
    echo "Mode: Default"
    cargo llvm-cov nextest
fi

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
