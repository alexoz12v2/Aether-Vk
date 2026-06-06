#!/bin/bash
set -e

echo "Compiling and running tests (this might take a few minutes, please wait)..."
# We force colors so cargo prints even without a TTY
export CARGO_TERM_COLOR=always

# We run all tests in the package to truly mirror CI, or just the physics ones
xvfb-run -a cargo nextest run --features "collisions,shader_debug_sync" -E "package(aethervk-core-rlib)" --no-fail-fast
