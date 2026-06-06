#!/bin/bash
set -e

echo '=== Wiping aethervk-core-rlib artifacts ==='
rm -rf /build-target/debug/.fingerprint/aethervk* \
       /build-target/debug/deps/aethervk* \
       /build-target/debug/aethervk* \
       /build-target/debug/incremental/aethervk*

echo '=== Building test binary (no-run) ==='
cargo test --no-run \
  --features collisions,shader_debug_sync \
  -p aethervk-core-rlib 2>&1 | tail -5

BIN=$(ls /build-target/debug/deps/aethervk_core_rlib-* 2>/dev/null | grep -v '\.d$' | head -1)
echo "=== Test binary: $BIN ==="

echo '=== Running under GDB ==='
xvfb-run -a gdb -batch \
  -ex 'set pagination off' \
  -ex 'set print thread-events off' \
  -ex 'handle SIGSEGV stop print pass' \
  -ex 'run' \
  -ex 'bt full' \
  -ex 'x/20i ($pc-32)' \
  -ex 'info registers' \
  --args "$BIN" \
  physics::vulkan_math_tests::tests::test_energy_conservation_bounce \
  --nocapture 2>&1
