#!/bin/bash
# set -e: Instructs bash to immediately exit if any command returns a non-zero exit status (i.e., if anything fails).
set -e

# echo: Prints to the console that we are cleaning up previous build artifacts.
echo '=== Wiping aethervk-core-rlib artifacts ==='

# rm -rf: Forcefully removes directories and files recursively without prompting.
# This cleans up the fingerprint, dependency tracking, main library outputs, and incremental build cache 
# for 'aethervk' to ensure a completely fresh build for the tests.
rm -rf /build-target/debug/.fingerprint/aethervk* \
       /build-target/debug/deps/aethervk* \
       /build-target/debug/aethervk* \
       /build-target/debug/incremental/aethervk*

# echo: Prints that we are now building the test binary.
echo '=== Building test binary (no-run) ==='

# cargo test --no-run: Compiles the tests for the 'aethervk-core-rlib' package but does not execute them.
# --features: Enables specific features 'collisions' and 'shader_debug_sync' for the build.
# 2>&1: Redirects standard error to standard output so both can be piped.
# | tail -5: Pipes the combined output and only prints the last 5 lines (usually the build summary/success message).
cargo test --no-run \
  --features collisions,shader_debug_sync \
  -p aethervk-core-rlib 2>&1 | tail -5

# BIN=$(...): Finds the exact path to the newly compiled test executable and saves it in the BIN variable.
# ls /build-target/debug/deps/...: Lists the built test binaries.
# 2>/dev/null: Silences any "No such file or directory" errors from ls.
# grep -v '\.d$': Excludes dependency files (.d) from the list.
# head -1: Takes the first resulting file path.
BIN=$(ls /build-target/debug/deps/aethervk_core_rlib-* 2>/dev/null | grep -v '\.d$' | head -1)

# echo: Prints the path of the found test binary executable.
echo "=== Test binary: $BIN ==="

# echo: Prints that the GDB debugging session is starting.
echo '=== Running under GDB ==='

# LVP_DEBUG=noopt disables lavapipe's LLVM shader optimisations.
# This prevents the ARM64 JIT register-aliasing crash where x30 is
# reused as both a loop counter and a constant-pool base address.
export LVP_DEBUG=noopt

# LP_NUM_THREADS=1: Restricts Lavapipe to use only 1 thread. 
# This makes debugging significantly easier by avoiding concurrent thread interference and making execution deterministic.
export LP_NUM_THREADS=1

# === GDB Batch Mode Execution ===
# xvfb-run -a: Runs GDB within a virtual X11 server environment (required for Vulkan/graphics without a physical display).
# gdb -batch: Runs GDB in non-interactive batch mode. It executes the provided '-ex' commands sequentially and then automatically exits.
#
# GDB Commands executed in order:
# -ex 'set pagination off'               : Prevents GDB from pausing and prompting "Type <return> to continue" when printing long outputs like full backtraces.
# -ex 'set print thread-events off'      : Suppresses noisy messages about threads starting or exiting during execution.
# -ex 'handle SIGSEGV stop print pass'   : Configures GDB to intercept segmentation faults (crashes). It will stop execution, print a message, and pass the signal to the program.
# -ex 'run'                              : Starts the execution of the test binary.
# -ex 'bt full'                          : If the program crashes, this prints a full backtrace of the call stack, including the values of all local variables in each frame.
# -ex 'x/20i ($pc-32)'                   : Examines and prints 20 assembly instructions starting 32 bytes before the program counter ($pc). This shows the exact assembly code around the crash site.
# -ex 'info registers'                   : Prints the current values of all CPU registers at the exact moment the program crashed.
#
# Optional Advanced Debugging Commands (You can add these as -ex flags before --args when needed):
#
# -ex 'info proc mappings' : Prints the memory map of the process (showing which memory addresses correspond to which files/libraries or heap/stack).
#                            WHEN TO ADD: Use this when debugging segfaults to see if the crashed address ($pc) or accessed memory is within a valid mapped region, especially useful for JIT-compiled code (like Lavapipe shaders) or memory corruption issues.
#
# -ex 'info symbol $pc'    : Looks up and prints the symbol (function name and offset) corresponding to the current program counter ($pc).
#                            WHEN TO ADD: Use this if the backtrace only gives raw memory addresses without function names. It helps pinpoint exactly which function the crash occurred in (very common in dynamically generated code).
#
# -ex 'info sharedlibrary' : Lists all loaded shared libraries (.so files) along with the memory address ranges where they are mapped.
#                            WHEN TO ADD: Use this when you suspect a crash is caused by a library version mismatch, missing symbols, or to determine which specific library a crashed memory address belongs to.
#
# --args "$BIN" ...                      : Tells GDB which executable to debug, followed by the arguments passed to that executable.
# physics::...::test_energy_conservation_bounce : The specific Rust test function to run.
# --nocapture                            : Prevents the Rust test runner from hiding stdout/stderr, ensuring any print statements in the code are visible.
# 2>&1                                   : Redirects standard error to standard output so all GDB and test output is combined into a single stream.
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
