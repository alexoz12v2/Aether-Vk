# Aether-Vk Testing Guide

This guide covers critical architectural details and practical tips for running the test suite effectively, particularly focusing on Vulkan GPU testing and common CI/CD pitfalls.

## Critical Architecture: AOSOA Memory Packing

Aether-Vk uses **Array of Structs of Arrays (AOSOA)** memory layouts for physics data (particles, rigid bodies, etc.) to maximize memory coalescing and GPU cache hits.

### Subgroup Size Dependency
The AOSOA block width is strictly tied to the Vulkan hardware's **Subgroup Size** (SIMD width).
- **Apple (Metal/MoltenVK):** 32
- **Nvidia:** 32
- **AMD:** 64
- **Lavapipe (llvmpipe / CI):** 4

**WARNING: Never hardcode subgroup sizes in tests!**
If you hardcode an assumed packing width (e.g., `32`) in your CPU test logic:
```rust
// ❌ WRONG: Assumes Apple/Nvidia hardware! Will fail on Lavapipe/AMD.
let packed = pack_particles_aosoa(&particles, 32, PARTICLE_FIELDS);
```
The compute shader (which dynamically compiles using the hardware's native subgroup size via specialization constants) will calculate completely different byte offsets. It will read `0.0` from empty padded slots instead of your actual data, causing silent failures, early returns (e.g., `mass <= 0.0`), and NaN explosions.

**Always query the device:**
```rust
// ✅ CORRECT: Dynamically matches the GPU's execution pipeline.
let sg = device.kernels.pipelines.subgroup_size as usize;
let packed = pack_particles_aosoa(&particles, sg, PARTICLE_FIELDS);
```

---

## Running Tests with Cargo Nextest

Aether-Vk uses `cargo nextest` for parallel, fast, and structured test execution. Standard `cargo test` is discouraged.

### Basic Commands
Run the entire suite (default profile):
```bash
cargo nextest run
```

Run a specific package:
```bash
cargo nextest run -p aethervk-core-rlib
```

### Filtering Tests
Nextest supports powerful filter expressions.
Run only tests matching a specific substring (e.g., `integrate_particles`):
```bash
cargo nextest run integrate_particles
```

Run tests using exact match (useful if substrings overlap):
```bash
cargo nextest run -- --exact integrate_particles_p1_p2
```

Use filter syntax to run all tests in a specific module:
```bash
cargo nextest run -E 'test(gpu_backends::vulkan::physics)'
```

### Debugging Failed Tests
By default, `nextest` captures standard output and standard error and only prints them if a test fails.

**View output inline:**
If you are iterating and want to see your `println!` macros in real-time, disable output capturing:
```bash
cargo nextest run --no-capture integrate_particles_p1_p2
```

**Single-threaded execution:**
If you have global Vulkan instances failing due to concurrency or you are debugging a race condition, run tests on a single thread:
```bash
cargo nextest run --test-threads 1
```

**Retries:**
If a test is flaky due to random OS scheduling or driver warmup, you can automatically retry it:
```bash
cargo nextest run --retries 2
```

**Debugger Attach:**
Nextest runs each test in its own isolated process, which can make attaching GDB/LLDB tricky. To easily step through a test, you can ask nextest to run it sequentially or simply drop down to native Cargo:
```bash
cargo test -p aethervk-core-rlib -- --exact integrate_particles_p1_p2
```
Or, inside `lldb`:
```bash
lldb -- target/debug/deps/aethervk_core_rlib-<hash> integrate_particles_p1_p2 --exact
```

### Linux / Docker (Lavapipe)
When testing under Docker with Lavapipe (software rasterization), you must often provide a virtual X server for Vulkan presentation:
```bash
xvfb-run cargo nextest run test_visual_physics_render_sync
```

## Known VVL Quirks (Linux/Lavapipe)

If you encounter Vulkan Validation Layer errors like `[ UNASSIGNED-CoreValidation-DrawState-InvalidImageLayout ] VK_IMAGE_LAYOUT_UNDEFINED` during test teardown:
1. **Deferred Destruction**: Ensure transient command pools aren't destroyed while their queues are still retiring. AetherVk defers these to `Device::drop`.
2. **Cross-Queue Host Synchronization**: VVL sometimes fails to propagate layout transitions across queues if only `vkWaitForFences` is used (without a `VkSemaphore`). Using `device_wait_idle()` immediately after `wait_for_fences` flushes VVL's tracking state and resolves false positives.
3. Vulkan Validation Layers (VVL), particularly in older or less robust environments like Lavapipe, has a known limitation in its Core Validation tracking: it sometimes fails to track execution dependencies established purely through host synchronization (`vkWaitForFences`).
  If a Command Buffer transitions an image layout, and you synchronize via a CPU wait instead of a device-side Semaphore, VVL loses the layout state. When the presentation engine subsequently tries to draw the image, VVL throws `VK_IMAGE_LAYOUT_UNDEFINED` even
  though the actual device memory is perfectly valid. `device_wait_idle` was acting as a massive global hammer that forced VVL to process all pending states in its tracker, bypassing the bug at the cost of stalling the GPU.

  Soluition: Wait on a timeline semaphore instead

## Docker Environment & CI Testing

AetherVk relies on Docker to replicate the exact GitHub Actions environment locally (especially to test the `lavapipe` software rasterizer and Vulkan Validation Layers on Linux).

### Running Tests in Docker

If you want to manually run the CI environment locally (for `amd64` or `arm64`):

1. **Build the image**:
   ```bash
   # For x86_64
   docker build --platform linux/amd64 -f Dockerfile.test -t aethervk-test .

   # For ARM64
   docker build --platform linux/arm64 -f Dockerfile.test.arm64 -t aethervk-test-arm64 .
   ```

2. **Run the container**:
   ```bash
   # direct run in x86_64
   docker run --platform linux/amd64 --rm -v "$(pwd):/workspace" -w /workspace aethervk-test

   # run with GDB in ARM64 (I'm building from Apple Silicon)
   docker run --platform linux/arm64 --rm \
    -v "$(pwd):/workspace" \
    -v aethervk-arm64-target:/build-target \
    -w /workspace \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    aethervk-test-arm64
   ```

### Important Docker Testing Quirks

If you are writing your own Docker commands or scripts for AetherVk, be aware of two highly specific technical gotchas that will save you hours of debugging:

#### 1. `xvfb-run` hanging silently forever (The PID 1 Issue)
In headless Linux environments, you use `xvfb-run` to spin up a fake X11 display server so Vulkan instance creation succeeds. However, **if `xvfb-run` is invoked as the direct command (PID 1) in a Docker container, it will hang infinitely with zero output.**

**Why?** `xvfb-run` waits for the background `Xvfb` process to send a `SIGUSR1` signal to indicate it's ready. The Linux kernel completely ignores default signals sent to PID 1. The signal gets swallowed, and `xvfb-run` waits forever, preventing `cargo` from even starting.
**The Fix:** Always use `tini` as your Docker entrypoint, or pass `--init` to your `docker run` command. `tini` securely takes over PID 1, passing all signals down properly so `xvfb-run` can safely run as PID 2.

#### 2. `SIGSEGV` during teardown (The GPU-AV Lavapipe bug)
If you are running the test suite on `lavapipe` (the CPU emulation driver used in CI) and see random `(test aborted with signal 11: SIGSEGV)` crashes during teardown/cleanup, this is a known bug inside Khronos's **GPU-Assisted Validation (GPU-AV)** layer interacting poorly with `llvmpipe`. 

**The Fix:** You can selectively disable `GPU-AV` in AetherVk during CI runs by setting the `AETHERVK_DISABLE_GPU_AV=1` environment variable:
```bash
docker run --init -e AETHERVK_DISABLE_GPU_AV=1 --platform linux/amd64 --rm -v "$(pwd):/workspace" -w /workspace aethervk-test
```

### Register Pressure and Lavapipe

If you encounter `SIGSEGV` segmentation faults directly inside shader execution (like in `integrate_bodies`) when testing on `lavapipe` in Docker, it may be due to an LLVM JIT Compiler bug regarding register allocation. While the exact registers vary by CPU architecture (like `x30` on ARM64 or equivalent general-purpose registers on x86_64), the root cause is the same.

**Why it happens:**
`spirv-val` will report that your shader is perfectly valid. However, the driver (`llvmpipe`) compiles that SPIR-V into native machine code assembly using LLVM. In highly complex shaders (like physics integration), the compiler runs out of physical CPU registers—a scenario known as **High Register Pressure**. 

When `local_size_x` (e.g., 32) is heavily mismatched with the hardware's natural SIMD width (e.g., 4 or 8 depending on the CPU's vector instructions), the JIT compiler wraps the shader logic in a CPU loop to simulate the threads. In specific edge cases under high register pressure, the LLVM backend has a known bug where it accidentally re-uses critical registers for the inner SIMD loop counter, destroying the memory pointer it was holding for constant float data. When it tries to read the float data using the loop counter as a memory address, the program instantly segfaults.

**How to fix it:**
Since this is a compiler machine-code generation bug, we cannot fix it from Vulkan. We must alter the generated machine code by "jiggling the handle":
1. **Match the Subgroup Size:** Reduce `local_size_x` to match the hardware execution width (e.g., `4`). This eliminates the internal loop entirely, completely bypassing the register aliasing bug. (Preferably using Vulkan Specialization Constants so as not to penalize native hardware).
2. **Reduce Register Pressure:** Simplify the shader's internal variables and scope lifetimes, giving the compiler more registers to work with.

### Strategy to debug a GPU hang or crash in Lavapipe

If a test times out, hangs infinitely, or segfaults (such as `test_energy_conservation_bounce`), you can use GDB to inspect the Lavapipe worker threads.

#### 1. Run the specific test under GDB inside the container

We will append the GDB execution command to your Docker `run` command. We need to point GDB to the compiled test binary and filter for the specific test.

```bash
docker run -it --platform linux/arm64 --rm \
    -v "$(pwd):/workspace" \
    -v aethervk-arm64-target:/build-target \
    -w /workspace \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    aethervk-test-arm64 \
    bash -c "cargo test --features 'collisions,shader_debug_sync' --package aethervk-core-rlib test_energy_conservation_bounce --no-run && \
             xvfb-run -a rust-gdb --args \$(find /build-target/debug/deps -maxdepth 1 -name 'aethervk_core_rlib-*' -type f -executable -printf '%T@ %p\n' | sort -rn | head -n1 | cut -d' ' -f2) test_energy_conservation_bounce --exact --nocapture"
```

#### Alternative: Using `cargo nextest` (Recommended)

You can completely bypass the manual `find` parsing by using `cargo nextest`'s built-in `--debugger` flag. This will automatically locate the binary, spawn the debugger interactively, and stream all output directly to your terminal.

```bash
docker run -it --platform linux/arm64 --rm \
    -v "$(pwd):/workspace" \
    -v aethervk-arm64-target:/build-target \
    -w /workspace \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    aethervk-test-arm64 \
    xvfb-run -a cargo nextest run --features 'collisions,shader_debug_sync' -p aethervk-core-rlib -E 'test(test_energy_conservation_bounce)' --debugger 'rust-gdb --args'
```

#### 2. Trigger the hang and interrupt

1. Once GDB starts, type `run` and hit Enter.
2. Wait a few seconds for the test to reach the point where it normally hangs.
3. Press `Ctrl+C` (SIGINT). This will pause the test execution and drop you back into the GDB prompt.

#### 3. Inspect the threads (`thread apply all bt`)

Run the following command in GDB to see what every thread is doing:

```gdb
thread apply all bt
```

What to look for:
- **Host Thread**: You'll likely see the main Rust test thread stuck at `vkWaitForFences` or similar, waiting for the compute queue to finish.
- **Lavapipe Threads**: You will see several worker threads (`lp_scene_...` or generic thread pools). If the shader has an infinite loop, you will see threads stuck inside a frame with no symbol name (or an LLVM JIT symbol), which represents the compiled SPIR-V code.

#### 4. Analyze the Shader Disassembly

If you find a lavapipe thread spinning in JIT code:
1. Switch to that thread: `thread <ID>`
2. View the disassembly around the instruction pointer to see the loops:
   ```gdb
   layout asm
   # or
   x/20i $pc - 10
   ```
3. Because LLVM translates the SPIR-V CFG (Control Flow Graph), you can look for backward branches (e.g., `b.ne`, `cbnz` on ARM64) to identify the tight loop. You can cross-reference the loop structure (e.g., an unbounded `while` loop fetching `frame_bda`) with your Rust/GLSL compute shader source.

#### Automated Tracing

Since you can't easily press `Ctrl+C` in an interactive automated CI prompt, you can run an automated version of this command that launches the test in GDB, waits 30 seconds for it to hang, automatically sends a `SIGINT`, and dumps all thread backtraces to a file for analysis.

### Lavapipe Countermeasures

Yes! Because llvmpipe is a JIT software renderer, it has known edge-case bugs handling deep register spilling and stack allocation when compiling complex compute shaders. Based on recent Mesa/LLVM issues, here are the main countermeasures against this:

- **Gate the Prints:** Wrap `debugPrintfEXT` with `if (gl_LocalInvocationIndex == 0)` or similar subgroup limits. This massively reduces the instruction permutations the JIT has to compile for SIMT execution.
- **Limit JIT Optimization:** Set `GALLIUM_DRIVER=llvmpipe` along with debug flags to force LLVM to skip certain aggressive unrolling/vectorization steps that trigger the stack frame miscalculation.
- **Increase Thread Stack:** Since llvmpipe threads execute locally, running `ulimit -s unlimited` (or at least `8192`) before the Docker command can occasionally save the process if it's purely a stack-overflow crash rather than an x30 (arm only) link-register clobber.
- **SSBO Fallback:** When debugging on software renderers, many developers completely abandon `debugPrintfEXT` and manually write debug floats into a pre-allocated Shader Storage Buffer Object (SSBO), reading it back on the CPU instead.

