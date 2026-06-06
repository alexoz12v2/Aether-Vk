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

#### 2. Disabling GPU-AV

You can selectively disable `GPU-AV` in AetherVk during CI runs by setting the `AETHERVK_DISABLE_GPU_AV=1` environment variable:

```bash
docker run --init -e AETHERVK_DISABLE_GPU_AV=1 --platform linux/amd64 --rm -v "$(pwd):/workspace" -w /workspace aethervk-test
```
