# Aether-Vk

**Aether-Vk** is a high-performance simulation and rendering engine purpose-built for **comet dust tail simulations** and dynamic orbital physics. 

It aims to provide physically accurate, GPU-accelerated visualizations of celestial mechanics, focusing on the complex dynamics of comet emissions, particle trajectories under solar radiation pressure, and multi-scale coordinate systems.

## Architecture

The project is structured into a modern, cross-platform architecture:

- **Core Engine (`aethervk.core`) - Rust & Vulkan**: The computational workhorse of the application. It leverages Rust for memory-safe performance and Vulkan (via `ash` and `vma`) for highly parallelized GPU compute and graphics pipelines. It handles all macro/micro space coordinate transforms, rigid body physics, and particle simulation logic.
- **Frontend UI (`aethervk.ui-app` & `aethervk.ui-logic`) - C# Avalonia**: A sleek, cross-platform desktop interface built with C# and the Avalonia UI framework using the MVVM pattern. It communicates with the Rust core via a C-ABI, allowing users to interact with the scene, spawn comets, import models, and view the simulation in real-time.

## Core Features

- **Comet Dust Tail Simulation**: Real-time compute-shader driven particle simulations modeling dust emission and solar forces.
- **Multi-Scale Physics**: Seamless integration of Macro (Astronomical Units, Earth Masses) and Micro (Kilometers, Kilograms) coordinate frames.
- **Vulkan Rendering**: Low-overhead, high-performance rendering pipelines optimized for massive particle counts and complex 3D meshes.
- **Cross-Platform Desktop UI**: Native, hardware-accelerated experience on Windows, macOS, and Linux.

## Prerequisites

To build and run the project, you will need:
- [Rust Toolchain](https://rustup.rs/) (Cargo)
- [.NET 10.0 SDK](https://dotnet.microsoft.com/)
- [Vulkan SDK](https://vulkan.lunarg.com/) (>= 1.4)

## Getting Started

1. **Build the Rust Core (Library)**
   ```bash
   cd aethervk.core
   cargo build --release
   ```

2. **Run the Avalonia UI**
   ```bash
   cd ../aethervk.ui-app
   dotnet run
   ```

*(Ensure your environment is configured so the C# runtime can locate the compiled Rust `cdylib` native library).*

### Note on Rust compilation for `aarch64-pc-windows-msvc`

Our `test_utils` crate, which is not a necessity for a cdylib build, depends on the [`ring`](https://github.com/briansmith/ring) crate (brought by `reqwest` crate), which makes usage of some inline assembly.

In a windows build for ARMv8 64 bit, if `ring`'s build script detects MSVC/clang-cl, [it will swap for LLVM's clang](https://docs.rs/crate/ring/0.17.8/source/build.rs#572), which would invalidate MSVC-compatible injected flags,
such as those inserted by `cargo-xwin` during a cross-compilation.

In order to fix this, we'd have to manually translate MSVC-style flags to LLVM-style ones. In particular, the error encountered in our build is related to system headers specification, which can be patched with the following wrapper script

```bash
# 1. Find the real clang binary, bypassing this wrapper script
# Note: we assume we inserted this script in a directory called ".cargo-xwin-wrapper"
REAL_CLANG=$(which -a clang | grep -v "\.cargo-xwin-wrapper" | head -n 1)

if [ -z "$REAL_CLANG" ]; then
    echo "clang-wrapper: error: could not find real clang in PATH" >&2
    exit 1
fi

# 2. Translate MSVC-style /imsvc flags to GCC/Unix-style -isystem
args=()
for arg in "$@"; do
    if [ "$arg" = "/imsvc" ]; then
        args+=("-isystem")
    else
        args+=("$arg")
    fi
done

# 3. Execute the real clang with the translated arguments
exec "$REAL_CLANG" "${args[@]}"
```

> **Note on CI environments:** This workaround is only required for local cross-compilation from Unix-like hosts (such as macOS or Linux) using tools like `cargo-xwin`. It is **not** an issue for our GitHub Actions CI pipeline. Because our CI builds the `aarch64-pc-windows-msvc` target natively on a `windows-latest` runner, it uses the standard Windows MSVC toolchain and bypasses `cargo-xwin` entirely.

## Physics Loop Overview

The core simulation loop runs natively on the GPU via Vulkan Compute, orchestrated by the Rust backend (`gpu_backends.rs` and `vulkan/physics.rs`). The pipeline employs an **IMEX (Implicit-Explicit)** integration scheme combined with robust **Continuous Collision Detection (CCD)**. 

A standard simulation tick involves:

1. **Scene Preparation & TLAS**: The CPU constructs and uploads a Top-Level Acceleration Structure (TLAS) mapping out Macro and Micro coordinate frames. Entity components (Rigid Bodies, Particles, Kinematics) are synchronized to GPU buffers.
2. **Particle Emission**: New particles (e.g., comet dust) are emitted on the GPU based on the Sun's position and configured emitters.
3. **IMEX Integration (Predictor)**: 
   - *Particles*: Undergo a Velocity Verlet predictor step (half-kick + full drift).
   - *Rigid Bodies*: Forces are accumulated and integrated using an Implicit Midpoint Rule (IMR) solve with Picard gyro-stabilization.
4. **Gravity & Forces**: 
   - A BVH is built on the GPU to compute particle self-gravity. 
   - Macro-frame emitters (like planets or the sun) apply forces to micro-frame particles via inline coordinate transformations.
5. **Collision Suite (Broad & Narrow Phase)**: *[When collisions are enabled]*
   - **Broad-Phase**: Executes heavily parallelized classification kernels (`bp_classify`, `bp_cross_lca`, `bp_particle_self`) separating pairs into RigidBody-RigidBody, RigidBody-Particle, and Cross-Frame collisions.
   - **Narrow-Phase (CCD)**: Computes the earliest Time of Impact ($t_c$). 
   - **Resolution**: The engine rewinds dynamics to $t_c$, applies elastic/inelastic responses, and re-integrates the remainder of the timestep.
6. **IMEX Integration (Corrector)**: Completes the timestep with a final velocity corrector step.
7. **Write-Back**: Final state is downloaded asynchronously to update the CPU's ECS scene.

## 📄 License

See the `LICENSE` file for details.

