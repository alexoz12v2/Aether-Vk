# Aether-Vk

**Aether-Vk** is a high-performance simulation and rendering engine purpose-built for **comet dust tail simulations** and dynamic orbital physics. 

It aims to provide physically accurate, GPU-accelerated visualizations of celestial mechanics, focusing on the complex dynamics of comet emissions, particle trajectories under solar radiation pressure, and multi-scale coordinate systems.

## 🚀 Architecture

The project is structured into a modern, cross-platform architecture:

- **Core Engine (`aethervk.core`) - Rust & Vulkan**: The computational workhorse of the application. It leverages Rust for memory-safe performance and Vulkan (via `ash` and `vma`) for highly parallelized GPU compute and graphics pipelines. It handles all macro/micro space coordinate transforms, rigid body physics, and particle simulation logic.
- **Frontend UI (`aethervk.ui-app` & `aethervk.ui-logic`) - C# Avalonia**: A sleek, cross-platform desktop interface built with C# and the Avalonia UI framework using the MVVM pattern. It communicates with the Rust core via a C-ABI, allowing users to interact with the scene, spawn comets, import models, and view the simulation in real-time.

## 🌌 Core Features

- **Comet Dust Tail Simulation**: Real-time compute-shader driven particle simulations modeling dust emission and solar forces.
- **Multi-Scale Physics**: Seamless integration of Macro (Astronomical Units, Earth Masses) and Micro (Kilometers, Kilograms) coordinate frames.
- **Vulkan Rendering**: Low-overhead, high-performance rendering pipelines optimized for massive particle counts and complex 3D meshes.
- **Cross-Platform Desktop UI**: Native, hardware-accelerated experience on Windows, macOS, and Linux.

## 🛠️ Prerequisites

To build and run the project, you will need:
- [Rust Toolchain](https://rustup.rs/) (Cargo)
- [.NET 10.0 SDK](https://dotnet.microsoft.com/)
- [Vulkan SDK](https://vulkan.lunarg.com/) (>= 1.4)

## 🏃 Getting Started

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

## 🔄 Physics Loop Overview

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
