# IMEX Physics & Rendering Integration Architecture Plan

## 1. Executive Summary

This document outlines the comprehensive architectural plan for integrating the Implicit-Explicit (IMEX) physics engine with the Vulkan rendering backend in the Aether-Vk engine. A critical challenge addressed in this design is the inherent limitation of `f32` precision on GPUs when simulating celestial-scale environments alongside microscopic particle dynamics. 

To overcome this, the architecture completely abandons a single, global coordinate system. Instead, it introduces a hierarchical spatial grid system termed **"Bubbles" (Reference Frames)**. This ensures that collision mathematics and rendering transformations occur entirely within local `f32` coordinate spaces, preserving precision, stability, and performance. Furthermore, it explicitly handles the extreme scale disparities in mass (Earth vs Dust) and distance (AU vs Meters), while integrating existing Rust API components (`gpu.rs`, `vulkan/device.rs`, `almanac.rs`) and all specific compute shaders (`assets/sim/*.comp`).

## 2. Framerate Governing and Dispatching (`logic_thread.rs`)

The `logic_thread.rs` orchestrates the core application loop, strictly coupling the simulation logic with rendering submission to guarantee determinism and visual consistency. The governing mechanism operates as follows:

### 2.1 Fixed Timestep and Frame Pacing
The logic thread operates in a continuous loop. For each active scene, it maintains a `PlayControl` struct containing `target_frame_time` (typically set to 16ms for ~60 FPS) and `last_frame_start`. The thread actively polls the monotonic clock, evaluating if `now - last_frame_start >= target_frame_time`.

### 2.2 Strict Tick Synchronization (Tick N-1 -> Tick N)
Before initiating `simulation_tick N`, the thread must validate the completion of the previous rendering step. It polls the `task_manager` using the `task_id` stored in `pc.last_render_tick`. If the previous render task is still evaluated as `Pending`, `can_tick` is forced to `false`. This effectively stalls the simulation tick, ensuring the GPU and rendering thread finish `render_tick N-1` before mutating the simulation state for `N`.

### 2.3 Dispatching (`execute_simulation_tick`)
Once the synchronization barrier is cleared, `execute_simulation_tick` is invoked. It queries the `time_state`, calculates the elapsed time based on the active `TimeScale` (e.g., Realtime, OneDay, OneWeek), and executes the planetary `step()` logic. If the scene configuration permits, it leverages the `ThreadPool` to dispatch updates concurrently across celestial bodies.

### 2.4 Render Submission
Finally, the logic thread dispatches a `RenderCommand::RenderFrame` to the rendering thread via the MPSC channel. It passes the newly generated `task_id` for the upcoming frame and saves this `task_id` back into `pc.last_render_tick`. This closes the loop, guaranteeing strict sequential dependencies between simulation and presentation phases.

## 3. Hierarchical Spatial Grids ("Bubbles") Architecture

Building a hierarchical physics engine from scratch in pure `f32` requires completely abandoning absolute global coordinates. The engine will rely on Reference Frames (Bubbles) and Unit Scaling.

Currently, `PhysicsScene::build_from_scene` builds a single world-wide `LinearBVH`. This must be fundamentally refactored to support the Reference Frame paradigm.

### 3.1 The Core Data Structure: Reference Frames
Every physical entity in the game must belong to a `ReferenceFrame`. A Reference Frame defines the local coordinate system origin and the scaling factor.

We define two primary types of frames:
*   **Macro Frame (The Solar System):**
    *   **Origin:** The Sun (or system barycenter).
    *   **Scale:** $1.0 \text{ unit} = 1,000,000 \text{ km}$ (or $1 \text{ AU}$).
    *   **Occupants:** Planets, Moons, Comet Nuclei.
*   **Micro Frame (Planets & Comets):**
    *   **Origin:** The center of mass of the specific celestial body.
    *   **Scale:** $1.0 \text{ unit} = 1 \text{ meter}$.
    *   **Occupants:** Dust particles, landers, the player character, surface debris.

Every object will utilize a unified structural representation:
```rust
struct PhysicsBody {
    local_position: Vec3f32,
    local_velocity: Vec3f32,
    mass: f32, // Mass in kg, but handled specially to avoid f32 precision loss
    parent_frame_id: u32, // ID referencing the spatial grid this body occupies
}
```

### 3.2 Handling Extreme Mass and Distance Disparities
In `f32`, mass can cause critical precision losses. For example, adding Earth's mass ($\approx 5.97 \times 10^{24}$ kg) and a dust particle's mass ($\approx 10^{-6}$ kg) will completely obliterate the dust's contribution due to floating-point truncation.

To resolve this, we leverage the fact that the dust particle's mass cancels out when computing its gravitational acceleration ($a_{dust} = G \cdot M_{earth} / r^2$). Thus, large celestial bodies act purely as `ForceEmitter`s providing a standard gravitational parameter ($\mu = G \cdot M$).

**The `ForceEmitter` structure natively maps to our GPU shaders (e.g., `p5_imex_particles.comp`):**
```glsl
struct ForceEmitter {
    vec3 position;
    float mu; // G * M
};
```

**Scaling $\mu$ between Reference Frames:**
Because $\mu$ has units of $L^3 T^{-2}$, it must be scaled whenever the distance scale factor ($S$) changes.
*   If transitioning from Micro ($1 \text{ unit} = 1 \text{ m}$) to Macro ($1 \text{ unit} = 10^9 \text{ m}$):
    *   $S = 10^{-9}$
    *   $\mu_{macro} = \mu_{micro} \times S^3 = \mu_{micro} \times 10^{-27}$

This ensures that the `float mu` inside the shader perfectly avoids $10^{30}$ ranges, staying within the optimal $1.0 - 1000.0$ range for `f32` ALUs.

### 3.3 Coordinate Transformation Math
Transitioning objects between different scales requires precise scale conversions using the distance scale factor $S$.

When a particle leaves the Comet's Micro frame and enters the Sun's Macro frame, we calculate its new macro position ($P_{macro}$) and macro velocity ($V_{macro}$):

$$P_{macro} = C_{macro} + (P_{micro} \times S)$$
$$V_{macro} = C_{vel\_macro} + (V_{micro} \times S)$$

*(Where $C$ represents the Comet's position/velocity in the Sun's macro frame).*

### 3.4 The "Handoff" Boundary Logic
Every Micro Frame defines a "Sphere of Influence" (SOI) radius. For a 20kg comet, this might be only a few hundred meters. For Earth, it is roughly 920,000 km.

In the physics loop, the engine evaluates the distance of every particle from its local origin:
*   **Escaping an SOI:** If $distance > SOI\_Radius$, the particle has escaped. It is immediately removed from the Comet Frame. The engine executes the transformation math above and inserts the particle into the Sun Frame.
*   **Entering a new SOI:** While a particle is evaluating within the Sun Frame, the engine checks its distance against all planetary/cometary SOIs. If it breaches Earth's SOI, the math is reversed: the particle is removed from the Sun Frame, its macro coordinates are converted back to micro coordinates relative to Earth, and it is added to the Earth Frame.

### 3.5 The Physics Update Loop
To prevent jitter, ensure exact collisions, and manage handoffs smoothly, the engine must update in a strict hierarchical order every frame:

1.  **Update Macro Bodies First:** Query SPICE kernels (`almanac.rs`) and compute planet-planet interactions to move the Planets and the Comet relative to the Sun.
2.  **Update Micro Bodies Second:** Calculate local forces (e.g., solar wind pressure, local body gravity via `ForceEmitter`) and integrate dust particles relative to their parent (Comet or Earth).
3.  **Process Handoffs:** Check if any particles crossed an SOI boundary and execute the coordinate transformations to switch their frames.
4.  **Resolve Collisions:**
    *   Dust in the Comet Frame *only* evaluates collisions against the Comet mesh (utilizing a local BVH).
    *   Dust in the Earth Frame *only* checks for collisions against the Earth mesh.
    *   Dust in the Sun Frame skips local mesh collisions entirely (space is empty).
5.  **Render:** The rendering camera pipeline applies an offset/transformation matrix based on which frame the player/camera currently resides within, maintaining pure `f32` precision down to the fragment shader.

## 4. `Kernels` Backend Implementations & Existing Code Integration

The `Kernels` trait in `aethervk.core/rlib/src/gpu.rs` defines a granular API for the IMEX physics solver mixed with Continuous Collision Detection (CCD). We map this explicitly to the three implementations, integrating the exact existing code components.

### 4.1 CPU Single-Threaded Implementation (`CpuScalarKernels`)
*   **Structure:** Implements `Kernels` using standard `Vec<T>` as the underlying `DeviceBuffer`. 
*   **Kinematics (SPICE Almanac):** `build_kinematic_bodies` queries `almanac.rs` (`get_ephem_full`) to acquire precise Macro positions and velocities, casting planets to `ForceEmitter` arrays.
*   **Execution:** Iterates sequentially for `compute_forces` and `step_ode`. Stream compaction uses standard `.retain()`.
*   **Particle Uploads:** Uses `upload_particle_systems` (from `RenderDevice` in `gpu.rs`) to transfer the updated state to the GPU mega-buffers before calling `draw_particle_indirect`.

### 4.2 CPU Multi-Threaded Implementation (`CpuSimdKernels`)
*   **Structure:** Leverages `aethervk_oshal_rlib::os::pool::ThreadPool` to partition tasks into `Workload`s.
*   **Memory Management (`vk-mem/pool.rs`):** Injects `AllocatorPool` directly into the Logic Context. Threads allocate temporary bump-buffers from the `AllocatorPool` to avoid global heap contention and ensure NUMA-aware, cache-aligned allocations for worker threads.
*   **Execution:** Uses chunked iterators distributed across the thread pool. Stream Compaction implements a CPU-based parallel prefix-sum over the chunks.

### 4.3 Vulkan Compute Implementation (`VulkanComputeKernels`)
This is the primary performance path. It implements `Kernels` using actual Vulkan Compute Pipelines, matching the `PhysicsPipelines` scaffold in `aethervk.core/rlib/src/gpu_backends/vulkan/physics.rs`.

*   **Memory Management:** `DeviceBuffer`s are backed by `vk::Buffer`s utilizing `VK_KHR_buffer_device_address` to pass pointers natively via Push Constants.
*   **Mixed CPU-GPU Execution Flow (Per Reference Frame):**
    1.  **CPU Host Setup:** The CPU queries `almanac.rs` to extract precise epoch positions.
    2.  **Upload to GPU:** Kinematic states are pushed into the `ForceEmitter` arrays natively via Vulkan staging buffers.
    3.  **Command Buffer Recording:** The `Kernels::step_ode` equivalent dispatches the exact shaders found in `assets/sim/*.comp`:
        *   **`p1-2_imex_particles.comp`:** Computes $v_{n+1/2}$ and $q_{mid}$ (Phase 1 & 2).
        *   *(vkCmdPipelineBarrier: Compute Write -> Compute Read)*
        *   **`lbvh_build.comp`:** Constructs the spatial hierarchy using Morton codes.
        *   *(vkCmdPipelineBarrier)*
        *   **`ccd.comp` & `p3-4_imex_rigidbody_imr.comp`:** Finds overlaps; implicitly integrates rigid bodies.
        *   *(vkCmdPipelineBarrier)*
        *   **`stream_compact.comp`:** Parallel prefix sum (using `gl_SubgroupInvocationID`) to compact collision pairs.
        *   *(vkCmdPipelineBarrier)*
        *   **`reduce_toi.comp`:** `atomicMin` workgroup reduction to find the global Time of Impact (TOI).
        *   *(vkCmdPipelineBarrier)*
        *   **`lcp_solver.comp` & `apply_impulses.comp`:** Solves interpenetration constraints via Projected Gauss-Seidel (PGS).
        *   *(vkCmdPipelineBarrier)*
        *   **`barnes_hut.comp`:** Evaluates $O(N \log N)$ self-gravity using the Multipole Acceptance Criterion.
        *   *(vkCmdPipelineBarrier)*
        *   **`p5_imex_particles.comp`:** Executes final drift to $q_{n+1}$ and evaluates forces $F_{n+1}$ utilizing `ForceEmitter`s for the big celestial bodies.
    4.  **Handoffs (SOI Transfers):** A dedicated pass flags escaping particles, utilizing the CPU for matrix transformations before pushing them to the appropriate frame's buffer.
    5.  **CCD Rewind & Sync:** The GPU issues a DMA copy of the `tc` buffer back to the CPU. The CPU blocks on the `WaitHandle` (Fence). If $t_c < dt$, the CPU commands a rewind from the `snapshot` buffer and re-dispatches.

*   **Direct Rendering Integration & `SyncMode` (`gpu_backends/vulkan.rs`):**
    *   The compute shaders *directly* populate the particle (`mega_particle_buffer`) and indirect draw (`mega_indirect_buffer`) mega-buffers utilized by `draw_particle_indirect`. The CPU-side `upload_particle_systems` function becomes strictly a fallback.
    *   **`SyncMode` Resolution:** A `KernelRenderBridge` implementation resolves the `SyncMode` barriers (e.g., `SameQueueCompute` or `CrossQueueCompute`). This is executed *before* the rendering pass to transition the buffers from `COMPUTE_WRITE` to `VERTEX_READ`/`INDIRECT_COMMAND_READ`.
    *   **Visual Compute Synergy:** Visual-only passes (e.g., `update_sun`) execute natively on the graphics queue, perfectly aligning with the `SyncMode` transition logic to prevent redundant queue family ownership transfers.