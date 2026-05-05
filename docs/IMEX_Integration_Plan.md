# IMEX Physics & Rendering Integration Plan

This document outlines the architectural plan for integrating the IMEX (Implicit-Explicit) physics engine with the existing Vulkan rendering backend, covering both CPU and GPU implementations.

## 1. Framerate Governing and Dispatching (`logic_thread.rs`)

The `logic_thread.rs` orchestrates the application loop, strictly coupling the simulation logic and rendering submission:

*   **Fixed Timestep and Frame Pacing:** The logic thread operates in a continuous loop. For each scene, it maintains a `PlayControl` struct containing `target_frame_time` (set to 16ms, or ~60 FPS) and `last_frame_start`. It checks if `now - last_frame_start >= target_frame_time`.
*   **Synchronization (Tick N-1 -> Tick N):** Before initiating `simulation_tick N`, the thread validates the completion of the previous render step. It polls the `task_manager` using `pc.last_render_tick`. If the previous render task is still `Pending`, `can_tick` is evaluated to `false`, effectively stalling the simulation tick until the GPU / render thread finishes `render_tick N-1`.
*   **Dispatching (`execute_simulation_tick`):** Once cleared, `execute_simulation_tick` runs. It queries the `time_state`, calculates the elapsed time based on the active `TimeScale`, and calls the planetary `step()` logic. If the scene is set to parallelize, it dispatches updates using the Thread Pool.
*   **Render Submission:** Finally, the logic thread dispatches a `RenderCommand::RenderFrame` to the rendering thread, passing the newly generated `task_id`, and saves this `task_id` back into `pc.last_render_tick`, guaranteeing strict sequential dependencies between simulation and presentation.

## 2. Kernels Backend Implementations

The `Kernels` trait in `gpu.rs` defines a granular API for an Implicit-Explicit (IMEX) physics solver, mixed with Continuous Collision Detection (CCD).

### A. CPU Single-Threaded Implementation (`CpuScalarKernels`)
*   **Structure:** Implements `Kernels` using standard `Vec<T>` as the underlying `DeviceBuffer`. The `CommandBuffer` is a simple mock or a deferred closure queue.
*   **Mixed CPU-GPU (Kinematics):** In `build_kinematic_bodies`, it queries the SPICE kernels via `almanac.rs` (`get_ephem_full`) to acquire precise planetary positions and velocities.
*   **Execution:** Iterates sequentially over the arrays. BVH construction uses standard recursive splitting.
*   **Particle Uploads:** Uses `upload_particle_systems` to transfer the updated simulation state from CPU memory to the GPU mega-buffers before calling `draw_particle_indirect`.

### B. CPU Multi-Threaded Implementation (`CpuSimdKernels`)
*   **Structure:** Leverages `aethervk_oshal_rlib::os::pool::ThreadPool` to partition tasks into `Workload`s.
*   **Memory Management:** Integrates `AllocatorPool` (from `external/vk-mem/src/pool.rs`) directly into the Logic Context. Threads allocate temporary scratch memory from the pool to avoid global heap contention and ensure NUMA-aware, cache-aligned allocations.
*   **Execution:** Integrators use chunked iterators distributed across the thread pool. BVH and Stream Compaction are parallelized.
*   **Particle Uploads:** Similar to the scalar approach, state is synced via `upload_particle_systems` using `SyncMode::CpuUpload` barriers prior to `draw_particle_indirect`.

### C. Vulkan Compute Implementation (`VulkanComputeKernels`)
*   **Structure:** Implements `Kernels` using Vulkan Compute Pipelines, aligning with the `PhysicsPipelines` scaffold in `vulkan/physics.rs`.
*   **Memory Management:** `DeviceBuffer`s are backed by `vk::Buffer`s using `VK_KHR_buffer_device_address`.
*   **Mixed CPU-GPU Execution:**
    1.  **CPU Host:** Queries `almanac.rs` to extract precise epoch positions for celestial bodies.
    2.  **Upload:** Pushes Kinematic states into the `ForceEmitter` arrays via mapped memory.
    3.  **Compute Dispatch:** Chains the shaders (`p1-2_imex` -> `lbvh_build` -> `ccd` -> `stream_compact` -> `reduce_toi` -> `barnes_hut` -> `p5_imex`), using compute-to-compute `vkCmdPipelineBarrier`s.
    4.  **Rewind:** If `reduce_toi` detects a collision ($t_c < dt$), a DMA read blocks the CPU, rewinds the state, and redispatches the step.
*   **Direct Rendering Integration:** 
    *   Unlike the CPU implementations, the compute shaders *directly* populate the particle and indirect draw mega-buffers used by `draw_particle_indirect`.
    *   The CPU-side `upload_particle_systems` function becomes strictly for CPU-fallback/demonstration logic.
    *   **Synchronization (`SyncMode`):** A `KernelRenderBridge` implementation resolves the `SyncMode` barriers from `gpu_backends/vulkan.rs` (e.g., `SameQueueCompute` or `CrossQueueCompute`). This is executed *before* rendering to transition the buffers from Compute Write to Vertex/Indirect Read.
    *   **Visual Compute:** Passes like `update_sun` or visual-only particle updates execute on the graphics queue natively, perfectly aligned with the `SyncMode` transition logic to avoid redundant queue family ownership transfers.
