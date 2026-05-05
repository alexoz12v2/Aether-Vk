# Architectural Plan for Kernels Trait (Vulkan Compute)

This document details the comprehensive plan to implement the `Kernels` trait for the physics and collision simulation, leveraging a high-performance Vulkan compute backend. 

## 1. Foundation: Memory & Data Structures

The foundation requires transitioning to a highly efficient compute memory model leveraging modern Vulkan features. Note that while the target is **Vulkan 1.1**, the architecture relies on specific extensions that provide modern capabilities: `VK_KHR_buffer_device_address`, `VK_KHR_vulkan_memory_model`, and `VK_KHR_spirv_1_4`.

*   **Buffer Reference & Bindless Architecture:** Implement the `DeviceBuffer`, `DeviceList`, and `DeviceBvh` traits using `vk::Buffer` allocations backed by `vk_mem`. Rely on `GL_EXT_buffer_reference` to pass 64-bit device memory addresses directly via Push Constants. This completely bypasses descriptor set updates for compute dispatches, significantly reducing CPU overhead.
*   **Data Layouts:** 
    *   **AOSOA (Array of Structures of Arrays) for Particles/Dynamics:** Map large homogeneous arrays (like `DynamicBody` and particle states) to AOSOA layouts in GLSL. Grouping data into blocks sized to the hardware warp (e.g., `SUBGROUP_SIZE = 32`) ensures perfectly coalesced global memory reads/writes.
    *   **AOS (Array of Structures) for Global Data:** Small, globally read arrays (such as force emitters) should remain AOS. Since all threads in a subgroup read the same emitter data simultaneously, this naturally benefits from scalar caches / broadcast mechanisms on the GPU.
*   **Asynchronous Readbacks:** Implement `DeviceBuffer::ReadHandle` as a synchronized wrapper around a dedicated host-visible staging buffer, utilizing a Vulkan Timeline Semaphore to safely await the exact `submit_value` where the DMA transfer completes.

## 2. Force Accumulation & Force Emitters

To evaluate position-dependent forces (such as the gravitational pull of the Sun and planets), the system needs to accumulate influences from multiple "force emitters".

### 2.1 Emitter Data Format
Force emitters (e.g., massive celestial bodies) should be uploaded as an array of structs (AOS) in a single Storage Buffer (SSBO) via `GL_EXT_buffer_reference`. 
*   **Why AOS?** The number of emitters is typically small (e.g., < 100). During force computation, every thread representing a particle will iterate over this same list. Because all threads in a warp access the exact same emitter memory address simultaneously, the GPU's scalar cache handles this extremely efficiently. 

### 2.2 Particle Force Buffering Strategy
The particles' accumulated forces must be stored because the Velocity Verlet algorithm requires the force at both the start and the end of the time step.
*   **Do we need Front/Back Buffering?** No. We only need a **single** buffer for the accumulated forces. 
*   **The Lifecycle:** 
    1.  At the very start of the simulation (Step 0), an initialization kernel computes $F_0 = F(q_0)$ and stores it in the `force_buffer`.
    2.  **Phase 1 (Half-Kick):** Reads $F_n$ from `force_buffer` to compute $v_{n+1/2} = v_n + \frac{h}{2} M^{-1} F_n$.
    3.  **Phase 5 (Force Eval & Final Kick):** After positions have advanced to $q_{n+1}$, a kernel computes the new force $F_{n+1} = F(q_{n+1})$, **overwrites** the value in `force_buffer`, and uses it immediately to compute $v_{n+1} = v_{n+1/2} + \frac{h}{2} M^{-1} F_{n+1}$.
    4.  The overwritten `force_buffer` now holds $F_{n+1}$, perfectly prepped for Phase 1 of the *next* simulation frame.
*   **Force Buffer Data Format:** Since each particle thread reads and writes its own specific accumulated force, this buffer **must** use the **AOSOA** format (based on the subgroup size specialization constant). This guarantees coalesced memory access when reading/writing the force vectors.

## 3. The IMEX Simulation Loop 

The physics loop couples the unconstrained particles with the rigid body nucleus. Since dust grains exert zero macroscopic back-reaction on the nucleus, the particle integration can be explicit, while the rigid body remains implicit.

### Phase 1 & 2: Explicit Particle Half-Kick & Drift
*   **Compute Shader (`particles_explicit_kick_drift.comp`):** 
*   **Action:** 
    *   Read the existing forces $F_n$ from the AOSOA `force_buffer`.
    *   Update particle velocities to $t_{n+1/2}$: $v_{n+1/2} = v_n + \frac{h}{2} M_p^{-1} F_n$.
    *   Drift positions to the temporal midpoint: $q_{mid} = q_n + \frac{h}{2} v_{n+1/2}$.
*   **Output:** The new $\mathbf{q}_{mid}$ is explicitly known and locked for the remainder of the implicit step.

### Phase 3 & 4: Implicit Rigid Body Solve (Lie Group Integration)
The nucleus rigid body operates on a non-Euclidean manifold ($\mathbb{R}^3 \times \text{SO}(3)$) and requires an unconditionally stable implicit solver to perfectly preserve orbital energy over long simulated durations.
*   **Compute Shader (`rigidbody_implicit_solve.comp`):** Dispatch a specialized, single-workgroup compute kernel. 
*   **Newton-Raphson Solver:** Because the particles don't back-react, the massive $(3N+6) \times (3N+6)$ Jacobian trivially collapses. We only need to invert a dense $6 \times 6$ matrix natively within the workgroup's shared memory.
*   **Action:** Iteratively evaluate the Implicit Midpoint Rule (IMR) residual. Once roots $\mathbf{v}_{mid}^*$ and $\boldsymbol{\omega}_{mid}^*$ converge, compute the final $t_{n+1}$ state via the Lie Group exponential map (Rodrigues' formula).

### Phase 5: Finalize Velocity Verlet & Next Force Eval
*   **Compute Shader (`particles_explicit_final_kick.comp`):** 
*   **Action:** 
    *   Advance particle positions from $t_{n+1/2}$ to $t_{n+1}$: $q_{n+1} = q_{mid} + \frac{h}{2} v_{n+1/2}$.
    *   **Force Evaluation:** Iterate over the AOS list of "force emitters" using the new positions $q_{n+1}$ to calculate the new accumulated forces $F_{n+1}$.
    *   Store $F_{n+1}$ back into the AOSOA `force_buffer`.
    *   Finalize velocities: $v_{n+1} = v_{n+1/2} + \frac{h}{2} M_p^{-1} F_{n+1}$.

## 4. Collision Pipeline (Subgroup Optimized)

The collision pipeline will natively use the `BVHNodeBlockAABB` definitions for collaborative cache fetching.

*   **BVH Construction (`Kernels::build_motion_bvh`):** Implement a GPU-accelerated parallel Radix-tree builder over the dynamic bodies' swept AABBs.
*   **Intersection (`Kernels::self_intersect_scene` & `intersect_instances`):** Execute a compute shader that traverses the BVH. Subgroups collaboratively load `BVHNodeBlockAABB` blocks into `shared` memory and test intersections. Pairs hitting the leaf nodes use atomic appends to populate the `CollisionPair` `DeviceList`.
*   **Stream Compaction (`Kernels::compact_collisions`):** Run a kernel leveraging `GL_KHR_shader_subgroup_arithmetic` to perform parallel prefix sums. This efficiently shrinks the collision list down to only the valid, unique impacts.
*   **Earliest TOI (`Kernels::find_earliest_collision`):** Execute a parallel reduction shader (`subgroupMin`) to extract the absolute lowest Time of Impact ($t_c$) into a 1-element buffer, enqueueing its readback to the CPU to decide if the simulation loop must rewind.

## 5. Synchronization & Rendering Bridge

*   **Module Setup:** Create a new `aethervk.core/rlib/src/gpu_backends/vulkan/physics.rs` to encapsulate the `Kernels` trait implementation on the `vulkan::LogicalDevice`.
*   **Trait Integration (`KernelRenderBridge`):** Implement `sync_compute_to_graphics` utilizing `vkCmdPipelineBarrier2`. This is critical for synchronizing the mega-buffers; it ensures that the physics compute queue fully flushes its `STORAGE_WRITE` operations before the graphics queue begins its `VERTEX_READ` phase for the particle archetype rendering.