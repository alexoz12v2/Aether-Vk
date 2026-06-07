# SOI Interaction and Top-Level BVH Architecture

## 1. Introduction
In a hierarchical "Bubbles" spatial engine, the universe is composed of multiple Reference Frames (Macro and Micro). Each frame maintains its own local coordinate system in `f32`. This document details how these frames interact, transmit forces, process collisions, and efficiently transition particles between them using a Top-Level Acceleration Structure (TLAS).

## 2. Force Transmission & Gravity
Gravity is a long-range force, meaning a particle inside a Comet's Micro frame still feels the Sun's gravity.
*   **Force Emitters:** Large celestial bodies (Sun, Planets, Comets) are represented as `ForceEmitter`s, containing a position and a gravitational parameter (mu = G * M).
*   **Local Evaluation:** When integrating a particle inside a Micro frame, the physics kernels evaluate:
    1.  The local body's gravity (origin is `0,0,0`).
    2.  The Macro body's (Sun) gravity. The Sun's position is transformed into the Micro frame's local space to compute the vector.
*   By representing massive bodies merely as point-mass force emitters, we avoid executing costly mesh-BVH collision checks for long-range gravity.

## 3. Collision Isolation
Collisions are strictly evaluated within the same Reference Frame to maintain `f32` precision.
*   **Per-Frame TLAS:** `PhysicsScene` builds a unique Bounding Volume Hierarchy (TLAS) for every active Reference Frame. This TLAS bounds all `PhysicalMeshComponent`s that belong to that specific frame.
*   **Raycasting and Intersections:** A particle in the Earth Micro frame *only* traverses the Earth TLAS. It does not check against the Moon or the Sun. This guarantees collision math stays close to the origin, eliminating floating-point jitter.

## 4. The SOI Handoff & The SOI TLAS
Particles must smoothly transition between Micro (Planet) and Macro (Sun) frames.
*   **Micro to Macro (Escape):** A particle calculates its distance from the origin `(0,0,0)`. If this exceeds the Micro frame's `soi_radius`, it is removed, its position/velocity are multiplied by the scale factor $S$ and offset by the planet's Macro position, and it is inserted into the Macro frame.
*   **Macro to Micro (Capture) via SOI TLAS:** In the Macro frame, a particle could potentially hit any planet's SOI. To avoid checking every particle against every planet in an O(N * M) loop, `PhysicsScene` builds an **SOI TLAS**. 
    *   This is a special BVH existing in the Macro frame.
    *   Its leaves are the Spheres of Influence (represented as AABBs or bounding spheres) of all active Micro frames.
    *   During the Macro frame update, particles query this SOI TLAS. If an intersection is found, the particle is transformed using 1/S into the respective Micro frame.

## 5. Integration with Kernels
In `cpu_kernels.rs` and the Vulkan compute pipeline:
*   The execution is batched by frame. `step_ode` processes the Macro frame, followed by all Micro frames.
*   Data arrays (`DynamicBody`, `KinematicBody`) are filtered or offset by `parent_frame_id`.
*   After the integration step, a dedicated `process_handoffs` kernel/function evaluates the SOI TLAS, flags transitioning bodies, applies the coordinate transformation, and re-inserts them into the appropriate buffers for the next tick.