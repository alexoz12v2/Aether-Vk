# Particle Simulation Enhancements: Emission, Clustering, and Trajectories

This document outlines the architectural updates required to enhance the particle system in `AetherVk`, specifically focusing on texture-based emission, optimizing spatial queries for N-body gravity, and extracting fluid Bezier trajectories from chaotic particle clouds.

## 1. Texture-Based Emission Logic

Currently, particle emission samples a mathematical `Distribution2D`. You want to transition to a painterly, artist-driven workflow where a host/device visible RGBA texture dictates both *where* particles spawn (Alpha = probability) and *what* they look like (RGB = color).

### Architectural Plan

1.  **The Probability Density Function (PDF):**
    *   When the texture is loaded (or painted via UI), construct a `Distribution2D` directly from the Alpha channel. The `Distribution2D` uses a 1D marginal/conditional CDF table that allows for perfectly uniform $\mathcal{O}(1)$ importance sampling.
    *   This ensures that dense regions (Alpha = 255) spawn many particles, while empty regions (Alpha = 0) spawn none, without wasting CPU cycles looping or rejecting random samples.

2.  **Emission Loop Update (`scene/particles.rs`):**
    *   The `emit_particles` function remains largely the same, drawing `(uv_x, uv_y)` from the `Distribution2D`.
    *   **New Step:** Sample the texture's RGB channels at the generated `(uv_x, uv_y)` coordinates using bilinear interpolation.
    *   Assign this exact RGB value, combined with the initial alpha (or a lifetime fade), to the `ParticleData`'s `color` attribute (which needs to be added).

3.  **Memory Management:**
    *   The texture should ideally be allocated as `HOST_VISIBLE | HOST_COHERENT` (or in a host staging buffer that syncs to device VRAM) so that the UI can "paint" on it, and the CPU/GPU can read it synchronously.

## 2. BVH vs. Spatial Hashing for Gravity

You asked if rebuilding the BVH every tick is necessary since particles move, and whether Spatial Hashing is better suited for clustering particles to compute mutual/external gravitational forces.

### The Problem with $N$-Body Gravity
Computing gravity naively is an $\mathcal{O}(N^2)$ operation. The Barnes-Hut algorithm reduces this to $\mathcal{O}(N \log N)$ by grouping distant particles into a single "macro-particle" located at their Center of Mass (CoM).

### Spatial Hashing vs. Hierarchical Trees (BVH / Octree)
*   **Spatial Hashing:** Excels at fixed-radius queries (e.g., SPH fluids, identical-size rigid body collisions). It is a *flat* structure. Because it lacks a hierarchy, you cannot easily group "distant clusters" into a single macro-node. You can only query neighbors within a specific grid cell. Therefore, Spatial Hashing is **bad** for gravity.
*   **BVH (or Octree):** A hierarchical structure. A top-level node inherently encapsulates the bounding box, total mass, and CoM of all its children. This is the **exact** structure required for Barnes-Hut gravity.

### Rebuilding vs. Updating
*   Since the dust tail is an expanding, highly dynamic cloud (rather than slightly vibrating rigid bodies), **updating** an existing BVH quickly degrades its quality, leading to massive overlap and terrible query performance.
*   **Yes, you must rebuild the tree every tick.**
*   **The Bottleneck:** The current `BVHBuilder` in `bvh_builder.rs` is a top-down SAH (Surface Area Heuristic) builder. SAH is extremely slow and designed for static geometry.
*   **The Solution (Linear BVH / LBVH):** For dynamic particles, you should implement a **Radix-Tree based LBVH**.
    1.  Compute a Morton Code (Z-order curve) for each particle's 3D position.
    2.  Radix-sort the particles based on their Morton Codes.
    3.  Build the hierarchy bottom-up from the sorted array.
    This process is incredibly fast, inherently parallelizable, and perfectly suited for GPU compute shaders (which you will need for the `Kernels` trait).

## 3. Tracing Trajectories via Rational Cubic Bezier Curves

The goal is to draw a smooth, continuous path representing the "main flow" of the dust tail over time. Dust particles emitted at roughly the same time form a "stratum" or "generation".

### Phase A: Temporal Clustering and Outlier Rejection
Every $N$ milliseconds of simulation time:
1.  **Group by Age:** Bucket all active particles into temporal clusters based on their `age` (e.g., Bucket 1: 0-10ms, Bucket 2: 10-20ms).
2.  **Compute Initial Statistics:** For each bucket, compute the mean position ($\mu$) and standard deviation ($\sigma$).
    *   *Note: Use the `compute_com_and_tensor` logic in `math.rs`.*
3.  **Outlier Rejection (Mahalanobis/Z-Score):**
    *   Iterate through the bucket. If a particle's position is further than, say, $2\sigma$ from $\mu$, flag it as an outlier.
    *   This successfully filters out particles that were emitted with extreme, randomized initial velocities that diverge from the main tail flow.
4.  **Refined CoM:** Recompute the final Center of Mass ($\text{CoM}_i$) of the bucket using only the *inliers*.

### Phase B: Curve Generation
You now have a time-ordered sequence of pristine, robust centroids: $P_0, P_1, P_2, \dots, P_k$.

To fit Rational Cubic Bezier curves to these points:
1.  **Catmull-Rom to Bezier Conversion:** The easiest way to generate a smooth curve passing *exactly* through these centroids is to treat them as control points of a Centripetal Catmull-Rom spline, and convert those segments into Cubic Bezier curves.
2.  For a segment between $P_i$ and $P_{i+1}$, the four Bezier control points ($B_0, B_1, B_2, B_3$) are:
    *   $B_0 = P_i$
    *   $B_1 = P_i + \frac{P_{i+1} - P_{i-1}}{6}$
    *   $B_2 = P_{i+1} - \frac{P_{i+2} - P_i}{6}$
    *   $B_3 = P_{i+1}$
3.  *(Handle boundaries $P_0$ and $P_k$ by duplicating endpoints or extrapolating velocities).*
4.  **Rationality (Weights):** If you want *Rational* Cubic Beziers (NURBS), assign a scalar weight $w$ to each control point. The weight could represent the total mass or density of the bucket, causing the curve to visually "bulge" or have a stronger gravitational presence where the tail is thickest.