# `ccd.comp`

## Purpose
Performs Continuous Collision Detection (CCD) between moving particles.

## Mathematical Foundation
In highly dynamic particle systems, objects can pass entirely through each other within a single time step $\Delta t$ if checked discretely (Chapter 1.4.3 & 1.4.5). CCD evaluates the continuous trajectory of the particles. Moving spheres are mathematically modeled as swept volumes (capsules), and roots for intersection times $t_c$ are solved using the quadratic form of the distance equations.

## Implementation Details
- **LBVH Traversal**: Uses a stack-based while loop to iteratively traverse the Linear Bounding Volume Hierarchy (LBVH).
- **AABB Culling**: Extends standard AABB checks to form bounding volumes encompassing the entire continuous motion from $t_n$ to $t_{n+1}$.
- **Output**: Writes candidate collisions (particle ID pairs) into a sparse output list for further stream compaction, preventing symmetric duplicates by enforcing `idA < idB`.
