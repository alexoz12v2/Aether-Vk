# `reduce_toi.comp`

## Purpose
Calculates the earliest fractional Time of Impact (TOI) across all continuous collision candidate pairs in the scene.

## Mathematical Foundation
When simulating continuous collision detection, identifying the exact moment of the first impact is necessary to rewind the simulation safely to a non-penetrating state before applying collision impulses (Chapter 1.4.3). For moving spheres, this is obtained by finding the smallest positive root of the quadratic distance function modeling the swept capsule of the particles.

## Implementation Details
- **Cooperative Reductions**: Instead of using heavy atomic locks for every collision candidate, the shader employs `subgroupMin()` to find the minimum TOI within a subgroup. 
- **Workgroup and Global Reductions**: Subgroup results are stored in a statically sized shared memory array (`[256 / SUBGROUP_SIZE]`), reduced across the workgroup, and finally collapsed into a global minimum using `atomicMin`.
- **Bitwise Floats**: Because `atomicMin` for floats is an extension, the shader represents float data bitwise as `uint`s, safely executing the minimum reduction without losing precision.
