# GPU Physics Pipeline Shaders

This document details the compute shaders implementing the mixed CPU/GPU physics engine based on the Linear Complementarity Problem (LCP) and Implicit Midpoint Rule (IMEX) formulations.

## Overview

The physics pipeline leverages compute shaders for massively parallel calculations (like particles), while falling back to CPU logic for sparse, complex structures (like sparse rigid bodies and kinematic SPICE entities).

### `lbvh_build.comp`
- **Purpose**: Constructs a Linear Bounding Volume Hierarchy (LBVH) from Morton-coded particles.
- **Method**: Bottom-up construction using atomic counters to traverse parents, updating AABBs as nodes are processed.

### `ccd.comp`
- **Purpose**: Continuous Collision Detection.
- **Method**: Uses a stack-based traversal of the LBVH to evaluate potential overlap candidates, executing continuous sphere-sphere or sphere-triangle mathematics.

### `stream_compact.comp`
- **Purpose**: Packs sparse collision pairs into dense arrays.
- **Method**: Subgroup parallel prefix sums (Stream Compaction). Subgroups independently sum active collisions, writing into workgroup shared memory to find global offsets.

### `reduce_toi.comp`
- **Purpose**: Finds the earliest Time of Impact (TOI).
- **Method**: Analyzes all overlapping bounding volumes to extract the exact fractional $t_c \in [0, 1)$ Time of Impact. Employs `subgroupMin` and atomic operations to collapse the entire scene's TOI into a single global scalar, signaling when the IMEX integrator needs to rewind.

### `lcp_solver.comp`
- **Purpose**: Solves multiple simultaneous collisions (LCP).
- **Method**: Applies Projected Gauss-Seidel (PGS). It distinguishes between elastic collisions ($v_{rel} < \text{thresh}$, $e>0$) and resting contacts ($e=0$), assembling an $A$ matrix whose off-diagonal elements map the normal vector correlations between coupled simultaneous contacts.

### `apply_impulses.comp`
- **Purpose**: Maps solved impulses to entity velocity modifications.
- **Method**: Prevents race conditions among particles involved in multiple contacts by employing a lock-free Compare-And-Swap (CAS) loop with `atomicCompSwap` over the AOSOA mapped float arrays (by coercing bits to `uint`).
