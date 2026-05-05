# `lbvh_build.comp`

## Purpose
Constructs a Linear Bounding Volume Hierarchy (LBVH) from a set of Morton-coded particles.

## Mathematical Foundation
Efficient spatial partitioning is required to reduce the $O(n^2)$ complexity of pairwise collision checks down to $O(n \log n)$ or better (Chapter 2.4). The LBVH achieves this by interpreting the 3D position of particles as a 1D sequence using Z-order (Morton) curves.

## Implementation Details
- **Topology Construction**: Uses atomic counters and bitwise prefixes to identify the split points in the sorted Morton array, creating internal nodes that connect adjacent sorted primitives.
- **Bottom-Up Refitting**: Once the topology is built, the algorithm processes the tree from the leaves up to the root. It computes the union of the child AABBs using `atomicAdd` on counters to ensure that a parent is only processed once both of its children have completed their AABB evaluations.
