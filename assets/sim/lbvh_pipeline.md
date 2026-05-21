# Karras' LBVH Algorithm (2012) Pipeline Flow

This describes the implementation of Tero Karras' famous algorithm for highly parallel BVH construction. Building trees on the GPU is notoriously hard because trees are sequential, but this algorithm makes it completely parallel. It happens in three distinct phases:

## Part 1: Topology Construction (Top-Down, fully parallel)
Notice how `determine_range` and `find_split` don't use pointers to traverse a tree? Instead, they look at the Sorted Morton Codes.
Morton codes map 3D positions into a 1D number. When you sort these numbers, particles that are close in 3D space end up next to each other in the array (the Z-order curve).
By comparing the "common bit prefix" of adjacent Morton codes (`findMSB(key1 ^ key2)`), the shader can mathematically deduce exactly where a node should split its children, allowing every internal node of the tree to be built instantly and in parallel, without waiting for a parent node to finish.

## Part 2: Leaf Setup
Each thread takes one particle, reads its position, velocity, and mass, and initializes a leaf node.
It calculates a swept AABB (`p1 = pos + vel * dt`), which means the bounding box covers where the particle is and where it will be next frame. This is crucial for Continuous Collision Detection (CCD).

## Part 3: Bottom-up Refitting (AABB and Mass/Center of Mass)
Part 1 built the "skeleton" of the tree, but the internal nodes don't know how big their bounding boxes should be yet.
This phase starts at the leaves and walks up to the root.
The Atomic Counter Trick: Two children will try to process their shared parent at the same time. The `atomicAdd(..., 1)` ensures that the first child to arrive simply dies (`break;`). The second child to arrive knows that both children are now ready, so it reads both children's bounding boxes, combines them, calculates the combined Center of Mass (essential for Barnes-Hut gravity), and moves up to the next parent.

## Where it fits in your Pipeline
Because of how this algorithm works, it relies on strict prerequisites. Your pipeline must execute in exactly this order:
1. **Morton Encoding** (`morton_encode.comp`): A shader that looks at particle positions and generates a Morton code for each, storing `uvec2(morton_code, particle_index)`.
2. **Radix Sort** (`radix_sort.comp`): A highly optimized GPU sort (usually across multiple dispatches) that sorts that array by the Morton codes. (If you skip this, the LBVH build will produce complete garbage).
3. **LBVH Build** (`lbvh_build.comp`)
4. **LBVH Collapse** (`lbvh_collapse.comp`): Reads the binary tree this shader just built and collapses it into your wide N-ary MultiBvhBuffer for faster traversal.
5. **Broad Phase / CCD / Barnes-Hut**: Your simulation passes that traverse the BVH.