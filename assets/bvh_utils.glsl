// ============================================================================
// BVH and Collision Utilities (Vulkan SPIR-V)
// ============================================================================
// Makes heavy use of Vulkan 1.2+ features:
// - Physical Storage Buffers (buffer_reference) for pointer-chasing in BVHs
// - Vulkan Memory Model for correct atomics
// - Subgroup operations for parallel traversal and work distribution
//
// Usage: #include "bvh_utils.glsl" in your compute shaders.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_memory_scope_semantics : require

// ----------------------------------------------------------------------------
// Data Structures
// ----------------------------------------------------------------------------

struct AABB {
    vec3 minBounds;
    vec3 maxBounds;
};

// Variation 1: Standard Array of Structures (AOS)
// Represents `LinearBVHNode<f32>` specialized for AABBs.
struct BVHNodeAABB {
    AABB bound;
    uint left_child_or_primitive_offset;
    uint right_child_offset;
    uint primitive_count;
    uint node_type; // e.g. 0 for AABB, 1 for OBB. Padding in strictly AABB case.
    uint parent_idx;
    float mass;
    vec3 center_of_mass;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer BVHArray {
    BVHNodeAABB nodes[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) writeonly buffer CollisionPairList {
    uint count;
    uvec2 pairs[]; // Pair of primitive/entity indices
};

// ----------------------------------------------------------------------------
// Variation 2: Array of Structures of Arrays (AOSOA)
// Optimized for Cooperative Subgroup operations (CUDA Warp coalescing logic)
// ----------------------------------------------------------------------------
// Injected by Rust to perfectly match the physical GPU's gl_SubgroupSize
layout(constant_id = 0) const uint SG_SIZE = 32;

// SOA block representing `SG_SIZE` nodes. 
// When a warp accesses `block.minX[gl_SubgroupInvocationID]`, memory accesses are perfectly coalesced.
// We can use this struct for shared memory, but NOT for SSBO due to GLSL spec limitation.
struct BVHNodeBlockAABB {
    float minX[SG_SIZE];
    float minY[SG_SIZE];
    float minZ[SG_SIZE];
    float maxX[SG_SIZE];
    float maxY[SG_SIZE];
    float maxZ[SG_SIZE];
    uint left_child_or_primitive_offset[SG_SIZE];
    uint right_child_offset[SG_SIZE];
    uint primitive_count[SG_SIZE];
    uint node_type[SG_SIZE];
};

// Flattened 1D array to bypass GLSL limitation on SSBO array sizing with specialization constants.
// 10 attributes per block: 6 floats, 4 uints. We use uint[] and floatBitsToUint/uintBitsToFloat.
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer BVHBlockArray {
    uint data[];
};

// ----------------------------------------------------------------------------
// Shared Memory Caching (Workgroup Local Memory)
// ----------------------------------------------------------------------------
// Defines a shared memory cache that subgroups can collaboratively populate.
// This is exceptionally useful when many threads (e.g. many rays or many particles) 
// are traversing the same top-level BVH nodes repeatedly.

// Macro to declare a shared memory cache.
#define DECLARE_SHARED_BVH_CACHE(name) shared BVHNodeBlockAABB name

// Macro to collaboratively load a block into a globally declared shared cache.
// Using a macro prevents deep-copy register spillage associated with `inout` params in GLSL.
#define CACHE_BLOCK_COLLABORATIVE(GLOBAL_BVH, BLOCK_IDX, SHARED_CACHE) \
    do { \
        uint _tid = gl_SubgroupInvocationID; \
        if (_tid < SG_SIZE) { \
            uint _stride = 10 * SG_SIZE; \
            uint _base = BLOCK_IDX * _stride + _tid; \
            SHARED_CACHE.minX[_tid] = uintBitsToFloat(GLOBAL_BVH.data[_base + 0 * SG_SIZE]); \
            SHARED_CACHE.minY[_tid] = uintBitsToFloat(GLOBAL_BVH.data[_base + 1 * SG_SIZE]); \
            SHARED_CACHE.minZ[_tid] = uintBitsToFloat(GLOBAL_BVH.data[_base + 2 * SG_SIZE]); \
            SHARED_CACHE.maxX[_tid] = uintBitsToFloat(GLOBAL_BVH.data[_base + 3 * SG_SIZE]); \
            SHARED_CACHE.maxY[_tid] = uintBitsToFloat(GLOBAL_BVH.data[_base + 4 * SG_SIZE]); \
            SHARED_CACHE.maxZ[_tid] = uintBitsToFloat(GLOBAL_BVH.data[_base + 5 * SG_SIZE]); \
            SHARED_CACHE.left_child_or_primitive_offset[_tid] = GLOBAL_BVH.data[_base + 6 * SG_SIZE]; \
            SHARED_CACHE.right_child_offset[_tid] = GLOBAL_BVH.data[_base + 7 * SG_SIZE]; \
            SHARED_CACHE.primitive_count[_tid] = GLOBAL_BVH.data[_base + 8 * SG_SIZE]; \
            SHARED_CACHE.node_type[_tid] = GLOBAL_BVH.data[_base + 9 * SG_SIZE]; \
        } \
    } while(false)

// Macro to fetch an AABB node out of the shared cache into a local `BVHNodeAABB` variable.
#define FETCH_NODE_FROM_CACHE(SHARED_CACHE, LOCAL_IDX, OUT_NODE) \
    do { \
        OUT_NODE.bound.minBounds = vec3(SHARED_CACHE.minX[LOCAL_IDX], SHARED_CACHE.minY[LOCAL_IDX], SHARED_CACHE.minZ[LOCAL_IDX]); \
        OUT_NODE.bound.maxBounds = vec3(SHARED_CACHE.maxX[LOCAL_IDX], SHARED_CACHE.maxY[LOCAL_IDX], SHARED_CACHE.maxZ[LOCAL_IDX]); \
        OUT_NODE.left_child_or_primitive_offset = SHARED_CACHE.left_child_or_primitive_offset[LOCAL_IDX]; \
        OUT_NODE.right_child_offset = SHARED_CACHE.right_child_offset[LOCAL_IDX]; \
        OUT_NODE.primitive_count = SHARED_CACHE.primitive_count[LOCAL_IDX]; \
        OUT_NODE.node_type = SHARED_CACHE.node_type[LOCAL_IDX]; \
    } while(false)

// ----------------------------------------------------------------------------
// Intersection Math
// ----------------------------------------------------------------------------

// AABB vs AABB Intersection
bool intersectAABB(AABB a, AABB b) {
    bool overlapX = a.maxBounds.x >= b.minBounds.x && a.minBounds.x <= b.maxBounds.x;
    bool overlapY = a.maxBounds.y >= b.minBounds.y && a.minBounds.y <= b.maxBounds.y;
    bool overlapZ = a.maxBounds.z >= b.minBounds.z && a.minBounds.z <= b.maxBounds.z;
    return overlapX && overlapY && overlapZ;
}

// Continuous Collision Detection (CCD): Swept Sphere vs Triangle
// Solves for Time of Impact (TOI) and the collision normal.
// Assumes sphere moves from p0 to p1, triangle is stationary.
bool ccdSphereTriangle(
    vec3 p0, vec3 p1, float radius, 
    vec3 v0, vec3 v1, vec3 v2, 
    out float toi, out vec3 normal
) {
    vec3 dir = p1 - p0;
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 triNormal = normalize(cross(edge1, edge2));

    // Distance from start pos to triangle plane
    float distToPlane = dot(p0 - v0, triNormal);
    
    // Check if moving towards the plane
    float dirDotN = dot(dir, triNormal);
    if (abs(dirDotN) < 1e-6) {
        return false; // Parallel, ignore sweeping across the plane
    }

    // t at which sphere surface touches the plane
    float t = (radius * sign(distToPlane) - distToPlane) / dirDotN;
    if (t < 0.0 || t > 1.0) return false;

    // Check if the intersection point is inside the triangle
    vec3 hitPoint = p0 + dir * t - triNormal * radius * sign(distToPlane);
    
    // Barycentric test
    vec3 w = hitPoint - v0;
    float uu = dot(edge1, edge1);
    float uv = dot(edge1, edge2);
    float vv = dot(edge2, edge2);
    float wu = dot(w, edge1);
    float wv = dot(w, edge2);
    float denom = uv * uv - uu * vv;
    
    float s = (uv * wv - vv * wu) / denom;
    float r = (uv * wu - uu * wv) / denom;
    
    if (s >= 0.0 && r >= 0.0 && (s + r) <= 1.0) {
        toi = t;
        normal = triNormal * sign(distToPlane);
        return true;
    }

    // Edge and vertex CCD tests omitted here for brevity
    return false;
}

// ----------------------------------------------------------------------------
// Parallel BVH Traversal (Subgroup Optimized)
// ----------------------------------------------------------------------------

#define STACK_SIZE 32

// Traverses a single BVH to find self-intersections (Motion bounds).
void computeSelfIntersections(
    BVHArray bvh, 
    uint rootIndex, 
    CollisionPairList outputList
) {
    uvec2 stack[STACK_SIZE];
    int stackPtr = 0;

    stack[stackPtr++] = uvec2(rootIndex, rootIndex);

    while (stackPtr > 0) {
        uvec2 nodePair = stack[--stackPtr];
        uint idxA = nodePair.x;
        uint idxB = nodePair.y;

        BVHNodeAABB nodeA = bvh.nodes[idxA];
        BVHNodeAABB nodeB = bvh.nodes[idxB];

        if (intersectAABB(nodeA.bound, nodeB.bound)) {
            bool aIsLeaf = (nodeA.primitive_count > 0);
            bool bIsLeaf = (nodeB.primitive_count > 0);

            if (aIsLeaf && bIsLeaf) {
                if (idxA != idxB) {
                    uint outIdx = atomicAdd(outputList.count, 1, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
                    outputList.pairs[outIdx] = uvec2(nodeA.left_child_or_primitive_offset, nodeB.left_child_or_primitive_offset);
                }
            } else if (aIsLeaf) {
                if (stackPtr + 2 <= STACK_SIZE) {
                    stack[stackPtr++] = uvec2(idxA, nodeB.left_child_or_primitive_offset);
                    stack[stackPtr++] = uvec2(idxA, nodeB.right_child_offset);
                }
            } else if (bIsLeaf) {
                if (stackPtr + 2 <= STACK_SIZE) {
                    stack[stackPtr++] = uvec2(nodeA.left_child_or_primitive_offset, idxB);
                    stack[stackPtr++] = uvec2(nodeA.right_child_offset, idxB);
                }
            } else {
                if (idxA == idxB) {
                    if (stackPtr + 3 <= STACK_SIZE) {
                        stack[stackPtr++] = uvec2(nodeA.left_child_or_primitive_offset, nodeA.right_child_offset);
                        stack[stackPtr++] = uvec2(nodeA.left_child_or_primitive_offset, nodeA.left_child_or_primitive_offset);
                        stack[stackPtr++] = uvec2(nodeA.right_child_offset, nodeA.right_child_offset);
                    }
                } else {
                    if (stackPtr + 4 <= STACK_SIZE) {
                        stack[stackPtr++] = uvec2(nodeA.left_child_or_primitive_offset, nodeB.left_child_or_primitive_offset);
                        stack[stackPtr++] = uvec2(nodeA.left_child_or_primitive_offset, nodeB.right_child_offset);
                        stack[stackPtr++] = uvec2(nodeA.right_child_offset, nodeB.left_child_or_primitive_offset);
                        stack[stackPtr++] = uvec2(nodeA.right_child_offset, nodeB.right_child_offset);
                    }
                }
            }
        }
    }
}

// Intersect two different BVHs (e.g. two rigid bodies)
void intersectTwoHierarchies(
    BVHArray bvhA, uint rootA, 
    BVHArray bvhB, uint rootB, 
    CollisionPairList outputList
) {
    uvec2 stack[STACK_SIZE];
    int stackPtr = 0;

    stack[stackPtr++] = uvec2(rootA, rootB);

    while (stackPtr > 0) {
        uvec2 pair = stack[--stackPtr];
        BVHNodeAABB nodeA = bvhA.nodes[pair.x];
        BVHNodeAABB nodeB = bvhB.nodes[pair.y];

        if (intersectAABB(nodeA.bound, nodeB.bound)) {
            bool aIsLeaf = (nodeA.primitive_count > 0);
            bool bIsLeaf = (nodeB.primitive_count > 0);

            if (aIsLeaf && bIsLeaf) {
                uint outIdx = atomicAdd(outputList.count, 1, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
                outputList.pairs[outIdx] = uvec2(nodeA.left_child_or_primitive_offset, nodeB.left_child_or_primitive_offset);
            } else if (aIsLeaf) {
                stack[stackPtr++] = uvec2(pair.x, nodeB.left_child_or_primitive_offset);
                stack[stackPtr++] = uvec2(pair.x, nodeB.right_child_offset);
            } else if (bIsLeaf) {
                stack[stackPtr++] = uvec2(nodeA.left_child_or_primitive_offset, pair.y);
                stack[stackPtr++] = uvec2(nodeA.right_child_offset, pair.y);
            } else {
                stack[stackPtr++] = uvec2(nodeA.left_child_or_primitive_offset, nodeB.left_child_or_primitive_offset);
                stack[stackPtr++] = uvec2(nodeA.left_child_or_primitive_offset, nodeB.right_child_offset);
                stack[stackPtr++] = uvec2(nodeA.right_child_offset, nodeB.left_child_or_primitive_offset);
                stack[stackPtr++] = uvec2(nodeA.right_child_offset, nodeB.right_child_offset);
            }
        }
    }
}
