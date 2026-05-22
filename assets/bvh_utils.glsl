// ============================================================================
// BVH and Collision Utilities (Vulkan SPIR-V)
// ============================================================================
// Makes heavy use of Vulkan 1.1+ features:
// - Physical Storage Buffers (buffer_reference) for pointer-chasing in BVHs
// - Vulkan Memory Model for correct atomics
// - Subgroup operations for parallel traversal and work distribution

#ifndef BVH_UTILS_GLSL
#define BVH_UTILS_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

// Define branching factor to match the hardware subgroup size.
// Can be injected via Vulkan Specialization Constants (layout(constant_id = X)).
#ifndef SUBGROUP_SIZE
layout(constant_id = 0) const uint SUBGROUP_SIZE = 32;
#endif

// ------------------------------------------------------------------
// 1. Unified Multi-BVH Node (Shared by TLAS, Body BLAS, Particle BLAS)
// ------------------------------------------------------------------
// Using std430/scalar, arrays of floats have a strict 4-byte stride.
struct MultiBvhNode {
    float min_x[SUBGROUP_SIZE];
    float max_x[SUBGROUP_SIZE];
    float min_y[SUBGROUP_SIZE];
    float max_y[SUBGROUP_SIZE];
    float min_z[SUBGROUP_SIZE];
    float max_z[SUBGROUP_SIZE];

    uint  child_indices[SUBGROUP_SIZE];
    uint  metadata[SUBGROUP_SIZE];
    float masses[SUBGROUP_SIZE];
    float com_x[SUBGROUP_SIZE];
    float com_y[SUBGROUP_SIZE];
    float com_z[SUBGROUP_SIZE];

    // Cluster Threshold Tracking
    uint  particle_start[SUBGROUP_SIZE];
    uint  particle_count[SUBGROUP_SIZE];

    uvec2 valid_mask; // 64-bit emulation using uvec2
    uint  permutations[8][SUBGROUP_SIZE]; // u32 per slot (low byte = ordering index); matches Rust [[u32;N];8]
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer MultiBvhBuffer {
    MultiBvhNode nodes[];
};

// ------------------------------------------------------------------
// 2. Instance Descriptor (Targeted by TLAS Leaves)
// ------------------------------------------------------------------
struct InstanceDescriptor {
    mat4 transform;       // Local/Micro Space -> Macro Space (+x=right, -y=forward, +z=up)
    mat4 inv_transform;   // Macro Space -> Local/Micro Space (Used to scale rays/velocities)
    vec4 shape_data;      // xyz: Half-Extents (OBB/AABB), x: Radius (Sphere)
    uint blas_root_idx;   // Pointer to the BLAS root MultiBvhNode in the BLAS buffer
    uint pad0, pad1, pad2;// 16-byte std430 alignment padding
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer InstanceBuffer {
    InstanceDescriptor descriptors[];
};

// ------------------------------------------------------------------
// 3. Metadata Bitfield Definitions & Helpers
// ------------------------------------------------------------------
// Bit 31:    IsLeaf (1 bit) -> 0 = Internal, 1 = Leaf
// Bit 29-30: Frame  (2 bits) -> 00 = Macro (AU/M_earth), 01 = Micro (km/kg)
// Bit 27-28: Shape  (2 bits) -> 00 = AABB, 01 = OBB, 10 = Sphere
// Bit 0-26:  Index  (27 bits) -> Points to child node OR InstanceDescriptor

#define BVH_FRAME_MACRO  0u
#define BVH_FRAME_MICRO  1u

#define BVH_SHAPE_AABB   0u
#define BVH_SHAPE_OBB    1u
#define BVH_SHAPE_SPHERE 2u

bool bvh_is_leaf(uint meta)   { return (meta & 0x80000000u) != 0u; }
uint bvh_get_frame(uint meta) { return (meta >> 29) & 0x3u; }
uint bvh_get_shape(uint meta) { return (meta >> 27) & 0x3u; }
uint bvh_get_index(uint meta) { return meta & 0x07FFFFFFu; }

uint bvh_pack_metadata(bool is_leaf, uint frame, uint shape, uint index) {
    uint meta = index & 0x07FFFFFFu;
    meta |= (shape & 0x3u) << 27;
    meta |= (frame & 0x3u) << 29;
    if (is_leaf) meta |= 0x80000000u;
    return meta;
}

// ----------------------------------------------------------------------------
// Legacy Support structs (To be removed/replaced as pipeline updates)
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
    uint parent_idx;
    uint node_type;
    float mass;
    vec3 center_of_mass;
    float _pad;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer BVHArray {
    BVHNodeAABB nodes[];
};

// Helper to safely read from scalar storage without triggering cross-compiler padding mismatch bugs
BVHNodeAABB read_bvh_node(BVHArray bvh, uint idx) {
    BVHNodeAABB node;
    node.bound.minBounds = bvh.nodes[idx].bound.minBounds;
    node.bound.maxBounds = bvh.nodes[idx].bound.maxBounds;
    node.left_child_or_primitive_offset = bvh.nodes[idx].left_child_or_primitive_offset;
    node.right_child_offset = bvh.nodes[idx].right_child_offset;
    node.primitive_count = bvh.nodes[idx].primitive_count;
    node.parent_idx = bvh.nodes[idx].parent_idx;
    node.node_type = bvh.nodes[idx].node_type;
    node.mass = bvh.nodes[idx].mass;
    node.center_of_mass = bvh.nodes[idx].center_of_mass;
    node._pad = 0.0;
    return node;
}

// ----------------------------------------------------------------------------
// Motion BLAS Node (Binary BVH, Exactly 64 bytes)
// ----------------------------------------------------------------------------
struct MotionBvhNode {
    AABB aabbs[2];
    uint child_ptrs[2];
    uint parent_idx;
    uint is_leaf;
    uint pad[2];
};

layout(buffer_reference, scalar, buffer_reference_align = 16) buffer MotionBvhBuffer {
    MotionBvhNode nodes[];
};

struct ColliderId {
    uint entity_id;
    uint primitive_index;
};

struct PackedPair {
    ColliderId a;
    ColliderId b;
    float toi;
    vec3 contact_normal;
    vec3 contact_point;
    float penetration_depth;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer PackedCollisions {
    uint dispatch_x;
    uint dispatch_y;
    uint dispatch_z;
    uint count;
    PackedPair pairs[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) writeonly buffer CollisionPairList {
    uint count;
    uvec2 pairs[]; // Legacy
};

// ----------------------------------------------------------------------------
// Shared Memory Caching (Workgroup Local Memory)
// ----------------------------------------------------------------------------
// Defines a shared memory cache that subgroups can collaboratively populate.
// This is exceptionally useful when many threads (e.g. many rays or many particles)
// are traversing the same top-level BVH nodes repeatedly.

#define DECLARE_SHARED_BVH_CACHE(name) shared MultiBvhNode name

#define CACHE_BLOCK_COLLABORATIVE(GLOBAL_BVH, BLOCK_IDX, SHARED_CACHE) \
    do { \
        uint _tid = gl_SubgroupInvocationID; \
        if (_tid < SUBGROUP_SIZE) { \
            SHARED_CACHE.min_x[_tid] = GLOBAL_BVH.nodes[BLOCK_IDX].min_x[_tid]; \
            SHARED_CACHE.max_x[_tid] = GLOBAL_BVH.nodes[BLOCK_IDX].max_x[_tid]; \
            SHARED_CACHE.min_y[_tid] = GLOBAL_BVH.nodes[BLOCK_IDX].min_y[_tid]; \
            SHARED_CACHE.max_y[_tid] = GLOBAL_BVH.nodes[BLOCK_IDX].max_y[_tid]; \
            SHARED_CACHE.min_z[_tid] = GLOBAL_BVH.nodes[BLOCK_IDX].min_z[_tid]; \
            SHARED_CACHE.max_z[_tid] = GLOBAL_BVH.nodes[BLOCK_IDX].max_z[_tid]; \
            SHARED_CACHE.child_indices[_tid] = GLOBAL_BVH.nodes[BLOCK_IDX].child_indices[_tid]; \
            SHARED_CACHE.metadata[_tid] = GLOBAL_BVH.nodes[BLOCK_IDX].metadata[_tid]; \
        } \
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
    float denom = uu * vv - uv * uv;

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

#define STACK_SIZE 64

#endif // BVH_UTILS_GLSL