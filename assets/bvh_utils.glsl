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

#ifndef SUBGROUP_SIZE
layout(constant_id = 0) const uint SUBGROUP_SIZE = 32;
#endif

// ------------------------------------------------------------------
// 1. Unified Multi-BVH Node (Shared by TLAS, Body BLAS, Particle BLAS)
// ------------------------------------------------------------------
struct MultiBvhNode {
    float min_x[SUBGROUP_SIZE]; float max_x[SUBGROUP_SIZE];
    float min_y[SUBGROUP_SIZE]; float max_y[SUBGROUP_SIZE];
    float min_z[SUBGROUP_SIZE]; float max_z[SUBGROUP_SIZE];
    uint  child_indices[SUBGROUP_SIZE]; uint metadata[SUBGROUP_SIZE];
    float masses[SUBGROUP_SIZE];
    float com_x[SUBGROUP_SIZE]; float com_y[SUBGROUP_SIZE]; float com_z[SUBGROUP_SIZE];
    uint  particle_start[SUBGROUP_SIZE]; uint particle_count[SUBGROUP_SIZE];

    uvec2 valid_mask;
    uint  parent_idx; // Unifies Binary & N-Ary Trees without legacy types
    uint  pad;        // 16-byte alignment
    uint  permutations[8][SUBGROUP_SIZE];
};

layout(buffer_reference, scalar, buffer_reference_align = 16) buffer MultiBvhBuffer {
    MultiBvhNode nodes[];
};

// ------------------------------------------------------------------
// 2. Metadata Bitfield Definitions & Helpers
// ------------------------------------------------------------------
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
    meta |= (shape & 0x3u) << 27; meta |= (frame & 0x3u) << 29;
    if (is_leaf) meta |= 0x80000000u;
    return meta;
}

bool bvh_node_is_valid(uvec2 valid_mask, uint lane_id) {
    if (lane_id < 32) return (valid_mask.x & (1u << lane_id)) != 0u;
    else return (valid_mask.y & (1u << (lane_id - 32))) != 0u;
}

// ----------------------------------------------------------------------------
// Physics & Struct Definitions
// ----------------------------------------------------------------------------
#define TYPE_PARTICLE_SYSTEM 0
#define TYPE_RIGID_BODY      1
#define TYPE_MICRO_LCA       2

#define AU_TO_KM 149597870.7
#define KM_TO_AU (1.0 / 149597870.7)
#define M_EARTH_TO_KG 5.9722e24
#define KG_TO_M_EARTH (1.0 / 5.9722e24)

struct ColliderId { uint entity_id; uint primitive_index; };
struct PackedPair { ColliderId a; ColliderId b; float toi; vec3 contact_normal; vec3 contact_point; float penetration_depth; };
struct SparseCollisionData { uint valid; uint entity_a; uint prim_a; uint entity_b; uint prim_b; float toi; vec3 contact_normal; vec3 contact_point; float penetration_depth; };

struct RigidBody {
    vec4 position_mass;       // xyz: world pos, w: mass
    vec4 orientation;         // quaternion (x,y,z,w)
    vec4 linear_vel_drag;     // xyz: linear velocity, w: linear damping coeff
    vec4 angular_vel_drag;    // xyz: angular velocity, w: angular damping coeff
    vec4 inertia_tensor_inv;  // xyz: diagonal local inverse inertia
    uint wrench_idx;          // Pointer to the associated accumulated Wrench
    uint leaf_start_idx;      // Mapping to the first leaf Wrench (for rb_force_assign)
    uint leaf_count;          // Number of leaves
    uint shape_type;          // BVH_SHAPE_*
    vec3 shape_extents;       // Dimensions
    uint pad2;
};
struct Wrench { uint force_x; uint force_y; uint force_z; uint torque_x; uint torque_y; uint torque_z; };
struct ForceEmitter {
    vec3 position;
    float mu; // type_id = 0 -> G * M, type_id = 1 -> base force
    vec3 normal; // type_id = 0 -> unused, type_id = 1 -> plane normal
    uint type_id; // 0 = Gravity, 1 = Planar
    float trunc_distance; // type_id = 0 -> unused, type_id = 1 -> max distance point - plane over which force is applied
    float scale_factor;
    uint _pad[2];
};
struct KinematicBody { uint own_frame_id; float scale; vec3 position; uint frame_type; float mu; };
struct LcaEntity { MultiBvhBuffer bvh; mat4 transform; mat4 inv_transform; vec3 linear_velocity; uint root_index; vec3 angular_velocity; uint type; uint primitive_offset; uint total_primitives; uint frame_scale_type; float scale_factor; uint shape_type; vec3 shape_data; };
struct TLASLeaf { vec3 min_bound; uint entity_idx; vec3 max_bound; uint metadata; };
struct RenderParticleData { uint id_low; uint id_high; uint age_low; uint age_high; vec3 position; float mass; vec3 velocity; uint is_active; };
struct DrawIndirectCommand { uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance; };

layout(buffer_reference, scalar, buffer_reference_align = 16) buffer ParticleData {
  // Note: semantically a float. Declared as a uint to allow for atomic operations
  // 4 floats per particle as the following in aosoa format:
  // After p1 p2
  // 0-2 position at midpoint
  // 3-5 velocity at midpoint (TODO check how it evolves)
  // 6 mass (TODO now it's set to zero. correct that)
  // 7-8 padding? (TODO Check better)
  uint data[];
  // equivalent:
  //   float mass[SUBGROUP_SIZE];
  //   float dvA[3 * SUBGROUP_SIZE];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer AtomicCounters { uint counts[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer MortonArray { uvec2 entries[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer PairBuffer { uint count; uvec2 pairs[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer PackedCollisions { uint dispatch_x; uint dispatch_y; uint dispatch_z; uint count; PackedPair pairs[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer SparseCollisions { uint count; SparseCollisionData pairs[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer RigidBodyArray { RigidBody bodies[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer RigidBodyUintArray {
  // TODO: convenient getters.
  // Declared as uint to allow for atomic operations (we don't support them for floats)
  // RigidBody struct is 20 floats in aosoa format
  // 0-3 pos_mass
  // 4-7 orient
  // 8-11 lin_vel
  // 12-15 ang_vel
  // 16-19 inv_inertia
  uint data[];
  // equivalent:
  //   float pos_mass[3 * SUBGROUP_SIZE]; // example: 32 xs, 32 ys, 32 zs
  //   float orient[3 * SUBGROUP_SIZE];
  //   float lin_vel[3 * SUBGROUP_SIZE];
  //   float ang_vel[3 * SUBGROUP_SIZE];
  //   float inv_inertia[3 * SUBGROUP_SIZE];
};
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer WrenchArray { Wrench wrenches[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer EmitterArray { ForceEmitter emitters[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer KinematicArray { KinematicBody bodies[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer LcaEntityArray { LcaEntity entities[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer LeafBuffer { TLASLeaf leaves[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer DepthIndices { uint count; uint indices[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer CollapseMapBuffer { uint binary_roots[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer ImpulseOutput { vec3 impulses[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer OutputTOI { uint min_tc_uint; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer ClusterListBuffer { uint cluster_indices[]; };
// ─── 64-bit engine clock (lo, hi) ───────────────────────────────────────────
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer ClockBuffer {
  // lo = [0], hi = [1] — stored as uvec2 (x=lo, y=hi)
  uvec2 global_time_us;
};
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer HistogramBuffer { uint counts[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer RenderParticleDataArray { RenderParticleData data[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer IndirectArray { DrawIndirectCommand commands[]; };

// Helper macros for easy read/write of uint-backed floats
#define P_READ(buf, idx) uintBitsToFloat((buf).data[idx])
#define P_WRITE(buf, idx, val) (buf).data[idx] = floatBitsToUint(val)

struct AABB { vec3 minBounds; vec3 maxBounds; };
bool intersectAABB(AABB a, AABB b) { return a.maxBounds.x >= b.minBounds.x && a.minBounds.x <= b.maxBounds.x && a.maxBounds.y >= b.minBounds.y && a.minBounds.y <= b.maxBounds.y && a.maxBounds.z >= b.minBounds.z && a.minBounds.z <= b.maxBounds.z; }
bool intersectAABB(vec3 amin, vec3 amax, vec3 bmin, vec3 bmax) { return (amin.x <= bmax.x && amax.x >= bmin.x) && (amin.y <= bmax.y && amax.y >= bmin.y) && (amin.z <= bmax.z && amax.z >= bmin.z); }

// ------------------------------------------------------------
// Cross LCA support structs
//------------------------------------------------------------

struct CrossPair { uint macro_id; uint micro_id; uint lca_id; uint pad; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer CrossPairBuffer { uint count; CrossPair pairs[]; };

struct CrossCollisionData { uint valid; uint macro_id; uint micro_id; uint lca_id; float toi; vec3 contact_normal; vec3 contact_point; float penetration_depth; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer CrossSparseCollisions { uint count; CrossCollisionData pairs[]; };

#endif // BVH_UTILS_GLSL