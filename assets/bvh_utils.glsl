// ============================================================================
// BVH and Collision Utilities (Vulkan SPIR-V)
// ============================================================================
// Makes heavy use of Vulkan 1.1+ features:
// - Physical Storage Buffers (buffer_reference) for pointer-chasing in BVHs
// - Vulkan Memory Model for correct atomics
// - Subgroup operations for parallel traversal and work distribution

#ifndef BVH_UTILS_GLSL
#define BVH_UTILS_GLSL

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_memory_scope_semantics : require
// TODO remove int8, which isn't universally supported (should be available on majority though)
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
// 16-bit STORAGE (load/store from SSBOs) — universally supported on Vulkan 1.1+ hardware.
#extension GL_EXT_shader_16bit_storage                         : require
// 16-bit ARITHMETIC (float16_t registers) — missing on Pascal (GTX 10xx) and older.
// Demoted to 'enable'; usage guarded by NATIVE_FLOAT16 macro below.
#extension GL_EXT_shader_explicit_arithmetic_types_float16     : enable
// ability to use [[dont_unroll]]
#extension GL_EXT_control_flow_attributes : require

// ── float16 arithmetic abstraction ──────────────────────────────────────────
// NATIVE_FLOAT16 is injected by compile_shaders.sh / compile_shaders.ps1:
//   -DNATIVE_FLOAT16=1  → shaderFloat16 available; use native f16vec4 / float16_t for ALU
//   -DNATIVE_FLOAT16=0  → no native fp16 ALU; promote registers to vec4 / float
// Buffer layout (f16vec4 fields in ParticleChunkData) is UNCHANGED in both paths:
// GL_EXT_shader_16bit_storage handles the 16↔32-bit widening on load/store.
#ifndef NATIVE_FLOAT16
#define NATIVE_FLOAT16 1   // default: assume capable (preserves existing behaviour)
#endif

#if NATIVE_FLOAT16
#  define FVEC4    f16vec4
#  define FP_T     float16_t
#  define FP(x)    float16_t(x)
#  define FP_ZERO  0.0hf
#  define STORE_FVEC4(x) (x)
#else
#  define FVEC4    vec4
#  define FP_T     float
#  define FP(x)    float(x)
#  define FP_ZERO  0.0
#  define STORE_FVEC4(x) f16vec4(x)
#endif

layout(constant_id = 0) const uint SUBGROUP_SIZE = 32;
layout(constant_id = 4) const uint PARTICLES_IN_LEAF = 64;

// ---------------------------------------------------------------------------
// BVH traversal stack depth (specialization constants, set by host).
// The host computes: min(16384 / (256/SUBGROUP_SIZE * 4) - 1, MAX)
// to guarantee total shared memory stays ≤ 16 KB.
//
// BVH_STACK_DEPTH:       broad-phase scene/cross-lca traversal (default 128)
// BVH_STACK_DEPTH_SHORT: gravity/particle-self traversal (default 64)
// ---------------------------------------------------------------------------
layout(constant_id = 2) const uint BVH_STACK_DEPTH       = 128;
layout(constant_id = 3) const uint BVH_STACK_DEPTH_SHORT = 64;

// ------------------------------------------------------------------
// 1. Unified Multi-BVH Node — raw buffer + arithmetic accessors
//
// The node layout mirrors TlasMultiNode<N> exactly (all fields are
// packed arrays of N floats/uints, in declaration order):
//
//   Field index  Name             Byte offset from node base
//   0            min_x            0
//   1            max_x            N*4
//   2            min_y            N*8
//   3            max_y            N*12
//   4            min_z            N*16
//   5            max_z            N*20
//   6            child_indices    N*24
//   7            metadata         N*28
//   8            masses           N*32
//   9            com_x            N*36
//   10           com_y            N*40
//   11           com_z            N*44
//   12           particle_start   N*48
//   13           particle_count   N*52
//   14           force_x          N*56
//   15           force_y          N*60
//   16           force_z          N*64
//   17           valid_mask.x     N*68 + 0
//   17           valid_mask.y     N*68 + 4
//   18           parent_idx       N*68 + 8
//   (pad)                         N*68 + 12
//
// Since SUBGROUP_SIZE is a specialization constant (no default array
// sizing), all offsets are pure arithmetic — no compile-time baking.
// ------------------------------------------------------------------

// One node is NODE_STRIDE uints = (17*N + 4) uints.
// (permutations field removed — never read by any shader)
// We expose the raw flat buffer and let accessors index into it.
layout(buffer_reference, std430, buffer_reference_align = 16) buffer TlasNodeBuffer {
    uint data[];
};

// Stride of one node in uints.  All arithmetic uses SUBGROUP_SIZE (spec const).
// node_stride = 17*N + 4  (17 per-lane fields + valid_mask.xy + parent + pad)
uint tlas_node_stride()                { return 17u * SUBGROUP_SIZE + 4u; }
uint tlas_node_base(uint node_idx)     { return node_idx * tlas_node_stride(); }


// Field base (in uints) within a node, at lane `lane`
uint tlas_min_x_u      (uint nb, uint lane) { return nb + 0u  * SUBGROUP_SIZE + lane; }
uint tlas_max_x_u      (uint nb, uint lane) { return nb + 1u  * SUBGROUP_SIZE + lane; }
uint tlas_min_y_u      (uint nb, uint lane) { return nb + 2u  * SUBGROUP_SIZE + lane; }
uint tlas_max_y_u      (uint nb, uint lane) { return nb + 3u  * SUBGROUP_SIZE + lane; }
uint tlas_min_z_u      (uint nb, uint lane) { return nb + 4u  * SUBGROUP_SIZE + lane; }
uint tlas_max_z_u      (uint nb, uint lane) { return nb + 5u  * SUBGROUP_SIZE + lane; }
uint tlas_child_u      (uint nb, uint lane) { return nb + 6u  * SUBGROUP_SIZE + lane; }
uint tlas_metadata_u   (uint nb, uint lane) { return nb + 7u  * SUBGROUP_SIZE + lane; }
uint tlas_mass_u       (uint nb, uint lane) { return nb + 8u  * SUBGROUP_SIZE + lane; }
uint tlas_com_x_u      (uint nb, uint lane) { return nb + 9u  * SUBGROUP_SIZE + lane; }
uint tlas_com_y_u      (uint nb, uint lane) { return nb + 10u * SUBGROUP_SIZE + lane; }
uint tlas_com_z_u      (uint nb, uint lane) { return nb + 11u * SUBGROUP_SIZE + lane; }
uint tlas_pstart_u     (uint nb, uint lane) { return nb + 12u * SUBGROUP_SIZE + lane; }
uint tlas_pcount_u     (uint nb, uint lane) { return nb + 13u * SUBGROUP_SIZE + lane; }
uint tlas_force_x_u    (uint nb, uint lane) { return nb + 14u * SUBGROUP_SIZE + lane; }
uint tlas_force_y_u    (uint nb, uint lane) { return nb + 15u * SUBGROUP_SIZE + lane; }
uint tlas_force_z_u    (uint nb, uint lane) { return nb + 16u * SUBGROUP_SIZE + lane; }
// valid_mask: 2 uints right after the 17 per-lane arrays (at offset 17*N uints)
uint tlas_valid_mask_x_u(uint nb)           { return nb + 17u * SUBGROUP_SIZE + 0u; }
uint tlas_valid_mask_y_u(uint nb)           { return nb + 17u * SUBGROUP_SIZE + 1u; }
// parent_idx is at 17*N + 2  (pad at 17*N + 3)
uint tlas_parent_u     (uint nb)            { return nb + 17u * SUBGROUP_SIZE + 2u; }


// Typed read helpers (float accessors call uintBitsToFloat)
float tlas_min_x    (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_min_x_u  (tlas_node_base(ni), lane)]); }
float tlas_max_x    (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_max_x_u  (tlas_node_base(ni), lane)]); }
float tlas_min_y    (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_min_y_u  (tlas_node_base(ni), lane)]); }
float tlas_max_y    (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_max_y_u  (tlas_node_base(ni), lane)]); }
float tlas_min_z    (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_min_z_u  (tlas_node_base(ni), lane)]); }
float tlas_max_z    (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_max_z_u  (tlas_node_base(ni), lane)]); }
uint  tlas_child    (TlasNodeBuffer b, uint ni, uint lane) { return               b.data[tlas_child_u    (tlas_node_base(ni), lane)];  }
uint  tlas_metadata (TlasNodeBuffer b, uint ni, uint lane) { return               b.data[tlas_metadata_u (tlas_node_base(ni), lane)];  }
float tlas_mass     (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_mass_u   (tlas_node_base(ni), lane)]); }
float tlas_com_x    (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_com_x_u  (tlas_node_base(ni), lane)]); }
float tlas_com_y    (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_com_y_u  (tlas_node_base(ni), lane)]); }
float tlas_com_z    (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_com_z_u  (tlas_node_base(ni), lane)]); }
uint  tlas_pstart   (TlasNodeBuffer b, uint ni, uint lane) { return               b.data[tlas_pstart_u   (tlas_node_base(ni), lane)];  }
uint  tlas_pcount   (TlasNodeBuffer b, uint ni, uint lane) { return               b.data[tlas_pcount_u   (tlas_node_base(ni), lane)];  }
float tlas_force_x  (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_force_x_u(tlas_node_base(ni), lane)]); }
float tlas_force_y  (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_force_y_u(tlas_node_base(ni), lane)]); }
float tlas_force_z  (TlasNodeBuffer b, uint ni, uint lane) { return uintBitsToFloat(b.data[tlas_force_z_u(tlas_node_base(ni), lane)]); }

uvec2 tlas_valid_mask(TlasNodeBuffer b, uint ni) {
    uint nb = tlas_node_base(ni);
    return uvec2(b.data[tlas_valid_mask_x_u(nb)], b.data[tlas_valid_mask_y_u(nb)]);
}
uint tlas_parent(TlasNodeBuffer b, uint ni) {
    return b.data[tlas_parent_u(tlas_node_base(ni))];
}

// Write helpers
void tlas_write_min_x   (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_min_x_u   (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_max_x   (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_max_x_u   (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_min_y   (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_min_y_u   (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_max_y   (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_max_y_u   (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_min_z   (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_min_z_u   (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_max_z   (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_max_z_u   (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_child   (TlasNodeBuffer b, uint ni, uint lane, uint  v) { b.data[tlas_child_u   (tlas_node_base(ni), lane)] = v; }
void tlas_write_metadata(TlasNodeBuffer b, uint ni, uint lane, uint  v) { b.data[tlas_metadata_u(tlas_node_base(ni), lane)] = v; }
void tlas_write_mass    (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_mass_u    (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_com_x   (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_com_x_u   (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_com_y   (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_com_y_u   (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_com_z   (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_com_z_u   (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_pstart  (TlasNodeBuffer b, uint ni, uint lane, uint  v) { b.data[tlas_pstart_u  (tlas_node_base(ni), lane)] = v; }
void tlas_write_pcount  (TlasNodeBuffer b, uint ni, uint lane, uint  v) { b.data[tlas_pcount_u  (tlas_node_base(ni), lane)] = v; }
void tlas_write_force_x (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_force_x_u (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_force_y (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_force_y_u (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_force_z (TlasNodeBuffer b, uint ni, uint lane, float v) { b.data[tlas_force_z_u (tlas_node_base(ni), lane)] = floatBitsToUint(v); }
void tlas_write_valid_mask(TlasNodeBuffer b, uint ni, uvec2 v) {
    uint nb = tlas_node_base(ni);
    b.data[tlas_valid_mask_x_u(nb)] = v.x;
    b.data[tlas_valid_mask_y_u(nb)] = v.y;
}
void tlas_write_parent(TlasNodeBuffer b, uint ni, uint v) {
    b.data[tlas_parent_u(tlas_node_base(ni))] = v;
}

// Keep MultiBvhBuffer as a type alias so old push-constant field names still compile.
// All node accesses go through the tlas_* accessors above.
#define MultiBvhBuffer TlasNodeBuffer

// ------------------------------------------------------------------
// 2. Metadata Bitfield Definitions & Helpers
// ------------------------------------------------------------------
#define BVH_FRAME_MACRO  0u
#define BVH_FRAME_MICRO  1u
#define BVH_SHAPE_AABB   0u
#define BVH_SHAPE_OBB    1u
#define BVH_SHAPE_SPHERE 2u
#define BVH_SHAPE_SUB_TLAS 3u

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
struct PackedPair {
    ColliderId a;
    ColliderId b;
    float toi;
    uint is_lca;
    uint lca_id;
    uint frame_bda_low;
    float norm_x;
    float norm_y;
    float norm_z;
    uint frame_bda_high;
    float pt_x;
    float pt_y;
    float pt_z;
    float penetration_depth;
};
struct SparseCollisionData { uint entity_a; uint prim_a; uint entity_b; uint prim_b; float toi; uint is_lca; uint lca_id; uint frame_bda_low; vec3 contact_normal; uint frame_bda_high; vec3 contact_point; float penetration_depth; };

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
    uint frame_idx;
};
struct Wrench { uint force_x; uint force_y; uint force_z; uint torque_x; uint torque_y; uint torque_z; };
struct ForceEmitter {
    // Gravity (type_id=0): position is emitter world-space in AU; mu is G*M in km³/s² (JPL default).
    // Planar  (type_id=1): position is plane origin; mu is base force magnitude.
    vec3  position;
    float mu;
    vec3  normal;           // Planar: plane normal. Gravity: unused.
    uint  type_id;          // 0 = Gravity, 1 = Planar
    float trunc_distance;   // Planar: max signed dist above which force is zero. Gravity: unused.
    float beta;             // Radiation-pressure β; mu_eff = (1−β)·mu. 0 = pure gravity.
    uint  _pad[2];
};
struct KinematicBody { uint own_frame_id; float scale; vec3 position; uint frame_type; float mu; };
struct GpuReferenceFrame {
    vec3 center_pos;
    float scale;
    vec3 center_vel;
    float soi_radius;
    uint frame_type;
    uint parent_frame_idx;
    uint bvh_root_index;
    uint pad0;
    uint entity_id_raw_low;
    uint entity_id_raw_high;
    uint frame_bda_low;
    uint frame_bda_high;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer GpuReferenceFrameArray { GpuReferenceFrame frames[]; };
struct TLASLeaf { vec3 min_bound; uint entity_idx; vec3 max_bound; uint metadata; };
struct RenderParticleData { uint id_low; uint id_high; uint age_low; uint age_high; vec3 position; float mass; vec3 velocity; uint is_active; vec3 force; };
struct DrawIndirectCommand { uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance; };

layout(buffer_reference, std430, buffer_reference_align = 16) buffer ParticleData {
  // See PARTICLE_FIELDS explaination in gpu.rs
  uint data[];
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer AtomicCounters { uint counts[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer MortonArray { uvec2 entries[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer PairBuffer { uint count; uint capacity; uvec2 pairs[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer PackedCollisions { uint dispatch_x; uint dispatch_y; uint dispatch_z; uint count; PackedPair pairs[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer SparseCollisions { uint count; uint capacity; SparseCollisionData pairs[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer RigidBodyArray { RigidBody bodies[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer RigidBodyUintArray {
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
layout(buffer_reference, std430, buffer_reference_align = 4) buffer WrenchArray { Wrench wrenches[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer EmitterArray { ForceEmitter emitters[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer KinematicArray { KinematicBody bodies[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer LeafBuffer { TLASLeaf leaves[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer DepthIndices { uint count; uint indices[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer CollapseMapBuffer { uint binary_roots[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer ImpulseOutput { vec3 impulses[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer OutputTOI { uint min_tc_uint; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer ClusterListBuffer { uint cluster_indices[]; };
// ─── 64-bit engine clock (lo, hi) ───────────────────────────────────────────
layout(buffer_reference, std430, buffer_reference_align = 8) buffer ClockBuffer {
  // lo = [0], hi = [1] — stored as uvec2 (x=lo, y=hi)
  uvec2 global_time_us;
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer HistogramBuffer { uint counts[]; };
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
layout(buffer_reference, std430, buffer_reference_align = 4) buffer CrossPairBuffer { uint count; uint capacity; CrossPair pairs[]; };

struct CrossCollisionData { uint valid; uint macro_id; uint micro_id; uint lca_id; float toi; vec3 contact_normal; vec3 contact_point; float penetration_depth; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer CrossSparseCollisions { uint count; uint capacity; CrossCollisionData pairs[]; };

// ------------------------------------------------------------
// New Particle system stuff
//------------------------------------------------------------

const int PCHUNK_SIZE = 256;
const int PCHUNK_VEC4_SIZE = 64;

layout(buffer_reference, std430, buffer_reference_align = 16) buffer ParticleChunkData {
    vec4 positionX[PCHUNK_VEC4_SIZE]; // p1p2: pos_N | emit: pos_N+1
    vec4 positionY[PCHUNK_VEC4_SIZE]; // in metres (m)
    vec4 positionZ[PCHUNK_VEC4_SIZE];

    f16vec4 velocityX[PCHUNK_VEC4_SIZE]; // p1p2: vel_N | emit: vel_N+1/2 | p4p5: vel_N+1
    f16vec4 velocityY[PCHUNK_VEC4_SIZE]; // in metres per second (m/s)
    f16vec4 velocityZ[PCHUNK_VEC4_SIZE];

    f16vec4 invMass[PCHUNK_VEC4_SIZE]; // 1 / grams (1/g)

    f16vec4 forceX[PCHUNK_VEC4_SIZE];
    f16vec4 forceY[PCHUNK_VEC4_SIZE];
    f16vec4 forceZ[PCHUNK_VEC4_SIZE]; // in g * m / s^2

    f16vec4 beta[PCHUNK_VEC4_SIZE];

    uvec4 spawnTime[PCHUNK_VEC4_SIZE]; // in 1/300 seconds of unscaled simulation time, which can then be scaled and compared to user provided TTL
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer ParticleChunkBuffer {
    ParticleChunkData chunks[];
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer ParticlePageTable {
    // 32 bytes header
    // --- Start VkDrawIndirectCommand (16 bytes) ---
    uint particleCount;   // Maps to vertexCount
    uint instanceCount;   // Maps to instanceCount (Initialize to 1)
    uint firstVertex;     // Maps to firstVertex   (Initialize to 0)
    uint firstInstance;   // Maps to firstInstance (Initialize to 0)
    // --- End VkDrawIndirectCommand ---

    // --- Rest of your header ---
    uint activeChunkCount;

    // Optional: 12 bytes of padding to keep `chunks` aligned to 16 bytes
    // (Helps with coalesced memory reads depending on how you use chunks)
    uint pad0;
    uint pad1;
    uint pad2;

    // if we compare this different than zero, then we can cast to ParticleChunk
    // There are indices to support 2^17 particles (131072). The number of allocated indices
    // is actually double the amount of supported particles such that we can have space to
    // host
    // first index of the second half is actually the write index into the second half, cause
    // if we are compacting, it means we are not full capacity
    uint chunks[];
};

uvec2 offsetAddress(uvec2 baseAddr, uint off) {
    uint carry;
    uint low = uaddCarry(baseAddr.x, off, carry);
    uint high = baseAddr.y + carry;
    return uvec2(low, high);
}

struct ParticleForceEmitter {
    // Gravity (type_id=0): position is particle-system local in metres (m). mu is G*M in m³/s² (JPL default is km³/s²).
    // Planar  (type_id=1): position is plane origin; mu is base force magnitude.
    vec4 positionMu;
    uint typeId;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer ParticleEmitterArray {
    ParticleForceEmitter[] emitters;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer ParticleFreeChunkStack {
    uint count;      // How many chunks are actually free
    uint indices[];  // physical indices of free chunks
};

#define PARTICLE_MINIMUM_MASS 0.001

#endif // BVH_UTILS_GLSL