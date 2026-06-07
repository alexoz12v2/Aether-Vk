// ============================================================================
// PhysicsEngine.hlsl
// Target: DXC -> SPIR-V 1.4+ / Vulkan 1.1+ (Bindless BDA via Intrinsics)
// Compile: dxc -spirv -T cs_6_5 -E <kernel> -D KERNEL_<kernel> -fspv-target-env=vulkan1.1 -fspv-extension=SPV_KHR_physical_storage_buffer
// ============================================================================

#ifndef PHYSICS_ENGINE_GLOBALS
#define PHYSICS_ENGINE_GLOBALS

[[vk::constant_id(0)]] const uint SUBGROUP_SIZE = 32;
[[vk::constant_id(1)]] const uint PRIMITIVE_TYPE = 0;

// ------------------------------------------------------------------
// SPIR-V OpBitcast (124) to replace GL_EXT_buffer_reference_uvec2
// Safely casts 64-bit uint2 directly to PhysicalStorageBuffer Pointers
// without ever requiring the Int64 Capability.
// ------------------------------------------------------------------
struct MultiBvhNode; struct RigidBody; struct Wrench; struct ForceEmitter;
struct LcaEntity; struct TLASLeaf; struct DrawIndirectCommand; struct CrossPair;
struct CrossCollisionData; struct EntityHeader; struct MegaParticleData;
struct PairBufferType; struct PackedCollisionsType; struct SparseCollisionsType;
struct DepthIndicesType; struct CrossPairBufferType; struct ClockBufferType;

// If target hardware lacks the shaderInt64 capability (many old laptops/mobile devices) an HLSL implementation
// using using uint64_t will fail to compile or create the pipeline, throwing a Capability Int64 is not supported error.
// In GLSL, GL_EXT_buffer_reference_uvec2 elegantly solves this by allowing uvec2 to be cast directly to a pointer.
// DXC (HLSL) doesn't have an exact equivalent pragma, but we can replicate it perfectly using SPIR-V's
// `OpBitcast` intrinsic ([[vk::ext_instruction(124)]]).
[[vk::ext_instruction(124)]] vk::BufferPointer<float> cast_u2_f(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<uint> cast_u2_u(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<uint2> cast_u2_u2(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<MultiBvhNode> cast_u2_bvh(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<RigidBody> cast_u2_rb(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<Wrench> cast_u2_wrench(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<ForceEmitter> cast_u2_emitter(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<LcaEntity> cast_u2_lca(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<TLASLeaf> cast_u2_leaf(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<EntityHeader> cast_u2_header(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<PairBufferType> cast_u2_pair(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<CrossPairBufferType> cast_u2_cpair(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<PackedCollisionsType> cast_u2_packed(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<SparseCollisionsType> cast_u2_sparse(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<CrossCollisionData> cast_u2_cdata(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<DepthIndicesType> cast_u2_depth(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<ClockBufferType> cast_u2_clock(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<DrawIndirectCommand> cast_u2_indirect(uint2 val);
[[vk::ext_instruction(124)]] vk::BufferPointer<MegaParticleData> cast_u2_mega(uint2 val);

// ------------------------------------------------------------------
// BDA Native Float Atomics
// ------------------------------------------------------------------
[[vk::ext_instruction(227)]] uint spvAtomicCompareExchange(uint64_t Pointer, uint Scope, uint SemanticsEqual, uint SemanticsUnequal, uint Value, uint Comparator);
void bda_atomic_add_float(vk::BufferPointer<uint> buf, uint idx, float val) {
    uint old_val = vk::RawBufferLoad<uint>((uint64_t)buf + idx * 4);
    uint assumed_val;
    do {
        assumed_val = old_val;
        uint new_val = asuint(asfloat(assumed_val) + val);
        old_val = spvAtomicCompareExchange((uint64_t)buf + idx * 4, 1, 0, 0, new_val, assumed_val);
    } while (assumed_val != old_val);
}

#define SHARED_ATOMIC_ADD_FLOAT(dest, val) \
    do { uint _saf_old = dest, _saf_assumed, _saf_orig; do { _saf_assumed = _saf_old; \
    InterlockedCompareExchange(dest, _saf_assumed, asuint(asfloat(_saf_assumed) + (val)), _saf_orig); \
    _saf_old = _saf_orig; } while (_saf_assumed != _saf_old); } while(false)

// ------------------------------------------------------------------
// Metadata Bitfield Definitions
// ------------------------------------------------------------------
#define BVH_FRAME_MACRO  0u
#define BVH_FRAME_MICRO  1u
#define BVH_SHAPE_AABB   0u
#define BVH_SHAPE_OBB    1u
#define BVH_SHAPE_SPHERE 2u

bool bvh_is_leaf(uint meta)   { return (meta & 0x80000000u) != 0u; }
uint bvh_get_index(uint meta) { return meta & 0x07FFFFFFu; }
uint bvh_pack_metadata(bool is_leaf, uint frame, uint shape, uint index) {
    uint meta = index & 0x07FFFFFFu; meta |= (shape & 0x3u) << 27; meta |= (frame & 0x3u) << 29;
    if (is_leaf) meta |= 0x80000000u;
    return meta;
}
bool bvh_node_is_valid(uint2 valid_mask, uint lane_id) {
    if (lane_id < 32) return (valid_mask.x & (1u << lane_id)) != 0u; else return (valid_mask.y & (1u << (lane_id - 32))) != 0u;
}

// ------------------------------------------------------------------
// Memory Layouts (std430) mapped exactly to C++ Host Layouts
// ------------------------------------------------------------------
#define TYPE_PARTICLE_SYSTEM 0
#define TYPE_RIGID_BODY      1
#define TYPE_MICRO_LCA       2

#define AU_TO_KM 149597870.7
#define KM_TO_AU (1.0 / 149597870.7)

struct ColliderId { uint entity_id; uint primitive_index; };

struct PackedPair {
    ColliderId a; ColliderId b; float toi; uint pad1[3];
    float norm_x; float norm_y; float norm_z; uint pad2;
    float pt_x; float pt_y; float pt_z; float penetration_depth;
};

struct SparseCollisionData {
    uint entity_a; uint prim_a; uint entity_b; uint prim_b; float toi; uint is_lca; uint lca_id; uint frame_bda_low; float3 contact_normal; uint frame_bda_high; float3 contact_point; float penetration_depth;
};

struct EntityHeader { uint type; uint pad[3]; };
struct RigidBody { EntityHeader header; float4 position_mass; float4 orientation; float4 linear_vel_drag; float4 angular_vel_drag; float4 inertia_tensor_inv; uint wrench_idx; uint leaf_start_idx; uint leaf_count; uint shape_type; float3 shape_extents; uint pad2; };
struct Wrench { uint force_x; uint force_y; uint force_z; uint torque_x; uint torque_y; uint torque_z; };
struct ForceEmitter { float3 position; float mu; float3 normal; uint type_id; float trunc_distance; float scale_factor; uint _pad[2]; };
struct LcaEntity {
    EntityHeader header;
    float3 center_pos; float scale;
    float3 center_vel; float soi_radius;
    uint frame_type; uint parent_frame_idx; uint bvh_root_index; uint pad0;
    uint64_t entity_id_raw; uint _pad1; uint _pad2;
};
struct TLASLeaf { float3 min_bound; uint entity_idx; float3 max_bound; uint metadata; uint64_t bda; };
struct AABB { float3 minBounds; float3 maxBounds; };
bool intersectAABB(AABB a, AABB b) { return a.maxBounds.x >= b.minBounds.x && a.minBounds.x <= b.maxBounds.x && a.maxBounds.y >= b.minBounds.y && a.minBounds.y <= b.maxBounds.y && a.maxBounds.z >= b.minBounds.z && a.minBounds.z <= b.maxBounds.z; }
struct DrawIndirectCommand { uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance; };
struct CrossPair { uint macro_id; uint micro_id; uint lca_id; uint pad; };
struct CrossCollisionData { uint valid; uint macro_id; uint micro_id; uint lca_id; float toi; uint pad1[3]; float norm_x; float norm_y; float norm_z; uint pad2; float pt_x; float pt_y; float pt_z; float penetration_depth; };
// EntityHeader moved to line 113
struct MegaParticleData { uint id_low; uint id_high; uint age_low; uint age_high; float pos_x; float pos_y; float pos_z; float mass; float vel_x; float vel_y; float vel_z; uint is_active; };

struct MultiBvhNode {
    float min_x[64]; float max_x[64]; float min_y[64]; float max_y[64]; float min_z[64]; float max_z[64];
    uint  child_indices[64]; uint metadata[64]; float masses[64];
    float com_x[64]; float com_y[64]; float com_z[64];
    uint  particle_start[64]; uint particle_count[64];
    uint2 valid_mask; uint  parent_idx; uint  pad;
    uint  permutations[8][64];
};

struct PairBufferType { uint count; uint2 pairs[1000000]; };
struct PackedCollisionsType { uint dispatch_x; uint dispatch_y; uint dispatch_z; uint count; PackedPair pairs[1000000]; };
struct SparseCollisionsType { uint count; SparseCollisionData pairs[1000000]; };
struct DepthIndicesType { uint count; uint indices[1000000]; };
struct CrossPairBufferType { uint count; CrossPair pairs[1000000]; };
struct ClockBufferType { uint2 global_time_us; };


// ------------------------------------------------------------------
// Core Math Helpers
// ------------------------------------------------------------------
uint get_ballot_count(uint4 ballot) { return countbits(ballot.x) + countbits(ballot.y) + countbits(ballot.z) + countbits(ballot.w); }
uint get_ballot_prefix(uint4 ballot, uint lane_idx) {
    uint sum = 0;
    if (lane_idx >= 32) sum += countbits(ballot.x); if (lane_idx >= 64) sum += countbits(ballot.y); if (lane_idx >= 96) sum += countbits(ballot.z);
    if (lane_idx < 32) sum += countbits(ballot.x & ((1u << lane_idx) - 1u)); else if (lane_idx < 64) sum += countbits(ballot.y & ((1u << (lane_idx - 32)) - 1u));
    else if (lane_idx < 96) sum += countbits(ballot.z & ((1u << (lane_idx - 64)) - 1u)); else sum += countbits(ballot.w & ((1u << (lane_idx - 96)) - 1u));
    return sum;
}

uint2 add64(uint2 a, uint2 b) { uint2 res; res.x = a.x + b.x; uint carry = (res.x < a.x) ? 1u : 0u; res.y = a.y + b.y + carry; return res; }
float dt_to_seconds(uint2 dt_micros) { return float(dt_micros.x) * 1e-6 + float(dt_micros.y) * 4294.967296; }

float4 quat_conj(float4 q) { return float4(-q.xyz, q.w); }
float4 quat_mult(float4 q1, float4 q2) { return float4(q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y, q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x, q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w, q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z); }
float3 quat_rotate(float4 q, float3 v) { float3 t = 2.0 * cross(q.xyz, v); return v + q.w * t + cross(q.xyz, t); }
float3 quat_rotate_inv(float4 q, float3 v) { return quat_rotate(quat_conj(q), v); }
float3x3 quat_to_mat3(float4 q) { float xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z, xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z, wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z; return float3x3(1.0-2.0*(yy+zz), 2.0*(xy+wz), 2.0*(xz-wy), 2.0*(xy-wz), 1.0-2.0*(xx+zz), 2.0*(yz+wx), 2.0*(xz+wy), 2.0*(yz-wx), 1.0-2.0*(xx+yy)); }

float4x4 affine_inverse(float4x4 m) {
    float3x3 r = float3x3(m[0].xyz, m[1].xyz, m[2].xyz); float3x3 rt = transpose(r);
    return float4x4(float4(rt[0], 0.0), float4(rt[1], 0.0), float4(rt[2], 0.0), float4(mul(rt, -m[3].xyz), 1.0));
}

bool intersectAABB(float3 amin, float3 amax, float3 bmin, float3 bmax) { return (amin.x <= bmax.x && amax.x >= bmin.x) && (amin.y <= bmax.y && amax.y >= bmin.y) && (amin.z <= bmax.z && amax.z >= bmin.z); }

float3 support_shape(uint shape_type, float3 shape_data, float4x4 transform, float3 dir) {
    float3 local_dir = mul(affine_inverse(transform), float4(dir, 0.0)).xyz; float3 result = (float3)0.0;
    if (shape_type == 0) {
        float radius = shape_data.x; float l = length(local_dir);
        result = (l > 1e-6 ? local_dir / l : float3(1.0, 0.0, 0.0)) * radius;
    } else if (shape_type == 1) {
        float3 extents = shape_data;
        result.x = dot(float3(1,0,0), local_dir) > 0.0 ? extents.x : -extents.x;
        result.y = dot(float3(0,1,0), local_dir) > 0.0 ? extents.y : -extents.y;
        result.z = dot(float3(0,0,1), local_dir) > 0.0 ? extents.z : -extents.z;
    }
    return mul(transform, float4(result, 1.0)).xyz;
}

// ------------------------------------------------------------------
// GJK Math
// ------------------------------------------------------------------
struct MinkowskiPoint { float3 pt; float3 point_a; float3 point_b; };
struct Simplex { MinkowskiPoint points[4]; int count; };
struct Face { int a, b, c; float3 normal; float distance; };

Face create_face_poly(MinkowskiPoint points[16], int a, int b, int c) {
    float3 ab = points[b].pt - points[a].pt, ac = points[c].pt - points[a].pt, n = cross(ab, ac);
    float3 normal = dot(n, n) > 1e-8 ? normalize(n) : float3(1.0, 0.0, 0.0);
    float d = dot(normal, points[a].pt);
    if (d < 0.0) { normal = -normal; d = -d; }
    Face f; f.a = a; f.b = b; f.c = c; f.normal = normal; f.distance = d; return f;
}

void epa_distance(inout Simplex simplex, uint type1, float3 data1, float4x4 trans1, uint type2, float3 data2, float4x4 trans2, out float dist, out float3 p_a, out float3 p_b) {
    MinkowskiPoint polytope_points[16]; int polytope_count = simplex.count;
    for(int i = 0; i < simplex.count; i++) polytope_points[i] = simplex.points[i];
    Face faces[32]; int num_faces = 0;
    if (polytope_count == 4) {
        faces[num_faces++] = create_face_poly(polytope_points, 0, 1, 2); faces[num_faces++] = create_face_poly(polytope_points, 0, 3, 1);
        faces[num_faces++] = create_face_poly(polytope_points, 0, 2, 3); faces[num_faces++] = create_face_poly(polytope_points, 1, 3, 2);
    } else { dist = 0.0; p_a = (float3)0.0; p_b = (float3)0.0; return; }

    for (int iter = 0; iter < 32; ++iter) {
        int closest_face_idx = 0; float min_dist = faces[0].distance;
        for (int i = 1; i < num_faces; ++i) { if (faces[i].distance < min_dist) { min_dist = faces[i].distance; closest_face_idx = i; } }
        Face closest_face = faces[closest_face_idx]; float3 search_dir = closest_face.normal;

        float3 supp_a = support_shape(type1, data1, trans1, search_dir), supp_b = support_shape(type2, data2, trans2, -search_dir);
        float3 new_pt = supp_a - supp_b;

        float d = dot(new_pt, search_dir);
        if (d - min_dist < 1e-4) {
            MinkowskiPoint a = polytope_points[closest_face.a], b = polytope_points[closest_face.b], c = polytope_points[closest_face.c];
            float3 n = closest_face.normal, p = n * min_dist;
            float3 v0 = b.pt - a.pt, v1 = c.pt - a.pt, v2 = p - a.pt;
            float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1), d20 = dot(v2, v0), d21 = dot(v2, v1);
            float denom = d00 * d11 - d01 * d01; float v = 0.333, w = 0.333;
            if (abs(denom) >= 1e-6) { v = (d11 * d20 - d01 * d21) / denom; w = (d00 * d21 - d01 * d20) / denom; }
            float u = 1.0 - v - w;
            dist = -min_dist; p_a = a.point_a * u + b.point_a * v + c.point_a * w; p_b = a.point_b * u + b.point_b * v + c.point_b * w;
            return;
        }
        if (polytope_count >= 16) break;

        MinkowskiPoint mp; mp.pt = new_pt; mp.point_a = supp_a; mp.point_b = supp_b; int new_idx = polytope_count; polytope_points[polytope_count++] = mp;

        int2 edges[64]; int num_edges = 0; int i = 0;
        while (i < num_faces) {
            if (dot(faces[i].normal, new_pt - polytope_points[faces[i].a].pt) > 0.0) {
                Face f = faces[i]; faces[i] = faces[--num_faces];
                int e[6] = {f.a, f.b, f.b, f.c, f.c, f.a};
                for (int j = 0; j < 3; ++j) {
                    int ea = e[j*2], eb = e[j*2+1]; bool found = false;
                    for (int k = 0; k < num_edges; ++k) { if (edges[k].x == eb && edges[k].y == ea) { edges[k] = edges[--num_edges]; found = true; break; } }
                    if (!found && num_edges < 64) edges[num_edges++] = int2(ea, eb);
                }
            } else i++;
        }
        if (num_edges == 0) break;
        for (int k = 0; k < num_edges; ++k) if (num_faces < 32) faces[num_faces++] = create_face_poly(polytope_points, edges[k].x, edges[k].y, new_idx);
    }
    Face closest_face = faces[0];
    MinkowskiPoint a = polytope_points[closest_face.a], b = polytope_points[closest_face.b], c = polytope_points[closest_face.c];
    dist = -closest_face.distance; p_a = a.point_a * 0.333 + b.point_a * 0.333 + c.point_a * 0.334; p_b = a.point_b * 0.333 + b.point_b * 0.333 + c.point_b * 0.334;
}

bool do_simplex(inout Simplex simplex, inout float3 dir) {
    if (simplex.count == 2) {
        MinkowskiPoint a = simplex.points[1], b = simplex.points[0]; float3 ab = b.pt - a.pt, ao = -a.pt;
        if (dot(ab, ao) > 0.0) dir = cross(cross(ab, ao), ab); else { simplex.points[0] = a; simplex.count = 1; dir = ao; }
        return false;
    } else if (simplex.count == 3) {
        MinkowskiPoint a = simplex.points[2], b = simplex.points[1], c = simplex.points[0];
        float3 ab = b.pt - a.pt, ac = c.pt - a.pt, ao = -a.pt, abc = cross(ab, ac);
        if (dot(cross(abc, ac), ao) > 0.0) {
            if (dot(ac, ao) > 0.0) { simplex.points[0] = c; simplex.points[1] = a; simplex.count = 2; dir = cross(cross(ac, ao), ac); }
            else { if (dot(ab, ao) > 0.0) { simplex.points[0] = b; simplex.points[1] = a; simplex.count = 2; dir = cross(cross(ab, ao), ab); } else { simplex.points[0] = a; simplex.count = 1; dir = ao; } }
        } else {
            if (dot(cross(ab, abc), ao) > 0.0) {
                if (dot(ab, ao) > 0.0) { simplex.points[0] = b; simplex.points[1] = a; simplex.count = 2; dir = cross(cross(ab, ao), ab); } else { simplex.points[0] = a; simplex.count = 1; dir = ao; }
            } else {
                if (dot(abc, ao) > 0.0) dir = abc; else { MinkowskiPoint temp = simplex.points[0]; simplex.points[0] = simplex.points[1]; simplex.points[1] = temp; dir = -abc; }
            }
        }
        return false;
    } else if (simplex.count == 4) {
        MinkowskiPoint a = simplex.points[3], b = simplex.points[2], c = simplex.points[1], d = simplex.points[0];
        float3 ab = b.pt - a.pt, ac = c.pt - a.pt, ad = d.pt - a.pt, ao = -a.pt;
        float3 abc = cross(ab, ac), acd = cross(ac, ad), adb = cross(ad, ab);
        if (dot(abc, ao) > 0.0) { simplex.points[0] = c; simplex.points[1] = b; simplex.points[2] = a; simplex.count = 3; dir = abc; return false; }
        else if (dot(acd, ao) > 0.0) { simplex.points[0] = d; simplex.points[1] = c; simplex.points[2] = a; simplex.count = 3; dir = acd; return false; }
        else if (dot(adb, ao) > 0.0) { simplex.points[0] = d; simplex.points[1] = b; simplex.points[2] = a; simplex.count = 3; dir = adb; return false; }
        else return true;
    }
    return false;
}

void compute_closest_points(in Simplex simplex, out float3 closest_a, out float3 closest_b) {
    if (simplex.count == 1) { closest_a = simplex.points[0].point_a; closest_b = simplex.points[0].point_b; }
    else if (simplex.count == 2) {
        MinkowskiPoint a = simplex.points[1], b = simplex.points[0];
        float3 ab = b.pt - a.pt; float l2 = dot(ab, ab); float t = l2 > 1e-6 ? clamp(dot(-a.pt, ab) / l2, 0.0, 1.0) : 0.0;
        closest_a = a.point_a + (b.point_a - a.point_a) * t; closest_b = a.point_b + (b.point_b - a.point_b) * t;
    } else if (simplex.count == 3) {
        MinkowskiPoint a = simplex.points[2], b = simplex.points[1], c = simplex.points[0];
        float3 ab = b.pt - a.pt, ac = c.pt - a.pt; float3 n = cross(ab, ac); float n_len_sq = dot(n, n);
        if (n_len_sq < 1e-6) { closest_a = a.point_a; closest_b = a.point_b; return; }
        float3 ao = -a.pt; float u = dot(cross(ac, n), ao) / n_len_sq, v = dot(cross(n, ab), ao) / n_len_sq, w = 1.0 - u - v;
        closest_a = a.point_a * w + b.point_a * u + c.point_a * v; closest_b = a.point_b * w + b.point_b * u + c.point_b * v;
    } else { closest_a = (float3)0.0; closest_b = (float3)0.0; }
}

float gjk_distance_generic(uint type1, float3 data1, float4x4 trans1, uint type2, float3 data2, float4x4 trans2, out float3 p_a, out float3 p_b) {
    float3 dir = float3(1.0, 0.0, 0.0);
    float3 support_a = support_shape(type1, data1, trans1, -dir); float3 support_b = support_shape(type2, data2, trans2, dir);
    Simplex simplex;
    simplex.points[0].pt = support_a - support_b; simplex.points[0].point_a = support_a; simplex.points[0].point_b = support_b; simplex.count = 1;
    float3 v = simplex.points[0].pt;

    for (int i = 0; i < 64; ++i) {
        if (dot(v, v) < 1e-6) break;
        dir = -v;
        float3 p1 = support_shape(type1, data1, trans1, dir); float3 p2 = support_shape(type2, data2, trans2, -dir);
        MinkowskiPoint w; w.pt = p1 - p2; w.point_a = p1; w.point_b = p2;
        if (dot(w.pt, dir) - dot(v, dir) < 1e-4) break;
        simplex.points[simplex.count++] = w;
        if (do_simplex(simplex, dir)) { float dist_out; epa_distance(simplex, type1, data1, trans1, type2, data2, trans2, dist_out, p_a, p_b); return dist_out; }
        if (simplex.count == 1) v = simplex.points[0].pt;
        else if (simplex.count == 2 || simplex.count == 3) { compute_closest_points(simplex, p_a, p_b); v = p_a - p_b; }
    }
    compute_closest_points(simplex, p_a, p_b); return length(p_a - p_b);
}

bool compute_toi_generic(
    uint type1, float3 data1, float4x4 trans1_start, float3 v1, uint type2, float3 data2, float4x4 trans2_start, float3 v2,
    float time_tolerance, int max_iterations, out float out_toi, out float3 out_normal, out float3 out_contact_point, out float out_depth
) {
    float t = 0.0; float3 v_rel = v1 - v2;
    if (length(v_rel) < 1e-6) {
        float3 p_a, p_b; float dist = gjk_distance_generic(type1, data1, trans1_start, type2, data2, trans2_start, p_a, p_b);
        if (dist <= 0.0) {
            out_toi = 0.0; out_depth = dist < 0.0 ? -dist : 0.0;
            float3 n = p_a - p_b; float n_len = length(n);
            out_normal = n_len > 1e-6 ? n / n_len : float3(1.0, 0.0, 0.0); out_contact_point = (p_a + p_b) * 0.5; return true;
        }
        return false;
    }
    for (int i = 0; i < max_iterations; ++i) {
        float4x4 cur_trans1 = trans1_start; cur_trans1[3].xyz += v1 * t;
        float4x4 cur_trans2 = trans2_start; cur_trans2[3].xyz += v2 * t;
        float3 p_a, p_b; float dist = gjk_distance_generic(type1, data1, cur_trans1, type2, data2, cur_trans2, p_a, p_b);
        if (dist <= time_tolerance) {
            out_toi = t; out_depth = dist < 0.0 ? -dist : 0.0;
            float3 n = p_a - p_b; float n_len = length(n);
            out_normal = n_len > 1e-6 ? n / n_len : float3(1.0, 0.0, 0.0); out_contact_point = (p_a + p_b) * 0.5; return true;
        }
        float3 n = dist > 1e-6 ? normalize(p_a - p_b) : float3(1.0, 0.0, 0.0);
        float v_closing = -dot(v_rel, n);
        if (v_closing <= 0.0) return false;
        t += dist / v_closing;
        if (t > 1.0) return false;
    }
    return false;
}

#endif // PHYSICS_ENGINE_GLOBALS
