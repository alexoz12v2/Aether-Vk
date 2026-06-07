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
struct RigidBody { EntityHeader header; float pos_x; float pos_y; float pos_z; float mass; float orient_x; float orient_y; float orient_z; float orient_w; float lin_vel_x; float lin_vel_y; float lin_vel_z; float lin_drag; float ang_vel_x; float ang_vel_y; float ang_vel_z; float ang_drag; float inv_inertia_x; float inv_inertia_y; float inv_inertia_z; float pad_inv; uint wrench_idx; uint leaf_start_idx; uint leaf_count; uint shape_type; float shape_x; float shape_y; float shape_z; uint pad2; };
struct Wrench { uint force_x; uint force_y; uint force_z; uint torque_x; uint torque_y; uint torque_z; };
struct ForceEmitter { float pos_x; float pos_y; float pos_z; float mu; float norm_x; float norm_y; float norm_z; uint type_id; float trunc_distance; float scale_factor; uint _pad[2]; };
struct LcaEntity {
    EntityHeader header;
    float center_pos_x; float center_pos_y; float center_pos_z; float scale;
    float center_vel_x; float center_vel_y; float center_vel_z; float soi_radius;
    uint frame_type; uint parent_frame_idx; uint bvh_root_index; uint pad0;
    uint64_t entity_id_raw; uint _pad1; uint _pad2;
};
struct TLASLeaf { float min_x; float min_y; float min_z; uint entity_idx; float max_x; float max_y; float max_z; uint metadata; };
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


// ============================================================================
// KERNELS START HERE
// ============================================================================

// --- motion_refit ---


// --- TRANSLATED KERNELS --- 

// --- hlsl_integrate_particles_p1_p2.txt ---
// @assets/sim/integrate_particles_p1_p2.comp
//
// Particle Velocity-Verlet Predictor — Phase 1 & 2
// ─────────────────────────────────────────────────
// Frame-start invariant: AOSOA slots 7/8/9 hold F(x_n) from the previous frame.
//
//   v_{n+½} = v_n + (dt/2) · M⁻¹ · F(x_n)     [half-kick]
//   x_{n+1} = x_n + dt · v_{n+½}               [full position leap]
//
// After writing, CLEARS slots 7/8/9 to 0 so the unified force-generation pass
// (barnes_hut, bp_particle_self, narrow-phase) can safely atomicAdd into them.
// The half-kick velocity v_{n+½} is stored temporarily in slots 3/4/5 for
// integrate_particles_p4_5 to complete the VV corrector step.
#ifndef P_READ
#define P_READ(ptr, offset) vk::RawBufferLoad<float>((uint64_t)ptr + (offset) * 4)
#define P_WRITE(ptr, offset, val) vk::RawBufferStore<float>((uint64_t)ptr + (offset) * 4, val)
#endif

struct PushConstants_integrate_particles_p1_p2 {
    MegaParticleData particles;
    float dt;
    uint total_particles;
};

[[vk::push_constant]]
PushConstants_integrate_particles_p1_p2 pc;

[numthreads(128, 1, 1)]
void integrate_particles_p1_p2(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint gid = DispatchThreadID.x;
    if (gid >= pc.total_particles) return;

    uint base = (gid / SUBGROUP_SIZE) * (10u * SUBGROUP_SIZE) + (gid % SUBGROUP_SIZE);

    float mass = P_READ(pc.particles, base + 6u * SUBGROUP_SIZE);
    if (mass <= 0.0) return;

    float inv_m = 1.0 / mass;
    float half_dt = 0.5 * pc.dt;

    float3 v_n = float3(P_READ(pc.particles, base + 3u * SUBGROUP_SIZE), P_READ(pc.particles, base + 4u * SUBGROUP_SIZE), P_READ(pc.particles, base + 5u * SUBGROUP_SIZE));
    float3 f_n = float3(P_READ(pc.particles, base + 7u * SUBGROUP_SIZE), P_READ(pc.particles, base + 8u * SUBGROUP_SIZE), P_READ(pc.particles, base + 9u * SUBGROUP_SIZE));

    float3 v_half = v_n + f_n * inv_m * half_dt;
    float3 pos_n = float3(P_READ(pc.particles, base + 0u * SUBGROUP_SIZE), P_READ(pc.particles, base + 1u * SUBGROUP_SIZE), P_READ(pc.particles, base + 2u * SUBGROUP_SIZE));
    float3 pos_next = pos_n + v_half * pc.dt;

    P_WRITE(pc.particles, base + 0u * SUBGROUP_SIZE, pos_next.x);
    P_WRITE(pc.particles, base + 1u * SUBGROUP_SIZE, pos_next.y);
    P_WRITE(pc.particles, base + 2u * SUBGROUP_SIZE, pos_next.z);

    P_WRITE(pc.particles, base + 3u * SUBGROUP_SIZE, v_half.x);
    P_WRITE(pc.particles, base + 4u * SUBGROUP_SIZE, v_half.y);
    P_WRITE(pc.particles, base + 5u * SUBGROUP_SIZE, v_half.z);

    P_WRITE(pc.particles, base + 7u * SUBGROUP_SIZE, 0.0);
    P_WRITE(pc.particles, base + 8u * SUBGROUP_SIZE, 0.0);
    P_WRITE(pc.particles, base + 9u * SUBGROUP_SIZE, 0.0);
}


// --- hlsl_integrate_bodies_p3.txt ---



#ifdef KERNEL_integrate_bodies_p3
struct PushConstants {
    uint64_t rigid_bodies;
    uint64_t wrenches;
    uint64_t emitters;
    float dt;
    uint n_bodies;
    uint n_iterations;
    uint num_emitters;
};

[[vk::push_constant]]
PushConstants pc;
#endif

[numthreads(32, 1, 1)]
void integrate_bodies_p3(uint3 tid : SV_DispatchThreadID) {
    uint id = tid.x;
    if (id >= pc.n_bodies) return;

    uint64_t body_addr = pc.rigid_bodies + id * sizeof(RigidBody);
    RigidBody body = BDA_LOAD(RigidBody, body_addr);

    float mass = body.position_mass.w;
    float inv_m = (mass > 0.0) ? 1.0 / mass : 0.0;
    float3 I_inv = body.inertia_tensor_inv.xyz;
    float3 I_fwd = float3(
        (I_inv.x > 1e-14) ? 1.0 / I_inv.x : 0.0,
        (I_inv.y > 1e-14) ? 1.0 / I_inv.y : 0.0,
        (I_inv.z > 1e-14) ? 1.0 / I_inv.z : 0.0
    );

    float3 pos_n = body.position_mass.xyz;
    float4 q_n = body.orientation;
    float3 v_n = body.linear_vel_drag.xyz;
    float3 w_n = body.angular_vel_drag.xyz;

    uint w_idx = body.wrench_idx;
    uint64_t wrench_addr = pc.wrenches + w_idx * sizeof(Wrench);
    Wrench wrench = BDA_LOAD(Wrench, wrench_addr);

    float3 f_n = float3(asfloat(wrench.force_x), asfloat(wrench.force_y), asfloat(wrench.force_z));
    float3 t_n = float3(asfloat(wrench.torque_x), asfloat(wrench.torque_y), asfloat(wrench.torque_z));

    for (uint e = 0; e < pc.num_emitters; ++e) {
        uint64_t emitter_addr = pc.emitters + e * sizeof(ForceEmitter);
        ForceEmitter emitter = BDA_LOAD(ForceEmitter, emitter_addr);

        float3 em_pos = emitter.position;
        float em_mu = emitter.mu;
        float3 em_norm = emitter.normal;
        uint em_type = emitter.type_id;
        float em_trunc = emitter.trunc_distance;
        float em_scale = emitter.scale_factor;

        if (em_type == 0) {
            float3 r = em_pos - pos_n;
            float s_dist_sq = dot(r, r) * em_scale * em_scale;
            if (s_dist_sq > 1e-6) {
                float s_dist = sqrt(s_dist_sq);
                float s_dist3 = s_dist_sq * s_dist;
                float s_dist5 = s_dist3 * s_dist_sq;
                float softening = 1.0 - exp(-s_dist5);
                float force_mag = (em_mu * mass * softening) / s_dist_sq;
                f_n += normalize(r) * force_mag;
            }
        } else if (em_type == 1) {
            float dist = dot(pos_n - em_pos, em_norm);
            if (dist >= 0.0 && dist <= em_trunc) {
                f_n += em_norm * em_mu;
            }
        }
    }

    float half_dt = 0.5 * pc.dt;
    float3 a_lin = f_n * inv_m;
    float3 v_mid = v_n + half_dt * a_lin;
    float3 pos_next = pos_n + pc.dt * v_mid;
    float3 v_next = v_n + pc.dt * a_lin;

    float3 t_local = quat_rotate_inv(q_n, t_n);
    float3 w_n_local = quat_rotate_inv(q_n, w_n);
    float3 w_mid_local = w_n_local;

    for (uint iter = 0; iter < pc.n_iterations; ++iter) {
        float3 gyro = cross(w_mid_local, I_fwd * w_mid_local);
        float3 a_ang = I_inv * (t_local - gyro);
        w_mid_local = w_n_local + half_dt * a_ang;
    }

    float3 w_next_local = 2.0 * w_mid_local - w_n_local;
    float3 w_next = quat_rotate(q_n, w_next_local);
    float3 w_mid_world = quat_rotate(q_n, w_mid_local);
    float4 omega_pure = float4(w_mid_world, 0.0);
    float4 q_next = normalize(q_n + half_dt * quat_mult(omega_pure, q_n));

    body.position_mass = float4(pos_next, mass);
    body.orientation = q_next;
    body.linear_vel_drag = float4(v_next, body.linear_vel_drag.w);
    body.angular_vel_drag = float4(w_next, body.angular_vel_drag.w);

    BDA_STORE(RigidBody, body_addr, body);

    wrench.force_x = 0;
    wrench.force_y = 0;
    wrench.force_z = 0;
    wrench.torque_x = 0;
    wrench.torque_y = 0;
    wrench.torque_z = 0;

    BDA_STORE(Wrench, wrench_addr, wrench);
}


// --- hlsl_rb_force_assign.txt ---



#ifndef BDA_LOAD
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)
#endif

#ifndef SPV_SCOPE_DEVICE
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(237)]] uint spvAtomicUMin([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(230)]] uint spvAtomicCompareExchange([[vk::ext_reference]] uint64_t ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);
#endif

void AtomicAddFloatBDA(uint64_t addr, float val) {
    uint old_val = BDA_LOAD(uint, addr);
    uint assumed_val;
    do {
        assumed_val = old_val;
        uint new_val = asuint(asfloat(assumed_val) + val);
        old_val = spvAtomicCompareExchange(addr, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, SPV_SEMANTICS_RELAXED, new_val, assumed_val);
    } while (assumed_val != old_val);
}

#ifdef KERNEL_rb_force_assign
struct PushConstants {
    uint64_t rigid_bodies;
    uint64_t wrenches;
    uint n_bodies;
    uint _pad;
};

[[vk::push_constant]]
PushConstants pc;
#endif

groupshared float3 sh_f[32];
groupshared float3 sh_t[32];

[numthreads(128, 1, 1)]
void rb_force_assign(
    uint3 GroupID : SV_GroupID,
    uint GroupIndex : SV_GroupIndex)
{
    uint body_id = GroupID.x;
    if (body_id >= pc.n_bodies) return;

    uint local_id = GroupIndex;
    
    uint wave_size = WaveGetLaneCount();
    uint sg_id = local_id / wave_size;
    uint sg_lane = WaveGetLaneIndex();
    uint subgroups_per_wg = 128u / wave_size;

    // RigidBody size is 128 bytes
    uint64_t body_addr = pc.rigid_bodies + body_id * 128;
    
    // Offsets: wrench_idx=96, leaf_start_idx=100, leaf_count=104
    uint com_wrench = BDA_LOAD(uint, body_addr + 96);
    uint leaf_start = BDA_LOAD(uint, body_addr + 100);
    uint leaf_count = BDA_LOAD(uint, body_addr + 104);

    float3 acc_f = float3(0.0, 0.0, 0.0);
    float3 acc_t = float3(0.0, 0.0, 0.0);

    for (uint i = local_id; i < leaf_count; i += 128u) {
        // Wrench size is 24 bytes
        uint64_t w_addr = pc.wrenches + (leaf_start + i) * 24;
        
        uint fx = BDA_LOAD(uint, w_addr + 0);
        uint fy = BDA_LOAD(uint, w_addr + 4);
        uint fz = BDA_LOAD(uint, w_addr + 8);
        uint tx = BDA_LOAD(uint, w_addr + 12);
        uint ty = BDA_LOAD(uint, w_addr + 16);
        uint tz = BDA_LOAD(uint, w_addr + 20);
        
        acc_f += float3(asfloat(fx), asfloat(fy), asfloat(fz));
        acc_t += float3(asfloat(tx), asfloat(ty), asfloat(tz));
    }

    acc_f.x = WaveActiveSum(acc_f.x);
    acc_f.y = WaveActiveSum(acc_f.y);
    acc_f.z = WaveActiveSum(acc_f.z);
    
    acc_t.x = WaveActiveSum(acc_t.x);
    acc_t.y = WaveActiveSum(acc_t.y);
    acc_t.z = WaveActiveSum(acc_t.z);

    if (sg_lane == 0u) {
        sh_f[sg_id] = acc_f;
        sh_t[sg_id] = acc_t;
    }

    GroupMemoryBarrierWithGroupSync();

    if (local_id == 0u) {
        float3 total_f = float3(0.0, 0.0, 0.0);
        float3 total_t = float3(0.0, 0.0, 0.0);
        for (uint s = 0u; s < subgroups_per_wg; ++s) {
            total_f += sh_f[s];
            total_t += sh_t[s];
        }
        
        uint64_t com_wrench_addr = pc.wrenches + com_wrench * 24;
        AtomicAddFloatBDA(com_wrench_addr + 0, total_f.x);
        AtomicAddFloatBDA(com_wrench_addr + 4, total_f.y);
        AtomicAddFloatBDA(com_wrench_addr + 8, total_f.z);
        AtomicAddFloatBDA(com_wrench_addr + 12, total_t.x);
        AtomicAddFloatBDA(com_wrench_addr + 16, total_t.y);
        AtomicAddFloatBDA(com_wrench_addr + 20, total_t.z);
    }
}


// --- hlsl_integrate_particles_p4_5.txt ---
// @assets/sim/integrate_particles_p4_5.comp
//
// Particle Velocity-Verlet Corrector — Phase 4 & 5
// ─────────────────────────────────────────────────
// Invariant entering this pass:
//   • AOSOA slots 3/4/5 hold v_{n+½}  (stored by integrate_particles_p1_p2)
//   • AOSOA slots 7/8/9 hold F(x_{n+1}) (written by force generators after p3)
//
//   v_{n+1} = v_{n+½} + (dt/2) · M⁻¹ · F(x_{n+1})    [VV corrector]
//
// The force buffer is intentionally NOT cleared — F(x_{n+1}) persists as
// F(x_n) for the NEXT frame's integrate_particles_p1_p2 pass.
//
// Thread 0 additionally advances the emulated 64-bit engine clock:
//   global_time_us += dt_us    (uvec2 carry-propagating addition from imex_math.glsl)
//
// Target: SPIR-V 1.4 · Vulkan 1.1 · flexible across all hardware subgroup sizes.

struct PushConstants_integrate_particles_p4_5 {
    uint64_t particles;
    uint64_t clock;
    float dt;
    uint total_particles;
    uint dt_us_lo;
    uint dt_us_hi;
    uint current_time_lo;
    uint current_time_hi;
};

[[vk::push_constant]]
PushConstants_integrate_particles_p4_5 pc;
[numthreads(128, 1, 1)]
void integrate_particles_p4_5(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint gid = DispatchThreadID.x;

    // ── Thread 0: advance the 64-bit engine clock exactly once per frame ─────
    // This must happen regardless of particle count so the clock always ticks.
    if (gid == 0u) {
        uint2 t_n  = uint2(pc.current_time_lo, pc.current_time_hi);
        uint2 dt_u = uint2(pc.dt_us_lo, pc.dt_us_hi);
        uint2 res;
        res.x = t_n.x + dt_u.x;
        uint carry = (res.x < t_n.x) ? 1u : 0u;
        res.y = t_n.y + dt_u.y + carry;
        BDA_STORE(uint2, pc.clock, res);
    }

    if (gid >= pc.total_particles) return;

    uint block = gid / SUBGROUP_SIZE;
    uint lane  = gid % SUBGROUP_SIZE;
    uint base  = block * (10u * SUBGROUP_SIZE) + lane;

    // ── Skip inactive / massless particles ────────────────────────────────
    uint mass_offset = (base + 6u * SUBGROUP_SIZE) * 4;
    float mass = asfloat(BDA_LOAD(uint, pc.particles + mass_offset));
    if (mass <= 0.0) return;

    float inv_m   = 1.0 / mass;
    float half_dt = 0.5 * pc.dt;

    // ── Load v_{n+½} (written by p1_p2) ──────────────────────────────────
    float3 v_half = float3(
        asfloat(BDA_LOAD(uint, pc.particles + (base + 3u * SUBGROUP_SIZE) * 4)),
        asfloat(BDA_LOAD(uint, pc.particles + (base + 4u * SUBGROUP_SIZE) * 4)),
        asfloat(BDA_LOAD(uint, pc.particles + (base + 5u * SUBGROUP_SIZE) * 4))
    );

    // ── Load F(x_{n+1}) (written by force generators after p3) ───────────
    float3 f_next = float3(
        asfloat(BDA_LOAD(uint, pc.particles + (base + 7u * SUBGROUP_SIZE) * 4)),
        asfloat(BDA_LOAD(uint, pc.particles + (base + 8u * SUBGROUP_SIZE) * 4)),
        asfloat(BDA_LOAD(uint, pc.particles + (base + 9u * SUBGROUP_SIZE) * 4))
    );

    // ── VV Corrector ─────────────────────────────────────────────────────
    float3 v_next = v_half + f_next * inv_m * half_dt;

    // Write v_{n+1} back — force buffer stays intact for next frame
    BDA_STORE(uint, pc.particles + (base + 3u * SUBGROUP_SIZE) * 4, asuint(v_next.x));
    BDA_STORE(uint, pc.particles + (base + 4u * SUBGROUP_SIZE) * 4, asuint(v_next.y));
    BDA_STORE(uint, pc.particles + (base + 5u * SUBGROUP_SIZE) * 4, asuint(v_next.z));
}

// --- hlsl_bp_clear.txt ---
// BDA Memory Access Macros
// ------------------------------------------------------------------
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

// ------------------------------------------------------------------
// SPIR-V Atomic Intrinsics for 64-bit BDA Pointers
// ------------------------------------------------------------------
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

// Force DXC to emit explicit SPIR-V atomics mapped to 64-bit PhysicalStorageBuffers
[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(237)]] uint spvAtomicUMin([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(230)]] uint spvAtomicCompareExchange([[vk::ext_reference]] uint64_t ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);

#include "../debug_utils.h"


struct PushConstants {
    uint64_t raw_scene_pairs;
    uint64_t out_rb_rb;
    uint64_t out_rb_ps;
    uint64_t out_rb_lca;
    uint64_t internal_pairs;
};

[[vk::push_constant]]
PushConstants pc;

[numthreads(1, 1, 1)]
void bp_clear(uint3 DispatchThreadID : SV_DispatchThreadID) {
    BDA_STORE(uint, pc.raw_scene_pairs, 0u);
    BDA_STORE(uint, pc.out_rb_rb, 0u);
    BDA_STORE(uint, pc.out_rb_ps, 0u);
    BDA_STORE(uint, pc.out_rb_lca, 0u);
    BDA_STORE(uint, pc.internal_pairs, 0u);
}

// --- hlsl_bp_bounds_gen.txt ---
// @assets/sim/bp_bounds_gen.comp

#ifdef KERNEL_bp_bounds_gen
struct PushConstants {
    uint64_t scene_entities;
    uint64_t particles;
    uint64_t tlas_leaves;
    uint2    dt_us;
    uint     total_entities;
    uint     num_rigid_bodies;
    float    particle_radius;
};

[[vk::push_constant]]
PushConstants pc;
#endif

[numthreads(256, 1, 1)]
void bp_bounds_gen(uint3 tid : SV_DispatchThreadID) {
    uint id = tid.x;
    if (id >= pc.total_entities) return;

    float dt = dt_to_seconds(pc.dt_us);
    float3 center, extents, vel;
    uint shape_type;
    uint64_t bda;

    if (id < pc.num_rigid_bodies) {
        uint64_t body_addr = pc.scene_entities + id * sizeof(RigidBody);
        RigidBody body = BDA_LOAD(RigidBody, body_addr);
        
        center = body.position_mass.xyz;
        extents = body.shape_extents;
        vel = body.linear_vel_drag.xyz;
        shape_type = body.shape_type;
        bda = body_addr;
    } else {
        uint particle_system_idx = id - pc.num_rigid_bodies;
        // The bounds of a particle system should ideally be computed over all its particles.
        // For now, since particles are grouped into entities (32 particles per entity),
        // we approximate the bounds using the center of the first particle in the group.
        uint base = particle_system_idx * (10 * SUBGROUP_SIZE);
        
        center = float3(
            asfloat(BDA_LOAD(uint, pc.particles + (base + 0) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (base + 1 * SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (base + 2 * SUBGROUP_SIZE) * 4))
        );
        
        extents = float3(pc.particle_radius * 16.0, pc.particle_radius * 16.0, pc.particle_radius * 16.0); // Rough approximation for 32 particles
        
        vel = float3(
            asfloat(BDA_LOAD(uint, pc.particles + (base + 3 * SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (base + 4 * SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (base + 5 * SUBGROUP_SIZE) * 4))
        );
        
        shape_type = BVH_SHAPE_SPHERE;
        bda = pc.particles + particle_system_idx * (10 * SUBGROUP_SIZE * 4); // Address of this chunk of 32 particles
    }

    float3 static_min = center - extents;
    float3 static_max = center + extents;
    float3 sweep = vel * dt;

    uint64_t leaf_addr = pc.tlas_leaves + id * sizeof(TLASLeaf);
    TLASLeaf leaf = BDA_LOAD(TLASLeaf, leaf_addr);
    
    leaf.min_bound = min(static_min, static_min + sweep);
    leaf.max_bound = max(static_max, static_max + sweep);
    leaf.entity_idx = id;
    leaf.metadata = bvh_pack_metadata(true, BVH_FRAME_MACRO, shape_type, id);
    leaf.bda = bda;

    BDA_STORE(TLASLeaf, leaf_addr, leaf);
}

// --- hlsl_bp_scene.txt ---
#include "../debug_utils.hlsl"


// BDA Memory Access Macros
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);

struct PushConstants {
    uint64_t tlas_bvh;
    uint64_t query_leaves;
    uint64_t overlapping_pairs;
    uint     tlas_root_index;
    uint     total_queries;
};

[[vk::push_constant]]
PushConstants pc;

#define SUBGROUP_SIZE 32
#define SUBGROUPS_PER_WG 8

#define NODE_STRIDE 2832
#define NODE_MIN_X 0
#define NODE_MAX_X 128
#define NODE_MIN_Y 256
#define NODE_MAX_Y 384
#define NODE_MIN_Z 512
#define NODE_MAX_Z 640
#define NODE_CHILD_INDICES 768
#define NODE_METADATA 896
#define NODE_VALID_MASK 1792

#define LEAF_STRIDE 48
#define LEAF_MIN_BOUND 0
#define LEAF_ENTITY_IDX 12
#define LEAF_MAX_BOUND 16

groupshared uint shared_stacks[SUBGROUPS_PER_WG][32];
groupshared uint shared_stack_ptrs[SUBGROUPS_PER_WG];

[numthreads(256, 1, 1)]
void bp_scene(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    uint lane_id = WaveGetLaneIndex();
    uint subgroup_id = gl_LocalInvocationID.x / SUBGROUP_SIZE;
    uint query_idx = gl_WorkGroupID.x * SUBGROUPS_PER_WG + subgroup_id;

    if (query_idx >= pc.total_queries) return;

    float3 my_min, my_max; 
    uint my_ent_id;

    if (lane_id == 0) {
        uint64_t leaf_addr = pc.query_leaves + query_idx * LEAF_STRIDE;
        my_min = BDA_LOAD(float3, leaf_addr + LEAF_MIN_BOUND);
        my_ent_id = BDA_LOAD(uint, leaf_addr + LEAF_ENTITY_IDX);
        my_max = BDA_LOAD(float3, leaf_addr + LEAF_MAX_BOUND);

        shared_stacks[subgroup_id][0] = pc.tlas_root_index;
        shared_stack_ptrs[subgroup_id] = 1;
    }

    my_min.x = WaveReadLaneAt(my_min.x, 0);
    my_min.y = WaveReadLaneAt(my_min.y, 0);
    my_min.z = WaveReadLaneAt(my_min.z, 0);
    my_max.x = WaveReadLaneAt(my_max.x, 0);
    my_max.y = WaveReadLaneAt(my_max.y, 0);
    my_max.z = WaveReadLaneAt(my_max.z, 0);
    my_ent_id = WaveReadLaneAt(my_ent_id, 0);

    while (true) {
        GroupMemoryBarrier();

        uint stack_ptr = shared_stack_ptrs[subgroup_id];
        if (stack_ptr == 0) break;

        stack_ptr--;
        uint node_idx = shared_stacks[subgroup_id][stack_ptr];
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr;

        uint64_t node_addr = pc.tlas_bvh + node_idx * NODE_STRIDE;
        uint meta = BDA_LOAD(uint, node_addr + NODE_METADATA + lane_id * 4);
        uint2 valid_mask = BDA_LOAD(uint2, node_addr + NODE_VALID_MASK);
        bool valid = bvh_node_is_valid(valid_mask, lane_id);

        float3 c_min = float3(
            BDA_LOAD(float, node_addr + NODE_MIN_X + lane_id * 4),
            BDA_LOAD(float, node_addr + NODE_MIN_Y + lane_id * 4),
            BDA_LOAD(float, node_addr + NODE_MIN_Z + lane_id * 4)
        );
        float3 c_max = float3(
            BDA_LOAD(float, node_addr + NODE_MAX_X + lane_id * 4),
            BDA_LOAD(float, node_addr + NODE_MAX_Y + lane_id * 4),
            BDA_LOAD(float, node_addr + NODE_MAX_Z + lane_id * 4)
        );
        uint child_payload = BDA_LOAD(uint, node_addr + NODE_CHILD_INDICES + lane_id * 4);

        uint entity_id = bvh_get_index(meta);

        bool hit = valid && intersectAABB(my_min, my_max, c_min, c_max);
        bool is_leaf = bvh_is_leaf(meta);

        bool hit_leaf = hit && is_leaf && (my_ent_id < entity_id);
        bool hit_node = hit && !is_leaf;

        uint leaf_count  = WaveActiveCountBits(hit_leaf);
        uint leaf_offset = WavePrefixCountBits(hit_leaf);

        if (leaf_count > 0) {
            uint base_idx = 0;
            if (lane_id == 0) {
                base_idx = spvAtomicIAdd(pc.overlapping_pairs, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, leaf_count);
            }
            base_idx = WaveReadLaneAt(base_idx, 0);

            if (hit_leaf && base_idx + leaf_offset < 10000u) {
                uint2 pair_val = uint2(my_ent_id, entity_id);
                uint64_t pair_addr = pc.overlapping_pairs + 8 + (base_idx + leaf_offset) * 8;
                BDA_STORE(uint2, pair_addr, pair_val);
            }
        }

        uint node_count  = WaveActiveCountBits(hit_node);
        uint push_offset = WavePrefixCountBits(hit_node);

        if (hit_node) shared_stacks[subgroup_id][stack_ptr + push_offset] = child_payload;
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr + node_count;
    }
}


// --- hlsl_bp_classify.txt ---


// BDA Memory Access Macros
#ifndef BDA_LOAD
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#endif
#ifndef BDA_STORE
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)
#endif

// SPIR-V Atomic Intrinsics for 64-bit BDA Pointers
#ifndef SPV_SCOPE_DEVICE
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
#endif

#ifndef TYPE_PARTICLE_SYSTEM
#define TYPE_PARTICLE_SYSTEM 0
#define TYPE_RIGID_BODY      1
#define TYPE_MICRO_LCA       2
#endif

struct PushConstants {
    uint64_t raw_pairs;
    uint2 out_rb_rb;
    uint2 out_rb_ps;
    uint2 out_ps_ps;
    uint64_t tlas_leaves;
    uint max_pairs;
    uint num_rigid_bodies;
};

[[vk::push_constant]]
PushConstants pc;

[numthreads(256, 1, 1)]
void bp_classify(uint3 tid : SV_DispatchThreadID) {
    uint id = tid.x;
    
    uint count = BDA_LOAD(uint, pc.raw_pairs);
    if (id >= count) return;

    // pairs starts at offset 8 (uint count is 4 bytes, padded to 8 for uint2 alignment)
    uint2 pair = BDA_LOAD(uint2, pc.raw_pairs + 8 + id * 8);
    uint ent_A = pair.x;
    uint ent_B = pair.y;

    // TLASLeaf is 48 bytes. bda is at offset 32.
    uint64_t bda_A = BDA_LOAD(uint64_t, pc.tlas_leaves + ent_A * 48 + 32);
    uint64_t bda_B = BDA_LOAD(uint64_t, pc.tlas_leaves + ent_B * 48 + 32);

    // EntityHeader starts with uint ty
    uint type_A = BDA_LOAD(uint, bda_A);
    uint type_B = BDA_LOAD(uint, bda_B);

    if (type_A > type_B) {
        uint temp = ent_A; ent_A = ent_B; ent_B = temp;
        temp = type_A; type_A = type_B; type_B = temp;
    }

    if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_PARTICLE_SYSTEM) {
        if (pc.out_ps_ps.x != 0 || pc.out_ps_ps.y != 0) {
            uint64_t buf_ptr = ((uint64_t)pc.out_ps_ps.y << 32) | pc.out_ps_ps.x;
            uint out_idx = spvAtomicIAdd(buf_ptr, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, 1);
            if (out_idx < pc.max_pairs) {
                BDA_STORE(uint2, buf_ptr + 8 + out_idx * 8, uint2(ent_A, ent_B));
            }
        }
    } else if (type_A == TYPE_RIGID_BODY && type_B == TYPE_PARTICLE_SYSTEM) {
        if (pc.out_rb_ps.x != 0 || pc.out_rb_ps.y != 0) {
            uint64_t buf_ptr = ((uint64_t)pc.out_rb_ps.y << 32) | pc.out_rb_ps.x;
            uint out_idx = spvAtomicIAdd(buf_ptr, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, 1);
            if (out_idx < pc.max_pairs) {
                BDA_STORE(uint2, buf_ptr + 8 + out_idx * 8, uint2(ent_A, ent_B));
            }
        }
    } else if (type_A == TYPE_RIGID_BODY && type_B == TYPE_RIGID_BODY) {
        if (pc.out_rb_rb.x != 0 || pc.out_rb_rb.y != 0) {
            uint64_t buf_ptr = ((uint64_t)pc.out_rb_rb.y << 32) | pc.out_rb_rb.x;
            uint out_idx = spvAtomicIAdd(buf_ptr, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, 1);
            if (out_idx < pc.max_pairs) {
                BDA_STORE(uint2, buf_ptr + 8 + out_idx * 8, uint2(ent_A, ent_B));
            }
        }
    }
}

// --- hlsl_bp_cross_lca.txt ---
// BDA Memory Access Macros
// ------------------------------------------------------------------
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

// ------------------------------------------------------------------
// SPIR-V Atomic Intrinsics for 64-bit BDA Pointers
// ------------------------------------------------------------------
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);

#include "../debug_utils.hlsl"


struct PushConstants {
    uint64_t lca_entities;
    uint64_t macro_leaves;
    uint64_t entity_headers;
    uint64_t lca_query_pairs;
    uint64_t out_rb_rb;
    uint64_t out_rb_ps;
    uint64_t out_ps_ps;
    uint64_t out_cross_pairs;
    uint64_t tlas_bvh;
    uint total_queries;
    uint max_pairs;
};

[[vk::push_constant]] PushConstants pc;

#define SUBGROUP_SIZE 32
#define SUBGROUPS_PER_WG (256 / SUBGROUP_SIZE)
#define AU_TO_KM 149597870.7

groupshared uint shared_stacks[SUBGROUPS_PER_WG][32];
groupshared uint shared_stack_ptrs[SUBGROUPS_PER_WG];
groupshared uint64_t shared_lca_bvh_addr[SUBGROUPS_PER_WG];

void transform_aabb_macro_to_micro(float3 lca_center, float lca_scale, float3 macro_center_au, float3 macro_extents_au, out float3 out_min, out float3 out_max) {
    float3 center_km = macro_center_au * AU_TO_KM;
    float3 extents_km = macro_extents_au * AU_TO_KM;

    float3 corners[8] = {
        float3(center_km.x - extents_km.x, center_km.y - extents_km.y, center_km.z - extents_km.z),
        float3(center_km.x + extents_km.x, center_km.y - extents_km.y, center_km.z - extents_km.z),
        float3(center_km.x - extents_km.x, center_km.y + extents_km.y, center_km.z - extents_km.z),
        float3(center_km.x + extents_km.x, center_km.y + extents_km.y, center_km.z - extents_km.z),
        float3(center_km.x - extents_km.x, center_km.y - extents_km.y, center_km.z + extents_km.z),
        float3(center_km.x + extents_km.x, center_km.y - extents_km.y, center_km.z + extents_km.z),
        float3(center_km.x - extents_km.x, center_km.y + extents_km.y, center_km.z + extents_km.z),
        float3(center_km.x + extents_km.x, center_km.y + extents_km.y, center_km.z + extents_km.z)
    };
    out_min = (float3)1e20; out_max = (float3)-1e20;
    for (int i = 0; i < 8; i++) {
        float3 local_p = (corners[i] - lca_center) / lca_scale;
        out_min = min(out_min, local_p);
        out_max = max(out_max, local_p);
    }
}

[numthreads(256, 1, 1)]
void bp_cross_lca(uint3 gl_WorkGroupID : SV_GroupID, uint gl_SubgroupInvocationID : SV_GroupIndex) {
    uint lane_id = WaveGetLaneIndex();
    uint subgroup_id = gl_SubgroupInvocationID / WaveGetLaneCount();
    uint query_idx = gl_WorkGroupID.x * SUBGROUPS_PER_WG + subgroup_id;

    uint lca_query_pairs_count = BDA_LOAD(uint, pc.lca_query_pairs);
    if (query_idx >= pc.total_queries || query_idx >= lca_query_pairs_count) return;

    uint2 query = BDA_LOAD(uint2, pc.lca_query_pairs + 8 + query_idx * 8);
    uint macro_ent_id = query.x;
    uint lca_ent_id = query.y;
    float3 query_min, query_max;

    if (lane_id == 0) {
        float3 lca_center = BDA_LOAD(float3, pc.lca_entities + lca_ent_id * 80 + 16);
        float lca_scale = BDA_LOAD(float, pc.lca_entities + lca_ent_id * 80 + 28);
        uint lca_bvh_root = BDA_LOAD(uint, pc.lca_entities + lca_ent_id * 80 + 56);
        
        shared_lca_bvh_addr[subgroup_id] = pc.tlas_bvh;
        
        float3 macro_min = BDA_LOAD(float3, pc.macro_leaves + macro_ent_id * 48 + 0);
        float3 macro_max = BDA_LOAD(float3, pc.macro_leaves + macro_ent_id * 48 + 16);

        float3 center_au = (macro_min + macro_max) * 0.5;
        float3 extents_au = (macro_max - macro_min) * 0.5;

        transform_aabb_macro_to_micro(lca_center, lca_scale, center_au, extents_au, query_min, query_max);

        shared_stacks[subgroup_id][0] = lca_bvh_root;
        shared_stack_ptrs[subgroup_id] = 1;
    }

    GroupMemoryBarrierWithGroupSync();

    query_min = WaveReadLaneFirst(query_min);
    query_max = WaveReadLaneFirst(query_max);
    macro_ent_id = WaveReadLaneFirst(macro_ent_id);

    uint64_t tlas_addr = shared_lca_bvh_addr[subgroup_id];

    while (true) {
        GroupMemoryBarrierWithGroupSync();
        uint stack_ptr = shared_stack_ptrs[subgroup_id];
        if (stack_ptr == 0) break;

        stack_ptr--;
        uint node_idx = shared_stacks[subgroup_id][stack_ptr];
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr;

        uint64_t node_addr = tlas_addr + node_idx * 2832;

        uint meta = BDA_LOAD(uint, node_addr + 896 + lane_id * 4);
        uint2 valid_mask = BDA_LOAD(uint2, node_addr + 1792);
        bool valid = bvh_node_is_valid(valid_mask, lane_id);
        
        float3 c_min = float3(
            BDA_LOAD(float, node_addr + 0 + lane_id * 4),
            BDA_LOAD(float, node_addr + 256 + lane_id * 4),
            BDA_LOAD(float, node_addr + 512 + lane_id * 4)
        );
        float3 c_max = float3(
            BDA_LOAD(float, node_addr + 128 + lane_id * 4),
            BDA_LOAD(float, node_addr + 384 + lane_id * 4),
            BDA_LOAD(float, node_addr + 640 + lane_id * 4)
        );
        uint child_payload = BDA_LOAD(uint, node_addr + 768 + lane_id * 4);

        bool hit = valid && intersectAABB(query_min, query_max, c_min, c_max);
        bool is_leaf = bvh_is_leaf(meta);

        bool hit_leaf = hit && is_leaf;
        bool hit_node = hit && !is_leaf;

        uint4 leaf_ballot = WaveActiveBallot(hit_leaf);
        uint leaf_count = countbits(leaf_ballot.x) + countbits(leaf_ballot.y) + countbits(leaf_ballot.z) + countbits(leaf_ballot.w);
        uint leaf_offset = WavePrefixCountBits(hit_leaf);

        if (leaf_count > 0) {
            uint base_idx = 0;
            if (lane_id == 0) {
                base_idx = spvAtomicIAdd(pc.out_cross_pairs, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, leaf_count);
            }
            base_idx = WaveReadLaneFirst(base_idx);

            if (hit_leaf && (base_idx + leaf_offset) < pc.max_pairs) {
                uint64_t pair_addr = pc.out_cross_pairs + 4 + (base_idx + leaf_offset) * 16;
                BDA_STORE(uint, pair_addr + 0, macro_ent_id);
                BDA_STORE(uint, pair_addr + 4, bvh_get_index(meta));
                BDA_STORE(uint, pair_addr + 8, lca_ent_id);
            }
        }

        for (uint i = 0; i < 4; i++) {
            uint m = leaf_ballot[i];
            while (m != 0) {
                uint bit = firstbitlow(m);
                m &= ~(1u << bit);

                uint src_lane = i * 32 + bit;
                uint micro_ent_id = bvh_get_index(WaveReadLaneAt(meta, src_lane));

                if (lane_id == 0) {
                    uint type_A = BDA_LOAD(uint, pc.entity_headers + macro_ent_id * 16);
                    uint type_B = BDA_LOAD(uint, pc.entity_headers + micro_ent_id * 16);
                    uint ent_A = macro_ent_id;
                    uint ent_B = micro_ent_id;

                    if (type_A > type_B) {
                        uint temp = ent_A; ent_A = ent_B; ent_B = temp;
                        temp = type_A; type_A = type_B; type_B = temp;
                    }

                    if (type_A == TYPE_RIGID_BODY && type_B == TYPE_RIGID_BODY) {
                        uint out_idx = spvAtomicIAdd(pc.out_rb_rb, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, 1);
                        if (out_idx < pc.max_pairs) {
                            BDA_STORE(uint2, pc.out_rb_rb + 8 + out_idx * 8, uint2(ent_A, ent_B));
                        }
                    } else if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_RIGID_BODY) {
                        uint out_idx = spvAtomicIAdd(pc.out_rb_ps, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, 1);
                        if (out_idx < pc.max_pairs) {
                            BDA_STORE(uint2, pc.out_rb_ps + 8 + out_idx * 8, uint2(ent_B, ent_A));
                        }
                    } else if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_PARTICLE_SYSTEM) {
                        uint out_idx = spvAtomicIAdd(pc.out_ps_ps, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, 1);
                        if (out_idx < pc.max_pairs) {
                            BDA_STORE(uint2, pc.out_ps_ps + 8 + out_idx * 8, uint2(ent_A, ent_B));
                        }
                    }
                }
            }
        }

        uint4 node_ballot = WaveActiveBallot(hit_node);
        uint node_count = countbits(node_ballot.x) + countbits(node_ballot.y) + countbits(node_ballot.z) + countbits(node_ballot.w);
        uint push_offset = WavePrefixCountBits(hit_node);

        if (hit_node) shared_stacks[subgroup_id][stack_ptr + push_offset] = child_payload;
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr + node_count;
    }
}

// --- hlsl_bp_particle_self.txt ---
// @assets/sim/bp_particle_self.comp
#include "../debug_utils.hlsli"
#include "../bvh_utils.hlsli"

#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(237)]] uint spvAtomicUMin([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(230)]] uint spvAtomicCompareExchange([[vk::ext_reference]] uint64_t ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);

#define SUBGROUP_SIZE 32
#define SUBGROUPS_PER_WG (256 / SUBGROUP_SIZE)

struct PushConstants_bp_particle_self {
    uint64_t bvh;
    uint64_t particles;
    uint64_t wrench_buffer;
    uint root_index;
    uint total_particles;
    float particle_radius;
    float stiffness;
};

[[vk::push_constant]]
PushConstants_bp_particle_self pc;

groupshared uint shared_stacks[SUBGROUPS_PER_WG][32];
groupshared uint shared_stack_ptrs[SUBGROUPS_PER_WG];

// Performs a secure float accumulation via purely standard CAS looping
void atomicAddWrench(uint64_t buf, uint index, float val) {
    uint64_t addr = buf + index * 4;
    uint old_val = BDA_LOAD(uint, addr);
    uint assumed_val;
    do {
        assumed_val = old_val;
        float new_val = asfloat(assumed_val) + val;
        old_val = spvAtomicCompareExchange(addr, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, SPV_SEMANTICS_RELAXED, asuint(new_val), assumed_val);
    } while (assumed_val != old_val);
}

#define MULTI_BVH_NODE_STRIDE 2832
#define MULTI_BVH_MIN_X 0
#define MULTI_BVH_MAX_X 128
#define MULTI_BVH_MIN_Y 256
#define MULTI_BVH_MAX_Y 384
#define MULTI_BVH_MIN_Z 512
#define MULTI_BVH_MAX_Z 640
#define MULTI_BVH_CHILD 768
#define MULTI_BVH_META 896
#define MULTI_BVH_VALID 1792

#ifdef KERNEL_bp_particle_self
[numthreads(256, 1, 1)]
void bp_particle_self(uint3 DispatchThreadID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID, uint3 GroupThreadID : SV_GroupThreadID) {
    uint subgroup_id = GroupThreadID.x / SUBGROUP_SIZE;
    uint lane_id     = WaveGetLaneIndex();

    uint my_p_idx = GroupID.x * SUBGROUPS_PER_WG + subgroup_id;
    if (my_p_idx >= pc.total_particles) return;

    float3 my_pos, my_min, my_max;
    float my_radius = pc.particle_radius;

    if (lane_id == 0) {
        uint block_idx = my_p_idx / SUBGROUP_SIZE;
        uint local_idx = my_p_idx % SUBGROUP_SIZE;
        uint base = block_idx * (10u * SUBGROUP_SIZE) + local_idx;

        my_pos = float3(
            asfloat(BDA_LOAD(uint, pc.particles + (base + 0u * SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (base + 1u * SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (base + 2u * SUBGROUP_SIZE) * 4))
        );

        my_min = my_pos - float3(my_radius, my_radius, my_radius);
        my_max = my_pos + float3(my_radius, my_radius, my_radius);

        shared_stacks[subgroup_id][0] = pc.root_index;
        shared_stack_ptrs[subgroup_id] = 1;
    }

    my_pos = float3(WaveReadLaneAt(my_pos.x, 0), WaveReadLaneAt(my_pos.y, 0), WaveReadLaneAt(my_pos.z, 0));
    my_min = float3(WaveReadLaneAt(my_min.x, 0), WaveReadLaneAt(my_min.y, 0), WaveReadLaneAt(my_min.z, 0));
    my_max = float3(WaveReadLaneAt(my_max.x, 0), WaveReadLaneAt(my_max.y, 0), WaveReadLaneAt(my_max.z, 0));
    my_p_idx = WaveReadLaneAt(my_p_idx, 0);

    float3 local_repulsive_force = float3(0.0, 0.0, 0.0);

    while (true) {
        GroupMemoryBarrierWithGroupSync();
        uint stack_ptr = shared_stack_ptrs[subgroup_id];
        if (stack_ptr == 0) break;

        stack_ptr--;
        uint node_idx = shared_stacks[subgroup_id][stack_ptr];
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr;

        uint64_t node_addr = pc.bvh + node_idx * MULTI_BVH_NODE_STRIDE;

        uint meta = BDA_LOAD(uint, node_addr + MULTI_BVH_META + lane_id * 4);
        uint2 valid_mask = BDA_LOAD(uint2, node_addr + MULTI_BVH_VALID);
        bool valid = bvh_node_is_valid(valid_mask, lane_id);

        float3 c_min = float3(
            BDA_LOAD(float, node_addr + MULTI_BVH_MIN_X + lane_id * 4),
            BDA_LOAD(float, node_addr + MULTI_BVH_MIN_Y + lane_id * 4),
            BDA_LOAD(float, node_addr + MULTI_BVH_MIN_Z + lane_id * 4)
        );
        float3 c_max = float3(
            BDA_LOAD(float, node_addr + MULTI_BVH_MAX_X + lane_id * 4),
            BDA_LOAD(float, node_addr + MULTI_BVH_MAX_Y + lane_id * 4),
            BDA_LOAD(float, node_addr + MULTI_BVH_MAX_Z + lane_id * 4)
        );
        uint child_payload = BDA_LOAD(uint, node_addr + MULTI_BVH_CHILD + lane_id * 4);

        bool hit_aabb = valid && intersectAABB(my_min, my_max, c_min, c_max);
        bool is_leaf = bvh_is_leaf(meta);

        bool hit_node = hit_aabb && !is_leaf;
        bool hit_leaf = hit_aabb && is_leaf && (my_p_idx != child_payload);

        uint4 leaf_ballot = WaveActiveBallot(hit_leaf);

        for (uint i = 0; i < 4; i++) {
            uint mask = leaf_ballot[i];
            while (mask != 0) {
                uint bit = firstbitlow(mask);
                mask &= ~(1u << bit);

                uint laneIndex = i * 32 + bit;
                uint other_idx = WaveReadLaneAt(child_payload, laneIndex);

                uint block_idx = other_idx / SUBGROUP_SIZE;
                uint local_idx = other_idx % SUBGROUP_SIZE;
                uint base_idx = block_idx * (10u * SUBGROUP_SIZE) + local_idx;

                float3 other_pos = float3(
                    asfloat(BDA_LOAD(uint, pc.particles + (base_idx + 0u * SUBGROUP_SIZE) * 4)),
                    asfloat(BDA_LOAD(uint, pc.particles + (base_idx + 1u * SUBGROUP_SIZE) * 4)),
                    asfloat(BDA_LOAD(uint, pc.particles + (base_idx + 2u * SUBGROUP_SIZE) * 4))
                );

                float3 diff = my_pos - other_pos;
                float dist_sq = dot(diff, diff);
                float min_dist = my_radius * 2.0;

                if (dist_sq > 1e-12 && dist_sq < min_dist * min_dist) {
                    float dist = sqrt(dist_sq);
                    float penetration = min_dist - dist;
                    float3 normal = diff / dist;

                    float force_mag = pc.stiffness * penetration;
                    local_repulsive_force += normal * force_mag;
                }
            }
        }

        uint4 node_ballot = WaveActiveBallot(hit_node);
        uint node_count = countbits(node_ballot.x) + countbits(node_ballot.y) + countbits(node_ballot.z) + countbits(node_ballot.w);
        
        uint push_offset = 0;
        if (lane_id >= 32) push_offset += countbits(node_ballot.x);
        if (lane_id >= 64) push_offset += countbits(node_ballot.y);
        if (lane_id >= 96) push_offset += countbits(node_ballot.z);
        uint shift = lane_id % 32;
        uint current_word = (lane_id < 32) ? node_ballot.x : (lane_id < 64) ? node_ballot.y : (lane_id < 96) ? node_ballot.z : node_ballot.w;
        uint mask = (1u << shift) - 1u;
        push_offset += countbits(current_word & mask);

        if (hit_node) {
            shared_stacks[subgroup_id][stack_ptr + push_offset] = child_payload;
        }
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr + node_count;
    }

    local_repulsive_force.x = WaveActiveSum(local_repulsive_force.x);
    local_repulsive_force.y = WaveActiveSum(local_repulsive_force.y);
    local_repulsive_force.z = WaveActiveSum(local_repulsive_force.z);

    if (lane_id == 0 && dot(local_repulsive_force, local_repulsive_force) > 0.0) {
        atomicAddWrench(pc.wrench_buffer, my_p_idx * 6 + 0, local_repulsive_force.x);
        atomicAddWrench(pc.wrench_buffer, my_p_idx * 6 + 1, local_repulsive_force.y);
        atomicAddWrench(pc.wrench_buffer, my_p_idx * 6 + 2, local_repulsive_force.z);
    }
}
#endif


// --- hlsl_ccd.txt ---
[numthreads(128, 1, 1)]


#include "gjk_cta_utils.glsl"

struct PushConstants_ccd {
    uint64_t particle_bvh;
    uint64_t output_list;
    uint64_t particles;
    uint root_index;
    uint total_particles;
    float particle_radius;
    float dt;
};

#ifdef KERNEL_ccd
[[vk::push_constant]]
PushConstants_ccd pc;
#endif

void ccd(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint idx = DispatchThreadID.x; 
    if (idx >= pc.total_particles) return;

    uint my_prim_id = idx;
    uint baseA = (my_prim_id / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (my_prim_id % SUBGROUP_SIZE);
    
    float3 my_center = float3(
        BDA_LOAD(float, pc.particles + (baseA + 0) * 4),
        BDA_LOAD(float, pc.particles + (baseA + 1 * SUBGROUP_SIZE) * 4),
        BDA_LOAD(float, pc.particles + (baseA + 2 * SUBGROUP_SIZE) * 4)
    );
    
    float3 my_vel = float3(
        BDA_LOAD(float, pc.particles + (baseA + 3 * SUBGROUP_SIZE) * 4),
        BDA_LOAD(float, pc.particles + (baseA + 4 * SUBGROUP_SIZE) * 4),
        BDA_LOAD(float, pc.particles + (baseA + 5 * SUBGROUP_SIZE) * 4)
    );
    
    float3 p1 = my_center + my_vel * pc.dt;

    AABB my_aabb;
    my_aabb.minBounds = min(my_center - float3(pc.particle_radius, pc.particle_radius, pc.particle_radius), p1 - float3(pc.particle_radius, pc.particle_radius, pc.particle_radius));
    my_aabb.maxBounds = max(my_center + float3(pc.particle_radius, pc.particle_radius, pc.particle_radius), p1 + float3(pc.particle_radius, pc.particle_radius, pc.particle_radius));

    uint stack[64]; 
    int stackPtr = 0; 
    if (pc.root_index != 0xFFFFFFFFu) stack[stackPtr++] = pc.root_index;
    
    uint collisions_found = 0;

    while (stackPtr > 0) {
        uint node_idx = stack[--stackPtr];
        uint64_t node_addr = pc.particle_bvh + node_idx * 2832;

        uint2 valid_mask = BDA_LOAD(uint2, node_addr + 1792);

        for (uint i = 0; i < SUBGROUP_SIZE; ++i) {
            if (!bvh_node_is_valid(valid_mask, i)) continue;

            AABB bound;
            bound.minBounds = float3(
                BDA_LOAD(float, node_addr + 0 + i * 4),
                BDA_LOAD(float, node_addr + 256 + i * 4),
                BDA_LOAD(float, node_addr + 512 + i * 4)
            );
            bound.maxBounds = float3(
                BDA_LOAD(float, node_addr + 128 + i * 4),
                BDA_LOAD(float, node_addr + 384 + i * 4),
                BDA_LOAD(float, node_addr + 640 + i * 4)
            );

            if (intersectAABB(my_aabb, bound)) {
                uint meta = BDA_LOAD(uint, node_addr + 896 + i * 4);
                uint offset = bvh_get_index(meta);

                if (bvh_is_leaf(meta)) {
                    if (my_prim_id < offset) {
                        float toi = 0.0, depth = 0.0; 
                        float3 normal = float3(0.0, 0.0, 0.0);
                        float3 point = float3(0.0, 0.0, 0.0);
                        
                        uint baseB = (offset / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (offset % SUBGROUP_SIZE);
                        float3 other_vel = float3(
                            BDA_LOAD(float, pc.particles + (baseB + 3 * SUBGROUP_SIZE) * 4),
                            BDA_LOAD(float, pc.particles + (baseB + 4 * SUBGROUP_SIZE) * 4),
                            BDA_LOAD(float, pc.particles + (baseB + 5 * SUBGROUP_SIZE) * 4)
                        ) * pc.dt;
                        
                        float4x4 transA = float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
                        transA[3].xyz = my_center;
                        
                        float4x4 transB = float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
                        transB[3].xyz = float3(
                            BDA_LOAD(float, node_addr + 1152 + i * 4),
                            BDA_LOAD(float, node_addr + 1280 + i * 4),
                            BDA_LOAD(float, node_addr + 1408 + i * 4)
                        );

                        if (compute_toi_generic(0, float3(pc.particle_radius, 0.0, 0.0), transA, my_vel * pc.dt, 0, float3(pc.particle_radius, 0.0, 0.0), transB, other_vel, 1e-3, 10, toi, normal, point, depth)) {
                            if (collisions_found < 16) {
                                uint outIdx = idx * 16 + collisions_found++;
                                uint64_t out_addr = pc.output_list + 16 + outIdx * 96;
                                
                                BDA_STORE(uint, out_addr + 0, 1);
                                BDA_STORE(uint, out_addr + 8, my_prim_id);
                                BDA_STORE(uint, out_addr + 16, offset);
                                BDA_STORE(float, out_addr + 48, toi);
                                BDA_STORE(float, out_addr + 52, depth);
                                BDA_STORE(float4, out_addr + 64, float4(normal, 0.0));
                                BDA_STORE(float4, out_addr + 80, float4(point, 1.0));
                            }
                        }
                    }
                } else if (offset != 0xFFFFFFFFu) {
                    stack[stackPtr++] = offset;
                }
            }
        }
    }
}

// --- hlsl_narrow_ccd.txt ---
// BDA Memory Access Macros
// ------------------------------------------------------------------
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

// ------------------------------------------------------------------
// SPIR-V Atomic Intrinsics for 64-bit BDA Pointers
// ------------------------------------------------------------------
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);


#include "imex_math.glsl"


struct PushConstants {
    uint64_t scene_entities;
    uint64_t output_list;
    uint64_t cross_output_list;
    uint64_t particles;
    uint64_t pair_buffer;
    uint64_t cross_pair_buffer;
    uint64_t lca_entities;
    float dt;
    float particle_radius;
    uint space_type;
};

[[vk::push_constant]]
PushConstants pc;

[numthreads(256, 1, 1)]
void narrow_ccd(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint pair_idx = DispatchThreadID.x;
    
    uint idA, idB, lca_id;
    bool is_partA = false, is_partB = false;

    if (pc.space_type == 1) { // Cross
        uint cross_pairs_count = BDA_LOAD(uint, pc.cross_pair_buffer);
        if (pair_idx >= cross_pairs_count) return;
        CrossPair pair = BDA_LOAD(CrossPair, pc.cross_pair_buffer + 16 + pair_idx * sizeof(CrossPair));
        idA = pair.macro_id;
        idB = pair.micro_id;
        lca_id = pair.lca_id;
    } else { // Standard
        uint pair_buffer_count = BDA_LOAD(uint, pc.pair_buffer);
        if (pair_idx >= pair_buffer_count) return;
        uint2 pair = BDA_LOAD(uint2, pc.pair_buffer + 8 + pair_idx * sizeof(uint2));
        idA = pair.x;
        idB = pair.y;
    }

    float3 pos_A, vel_A, extents_A;
    uint shape_A;
    float4 orient_A = float4(0, 0, 0, 1);

    if (idA == 0xFFFFFFFFu) { 
        is_partA = true;
    }
    
    RigidBody ent_A = BDA_LOAD(RigidBody, pc.scene_entities + idA * 128u);
    RigidBody ent_B = BDA_LOAD(RigidBody, pc.scene_entities + idB * 128u);
    
    shape_A = ent_A.shape_type;
    extents_A = ent_A.shape_extents;
    orient_A = ent_A.orientation;
    pos_A = ent_A.position_mass.xyz;
    vel_A = ent_A.linear_vel_drag.xyz;
    
    uint shape_B = ent_B.shape_type;
    float3 extents_B = ent_B.shape_extents;
    float4 orient_B = ent_B.orientation;
    float3 pos_B = ent_B.position_mass.xyz;
    float3 vel_B = ent_B.linear_vel_drag.xyz;

    float4x4 trans_A = float4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
    float4x4 trans_B = float4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
    
    if (pc.space_type == 1) {
        LcaEntity lca = BDA_LOAD(LcaEntity, pc.lca_entities + lca_id * sizeof(LcaEntity));
        float3 macro_rel_vel_au = vel_A - lca.linear_velocity;
        pos_A = mul(lca.inv_transform, float4(pos_A, 1.0)).xyz * AU_TO_KM;
        float3x3 lca_inv_trans_3x3 = (float3x3)lca.inv_transform;
        vel_A = mul(lca_inv_trans_3x3, macro_rel_vel_au) * AU_TO_KM;
        extents_A *= AU_TO_KM;
        trans_A = lca.inv_transform; 
    }
    
    float3x3 rotA = quat_to_mat3(orient_A);
    trans_A = float4x4(
        rotA[0][0], rotA[0][1], rotA[0][2], pos_A.x,
        rotA[1][0], rotA[1][1], rotA[1][2], pos_A.y,
        rotA[2][0], rotA[2][1], rotA[2][2], pos_A.z,
        0,          0,          0,          1
    );
    
    float3x3 rotB = quat_to_mat3(orient_B);
    trans_B = float4x4(
        rotB[0][0], rotB[0][1], rotB[0][2], pos_B.x,
        rotB[1][0], rotB[1][1], rotB[1][2], pos_B.y,
        rotB[2][0], rotB[2][1], rotB[2][2], pos_B.z,
        0,          0,          0,          1
    );

    float toi, depth;
    float3 normal, contact;
    
    if (compute_toi_generic(shape_A, extents_A, trans_A, vel_A, shape_B, extents_B, trans_B, vel_B, 1e-3, 10, toi, normal, contact, depth)) {
        if (pc.space_type == 1) {
            uint count = spvAtomicIAdd(pc.cross_output_list, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, 1);
            if (count < 4000u) {
                CrossPair out_pair;
                out_pair.valid = 1u;
                out_pair.macro_id = idA;
                out_pair.micro_id = idB;
                out_pair.lca_id = lca_id;
                out_pair.toi = toi;
                out_pair.contact_normal = float4(normal, 0.0);
                out_pair.contact_point = float4(contact, 1.0);
                out_pair.penetration_depth = depth;
                BDA_STORE(CrossPair, pc.cross_output_list + 16 + count * sizeof(CrossPair), out_pair);
            }
        } else {
            uint count = spvAtomicIAdd(pc.output_list, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, 1);
            if (count < 4000u) {
                SparseCollisionPair out_pair;
                out_pair.entity_a = idA;
                out_pair.prim_a = idA;
                out_pair.entity_b = idB;
                out_pair.prim_b = idB;
                out_pair.toi = toi;
                out_pair.contact_normal = float4(normal, 0.0);
                out_pair.contact_point = float4(contact, 1.0);
                out_pair.penetration_depth = depth;
                out_pair.bda_a = pc.scene_entities + idA * 128u;
                out_pair.bda_b = pc.scene_entities + idB * 128u;
                out_pair.frame_bda = 0; 
                out_pair.valid = 1u;
                BDA_STORE(SparseCollisionPair, pc.output_list + 16 + count * sizeof(SparseCollisionPair), out_pair);
            }
        }
    }
}


// --- hlsl_lcp_solver.txt ---



#define MAX_BODIES_PER_ISLAND 32
#define SUBGROUP_SIZE 32

struct PushConstants {
    uint64_t particles;
    uint64_t collisions;
    uint64_t outputs;
    uint total_clusters;
    uint64_t rigid_bodies;
    float dt;
    float restitution;
};

[[vk::push_constant]]
PushConstants pc;

groupshared uint shared_v_x[MAX_BODIES_PER_ISLAND];
groupshared uint shared_v_y[MAX_BODIES_PER_ISLAND];
groupshared uint shared_v_z[MAX_BODIES_PER_ISLAND];
groupshared uint shared_w_x[MAX_BODIES_PER_ISLAND];
groupshared uint shared_w_y[MAX_BODIES_PER_ISLAND];
groupshared uint shared_w_z[MAX_BODIES_PER_ISLAND];
groupshared float accumulated_normal[128];
groupshared float accumulated_t1[128];
groupshared float accumulated_t2[128];

void generate_tangents(float3 normal, out float3 t1, out float3 t2) {
    if (abs(normal.x) >= 0.57735) {
        t1 = normalize(float3(normal.y, -normal.x, 0.0));
    } else {
        t1 = normalize(float3(0.0, normal.z, -normal.y));
    }
    t2 = cross(normal, t1);
}

float compute_effective_mass(float3 dir, float3 rA, float3 rB, float invMA, float invMB, float3 invIA, float3 invIB, float4 qA, float4 qB) {
    float3 I_crossA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, dir)));
    float3 I_crossB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, dir)));
    return 1.0 / max(invMA + invMB + dot(I_crossA, cross(rA, dir)) + dot(I_crossB, cross(rB, dir)), 1e-6);
}

void AtomicAddFloatShared(inout uint dest, float val) {
    uint old_val = dest;
    uint assumed_val;
    uint new_val;
    [allow_uav_condition]
    while (true) {
        assumed_val = old_val;
        new_val = asuint(asfloat(assumed_val) + val);
        InterlockedCompareExchange(dest, assumed_val, new_val, old_val);
        if (assumed_val == old_val) break;
    }
}

[numthreads(128, 1, 1)]
void lcp_solver(uint3 DispatchThreadID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID, uint GroupIndex : SV_GroupIndex) {
    uint local_id = GroupIndex;
    uint contact_idx = DispatchThreadID.x;
    
    uint collisions_count = BDA_LOAD(uint, pc.collisions + 12);
    bool valid = (contact_idx < collisions_count);

    accumulated_normal[local_id] = 0.0;
    accumulated_t1[local_id] = 0.0;
    accumulated_t2[local_id] = 0.0;

    if (local_id < MAX_BODIES_PER_ISLAND) {
        uint64_t rb_addr = pc.rigid_bodies + local_id * 128;
        float4 linear_vel_drag = BDA_LOAD(float4, rb_addr + 48);
        float4 angular_vel_drag = BDA_LOAD(float4, rb_addr + 64);
        shared_v_x[local_id] = asuint(linear_vel_drag.x);
        shared_v_y[local_id] = asuint(linear_vel_drag.y);
        shared_v_z[local_id] = asuint(linear_vel_drag.z);
        shared_w_x[local_id] = asuint(angular_vel_drag.x);
        shared_w_y[local_id] = asuint(angular_vel_drag.y);
        shared_w_z[local_id] = asuint(angular_vel_drag.z);
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (!valid) return;

    uint64_t pair_addr = pc.collisions + 16 + contact_idx * 80;
    bool is_partA = (BDA_LOAD(uint, pair_addr) == 0xFFFFFFFFu);
    bool is_partB = (BDA_LOAD(uint, pair_addr + 8) == 0xFFFFFFFFu);
    uint idA = BDA_LOAD(uint, pair_addr + 4);
    uint idB = BDA_LOAD(uint, pair_addr + 12);

    float invMA = 0.0;
    float invMB = 0.0;
    float3 invIA = float3(0.0, 0.0, 0.0);
    float3 invIB = float3(0.0, 0.0, 0.0);
    float4 qA = float4(0, 0, 0, 1);
    float4 qB = float4(0, 0, 0, 1);
    float3 posA = float3(0.0, 0.0, 0.0);
    float3 posB = float3(0.0, 0.0, 0.0);
    float3 vA_init = float3(0.0, 0.0, 0.0);
    float3 wA_init = float3(0.0, 0.0, 0.0);
    float3 vB_init = float3(0.0, 0.0, 0.0);
    float3 wB_init = float3(0.0, 0.0, 0.0);

    if (is_partA) {
        uint baseA = (idA / SUBGROUP_SIZE) * 10u * SUBGROUP_SIZE + (idA % SUBGROUP_SIZE);
        posA = float3(
            asfloat(BDA_LOAD(uint, pc.particles + (baseA) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (baseA + SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (baseA + 2 * SUBGROUP_SIZE) * 4))
        );
        vA_init = float3(
            asfloat(BDA_LOAD(uint, pc.particles + (baseA + 3 * SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (baseA + 4 * SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (baseA + 5 * SUBGROUP_SIZE) * 4))
        );
        float mass = asfloat(BDA_LOAD(uint, pc.particles + (baseA + 6u * SUBGROUP_SIZE) * 4));
        invMA = (mass > 0.0) ? 1.0 / mass : 0.0;
    } else {
        uint64_t rbA_addr = pc.rigid_bodies + idA * 128;
        float4 position_mass = BDA_LOAD(float4, rbA_addr + 16);
        invMA = position_mass.w > 0.0 ? 1.0 / position_mass.w : 0.0;
        posA = position_mass.xyz;
        qA = BDA_LOAD(float4, rbA_addr + 32);
        vA_init = BDA_LOAD(float4, rbA_addr + 48).xyz;
        wA_init = BDA_LOAD(float4, rbA_addr + 64).xyz;
        invIA = BDA_LOAD(float4, rbA_addr + 80).xyz;
    }

    if (is_partB) {
        uint baseB = (idB / SUBGROUP_SIZE) * 10u * SUBGROUP_SIZE + (idB % SUBGROUP_SIZE);
        posB = float3(
            asfloat(BDA_LOAD(uint, pc.particles + (baseB) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (baseB + SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (baseB + 2 * SUBGROUP_SIZE) * 4))
        );
        vB_init = float3(
            asfloat(BDA_LOAD(uint, pc.particles + (baseB + 3 * SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (baseB + 4 * SUBGROUP_SIZE) * 4)),
            asfloat(BDA_LOAD(uint, pc.particles + (baseB + 5 * SUBGROUP_SIZE) * 4))
        );
        float mass = asfloat(BDA_LOAD(uint, pc.particles + (baseB + 6u * SUBGROUP_SIZE) * 4));
        invMB = (mass > 0.0) ? 1.0 / mass : 0.0;
    } else {
        uint64_t rbB_addr = pc.rigid_bodies + idB * 128;
        float4 position_mass = BDA_LOAD(float4, rbB_addr + 16);
        invMB = position_mass.w > 0.0 ? 1.0 / position_mass.w : 0.0;
        posB = position_mass.xyz;
        qB = BDA_LOAD(float4, rbB_addr + 32);
        vB_init = BDA_LOAD(float4, rbB_addr + 48).xyz;
        wB_init = BDA_LOAD(float4, rbB_addr + 64).xyz;
        invIB = BDA_LOAD(float4, rbB_addr + 80).xyz;
    }

    float3 n = BDA_LOAD(float4, pair_addr + 32).xyz;
    float3 contact_point = BDA_LOAD(float4, pair_addr + 48).xyz;
    float penetration_depth = BDA_LOAD(float, pair_addr + 64);
    
    float3 t1, t2;
    generate_tangents(n, t1, t2);
    float3 rA = contact_point - posA;
    float3 rB = contact_point - posB;
    
    float eff_m_n = compute_effective_mass(n, rA, rB, invMA, invMB, invIA, invIB, qA, qB);
    float eff_m_t1 = compute_effective_mass(t1, rA, rB, invMA, invMB, invIA, invIB, qA, qB);
    float eff_m_t2 = compute_effective_mass(t2, rA, rB, invMA, invMB, invIA, invIB, qA, qB);

    float3 v_rel_init = (vB_init + cross(wB_init, rB)) - (vA_init + cross(wA_init, rA));
    float bounce = dot(v_rel_init, n) < -0.1 ? -pc.restitution * dot(v_rel_init, n) : 0.0;
    float target_v_n = bounce + ((0.2 / max(pc.dt, 1e-6)) * max(penetration_depth - 0.01, 0.0));

    for (int iter = 0; iter < 20; ++iter) {
        GroupMemoryBarrierWithGroupSync();

        float3 vA = vA_init;
        float3 wA = wA_init;
        if (!is_partA && idA < MAX_BODIES_PER_ISLAND) {
            vA = float3(asfloat(shared_v_x[idA]), asfloat(shared_v_y[idA]), asfloat(shared_v_z[idA]));
            wA = float3(asfloat(shared_w_x[idA]), asfloat(shared_w_y[idA]), asfloat(shared_w_z[idA]));
        }

        float3 vB = vB_init;
        float3 wB = wB_init;
        if (!is_partB && idB < MAX_BODIES_PER_ISLAND) {
            vB = float3(asfloat(shared_v_x[idB]), asfloat(shared_v_y[idB]), asfloat(shared_v_z[idB]));
            wB = float3(asfloat(shared_w_x[idB]), asfloat(shared_w_y[idB]), asfloat(shared_w_z[idB]));
        }

        float3 v_rel = (vB + cross(wB, rB)) - (vA + cross(wA, rA));
        float jn_delta = eff_m_n * (-dot(v_rel, n) + target_v_n);
        float old_jn = accumulated_normal[local_id];
        float new_jn = max(old_jn + jn_delta, 0.0);
        jn_delta = new_jn - old_jn;
        accumulated_normal[local_id] = new_jn;
        float3 P_n = jn_delta * n;

        if (!is_partA && invMA > 0.0 && idA < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idA], -P_n.x * invMA);
            AtomicAddFloatShared(shared_v_y[idA], -P_n.y * invMA);
            AtomicAddFloatShared(shared_v_z[idA], -P_n.z * invMA);
            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -P_n)));
            AtomicAddFloatShared(shared_w_x[idA], dwA.x);
            AtomicAddFloatShared(shared_w_y[idA], dwA.y);
            AtomicAddFloatShared(shared_w_z[idA], dwA.z);
        }
        if (!is_partB && invMB > 0.0 && idB < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idB], P_n.x * invMB);
            AtomicAddFloatShared(shared_v_y[idB], P_n.y * invMB);
            AtomicAddFloatShared(shared_v_z[idB], P_n.z * invMB);
            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, P_n)));
            AtomicAddFloatShared(shared_w_x[idB], dwB.x);
            AtomicAddFloatShared(shared_w_y[idB], dwB.y);
            AtomicAddFloatShared(shared_w_z[idB], dwB.z);
        }

        GroupMemoryBarrierWithGroupSync();

        if (!is_partA && idA < MAX_BODIES_PER_ISLAND) {
            vA = float3(asfloat(shared_v_x[idA]), asfloat(shared_v_y[idA]), asfloat(shared_v_z[idA]));
            wA = float3(asfloat(shared_w_x[idA]), asfloat(shared_w_y[idA]), asfloat(shared_w_z[idA]));
        }
        if (!is_partB && idB < MAX_BODIES_PER_ISLAND) {
            vB = float3(asfloat(shared_v_x[idB]), asfloat(shared_v_y[idB]), asfloat(shared_v_z[idB]));
            wB = float3(asfloat(shared_w_x[idB]), asfloat(shared_w_y[idB]), asfloat(shared_w_z[idB]));
        }
        v_rel = (vB + cross(wB, rB)) - (vA + cross(wA, rA));

        float max_fric = 0.5 * accumulated_normal[local_id];
        float jt1_delta = eff_m_t1 * (-dot(v_rel, t1));
        float old_jt1 = accumulated_t1[local_id];
        float new_jt1 = clamp(old_jt1 + jt1_delta, -max_fric, max_fric);
        jt1_delta = new_jt1 - old_jt1;
        accumulated_t1[local_id] = new_jt1;

        float jt2_delta = eff_m_t2 * (-dot(v_rel, t2));
        float old_jt2 = accumulated_t2[local_id];
        float new_jt2 = clamp(old_jt2 + jt2_delta, -max_fric, max_fric);
        jt2_delta = new_jt2 - old_jt2;
        accumulated_t2[local_id] = new_jt2;

        float3 P_t = jt1_delta * t1 + jt2_delta * t2;

        if (!is_partA && invMA > 0.0 && idA < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idA], -P_t.x * invMA);
            AtomicAddFloatShared(shared_v_y[idA], -P_t.y * invMA);
            AtomicAddFloatShared(shared_v_z[idA], -P_t.z * invMA);
            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -P_t)));
            AtomicAddFloatShared(shared_w_x[idA], dwA.x);
            AtomicAddFloatShared(shared_w_y[idA], dwA.y);
            AtomicAddFloatShared(shared_w_z[idA], dwA.z);
        }
        if (!is_partB && invMB > 0.0 && idB < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idB], P_t.x * invMB);
            AtomicAddFloatShared(shared_v_y[idB], P_t.y * invMB);
            AtomicAddFloatShared(shared_v_z[idB], P_t.z * invMB);
            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, P_t)));
            AtomicAddFloatShared(shared_w_x[idB], dwB.x);
            AtomicAddFloatShared(shared_w_y[idB], dwB.y);
            AtomicAddFloatShared(shared_w_z[idB], dwB.z);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    
    float3 final_impulse = accumulated_normal[local_id] * n + accumulated_t1[local_id] * t1 + accumulated_t2[local_id] * t2;
    BDA_STORE(float3, pc.outputs + contact_idx * 16, final_impulse);
}

// --- hlsl_apply_impulses.txt ---
// BDA Memory Access Macros
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

[[vk::ext_instruction(230)]] uint spvAtomicCompareExchange([[vk::ext_reference]] uint64_t ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);

void AtomicAddFloat(uint64_t ptr, float val) {
    uint old_val = BDA_LOAD(uint, ptr);
    uint assumed_val;
    do {
        assumed_val = old_val;
        uint new_val = asuint(asfloat(assumed_val) + val);
        old_val = spvAtomicCompareExchange(ptr, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, SPV_SEMANTICS_RELAXED, new_val, assumed_val);
    } while (assumed_val != old_val);
}

float4 quat_conj(float4 q) { return float4(-q.xyz, q.w); }
float3 quat_rotate(float4 q, float3 v) {
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
float3 quat_rotate_inv(float4 q, float3 v) { return quat_rotate(quat_conj(q), v); }

struct ColliderId {
    uint entity_id;
    uint primitive_index;
};

struct PackedPair {
    ColliderId a;
    ColliderId b;
    float toi;
    uint pad0;
    uint pad1;
    uint pad2;
    float3 contact_normal;
    float pad_normal;
    float3 contact_point;
    float pad_point;
    float penetration_depth;
    uint pad3;
    uint pad4;
    uint pad5;
};

struct PushConstants {
    uint64_t particles;
    uint64_t collisions;
    uint64_t impulses;
    uint64_t rigid_bodies;
};

[[vk::push_constant]]
PushConstants pc;

static const uint SUBGROUP_SIZE = 32;

[numthreads(128, 1, 1)]
void apply_impulses(uint3 tid : SV_DispatchThreadID) {
    uint global_id = tid.x;
    
    // collisions struct: uint dispatch_x, y, z, count; PackedPair pairs[];
    uint count = BDA_LOAD(uint, pc.collisions + 12);
    if (global_id >= count) return;
    
    // Size of PackedPair is 80 bytes
    uint64_t pair_addr = pc.collisions + 16 + global_id * 80;
    
    ColliderId pair_a;
    pair_a.entity_id = BDA_LOAD(uint, pair_addr);
    pair_a.primitive_index = BDA_LOAD(uint, pair_addr + 4);
    
    ColliderId pair_b;
    pair_b.entity_id = BDA_LOAD(uint, pair_addr + 8);
    pair_b.primitive_index = BDA_LOAD(uint, pair_addr + 12);
    
    float3 contact_point = BDA_LOAD(float3, pair_addr + 48);
    
    float3 impulse = BDA_LOAD(float3, pc.impulses + global_id * 16);
    if (length(impulse) < 1e-6) return;
    
    uint pA_id = pair_a.primitive_index;
    uint pB_id = pair_b.primitive_index;
    
    bool is_rb_a = (pair_a.entity_id != 0xFFFFFFFFu);
    bool is_rb_b = (pair_b.entity_id != 0xFFFFFFFFu);
    
    if (is_rb_a) {
        uint base = pA_id * 28;
        float mass = asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 3) * 4));
        float invMA = mass > 0.0 ? 1.0 / mass : 0.0;
        
        float3 invIA = float3(
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 16) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 17) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 18) * 4))
        );
        float4 qA = float4(
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 4) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 5) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 6) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 7) * 4))
        );
        float3 posA = float3(
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 0) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 1) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 2) * 4))
        );
        
        float3 rA = contact_point - posA;
        
        if (invMA > 0.0) {
            float3 dvA = -impulse * invMA;
            AtomicAddFloat(pc.rigid_bodies + (base + 8) * 4, dvA.x);
            AtomicAddFloat(pc.rigid_bodies + (base + 9) * 4, dvA.y);
            AtomicAddFloat(pc.rigid_bodies + (base + 10) * 4, dvA.z);
            
            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -impulse)));
            AtomicAddFloat(pc.rigid_bodies + (base + 12) * 4, dwA.x);
            AtomicAddFloat(pc.rigid_bodies + (base + 13) * 4, dwA.y);
            AtomicAddFloat(pc.rigid_bodies + (base + 14) * 4, dwA.z);
        }
    } else {
        uint base = (pA_id / SUBGROUP_SIZE) * (10u * SUBGROUP_SIZE) + (pA_id % SUBGROUP_SIZE);
        float mass = asfloat(BDA_LOAD(uint, pc.particles + (base + 6u * SUBGROUP_SIZE) * 4));
        float invMA = mass > 0.0 ? 1.0 / mass : 0.0;
        if (invMA > 0.0) {
            float3 dvA = -impulse * invMA;
            AtomicAddFloat(pc.particles + (base + 3u * SUBGROUP_SIZE) * 4, dvA.x);
            AtomicAddFloat(pc.particles + (base + 4u * SUBGROUP_SIZE) * 4, dvA.y);
            AtomicAddFloat(pc.particles + (base + 5u * SUBGROUP_SIZE) * 4, dvA.z);
        }
    }
    
    if (is_rb_b) {
        uint base = pB_id * 28;
        float mass = asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 3) * 4));
        float invMB = mass > 0.0 ? 1.0 / mass : 0.0;
        
        float3 invIB = float3(
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 16) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 17) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 18) * 4))
        );
        float4 qB = float4(
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 4) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 5) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 6) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 7) * 4))
        );
        float3 posB = float3(
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 0) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 1) * 4)),
            asfloat(BDA_LOAD(uint, pc.rigid_bodies + (base + 2) * 4))
        );
        
        float3 rB = contact_point - posB;
        
        if (invMB > 0.0) {
            float3 dvB = impulse * invMB;
            AtomicAddFloat(pc.rigid_bodies + (base + 8) * 4, dvB.x);
            AtomicAddFloat(pc.rigid_bodies + (base + 9) * 4, dvB.y);
            AtomicAddFloat(pc.rigid_bodies + (base + 10) * 4, dvB.z);
            
            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, impulse)));
            AtomicAddFloat(pc.rigid_bodies + (base + 12) * 4, dwB.x);
            AtomicAddFloat(pc.rigid_bodies + (base + 13) * 4, dwB.y);
            AtomicAddFloat(pc.rigid_bodies + (base + 14) * 4, dwB.z);
        }
    } else {
        uint base = (pB_id / SUBGROUP_SIZE) * (10u * SUBGROUP_SIZE) + (pB_id % SUBGROUP_SIZE);
        float mass = asfloat(BDA_LOAD(uint, pc.particles + (base + 6u * SUBGROUP_SIZE) * 4));
        float invMB = mass > 0.0 ? 1.0 / mass : 0.0;
        if (invMB > 0.0) {
            float3 dvB = impulse * invMB;
            AtomicAddFloat(pc.particles + (base + 3u * SUBGROUP_SIZE) * 4, dvB.x);
            AtomicAddFloat(pc.particles + (base + 4u * SUBGROUP_SIZE) * 4, dvB.y);
            AtomicAddFloat(pc.particles + (base + 5u * SUBGROUP_SIZE) * 4, dvB.z);
        }
    }
}


// --- hlsl_stream_compact.txt ---
// BDA Memory Access Macros
// ------------------------------------------------------------------
#ifndef BDA_LOAD
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#endif
#ifndef BDA_STORE
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)
#endif

// ------------------------------------------------------------------
// SPIR-V Atomic Intrinsics for 64-bit BDA Pointers
// ------------------------------------------------------------------
#ifndef SPV_SCOPE_DEVICE
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

// Force DXC to emit explicit SPIR-V atomics mapped to 64-bit PhysicalStorageBuffers
[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(237)]] uint spvAtomicUMin([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(230)]] uint spvAtomicCompareExchange([[vk::ext_reference]] uint64_t ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);
#endif

#if defined(KERNEL_stream_compact)

struct PushConstants_stream_compact {
    uint64_t sparse_in;
    uint64_t packed_out;
    uint total_elements;
};

[[vk::push_constant]]
PushConstants_stream_compact pc;

[numthreads(128, 1, 1)]
void stream_compact(uint3 DispatchThreadID : SV_DispatchThreadID) {
#ifdef DEBUG_SHADERS
    if (DispatchThreadID.x == 0 && DispatchThreadID.y == 0 && DispatchThreadID.z == 0) {
        printf("Executing compute shader: stream_compact.comp");
    }
#endif

    uint id = DispatchThreadID.x;
    
    // offset of count in SparseCollisions is 0
    uint in_count = BDA_LOAD(uint, pc.sparse_in);

    if (id == 0) {
        // packed_out: dispatch_x(0), dispatch_y(4), dispatch_z(8), count(12)
        BDA_STORE(uint, pc.packed_out + 12, in_count);
        uint blocks = (in_count + 127) / 128;
        BDA_STORE(uint, pc.packed_out + 0, blocks);
        BDA_STORE(uint, pc.packed_out + 4, 1);
        BDA_STORE(uint, pc.packed_out + 8, 1);
    }

    if (id < in_count) {
        // sparse_in.pairs offset is 16, SparseCollisionData size is 96
        uint sparse_offset = 16 + id * 96;
        SparseCollisionData in_data = BDA_LOAD(SparseCollisionData, pc.sparse_in + sparse_offset);

        // packed_out.pairs offset is 16, PackedPair size is 80
        uint packed_offset = 16 + id * 80;
        
        BDA_STORE(uint, pc.packed_out + packed_offset + 0, in_data.entity_a);
        BDA_STORE(uint, pc.packed_out + packed_offset + 4, in_data.prim_a);
        
        BDA_STORE(uint, pc.packed_out + packed_offset + 8, in_data.entity_b);
        BDA_STORE(uint, pc.packed_out + packed_offset + 12, in_data.prim_b);
        
        BDA_STORE(float, pc.packed_out + packed_offset + 16, in_data.toi);
        
        BDA_STORE(float4, pc.packed_out + packed_offset + 32, in_data.contact_normal);
        
        BDA_STORE(float4, pc.packed_out + packed_offset + 48, in_data.contact_point);
        
        BDA_STORE(float, pc.packed_out + packed_offset + 64, in_data.penetration_depth);
    }
}

#endif // KERNEL_stream_compact

// --- hlsl_lbvh_build.txt ---
[numthreads(128, 1, 1)]

struct PushConstants_lbvh_build {
    uint64_t bvh;
    uint64_t sorted_morton;
    uint64_t counters;
    uint64_t particles;
    uint num_primitives;
    float particle_radius;
    float dt;
};

#ifdef KERNEL_lbvh_build
[[vk::push_constant]]
PushConstants_lbvh_build pc;
#endif

int common_prefix(uint n, int i, int j) {
    if (j < 0 || j >= (int)n) return -1;
    uint key1 = BDA_LOAD(uint, pc.sorted_morton + i * 8);
    uint key2 = BDA_LOAD(uint, pc.sorted_morton + j * 8);
    if (key1 == key2) {
        uint idx1 = BDA_LOAD(uint, pc.sorted_morton + i * 8 + 4);
        uint idx2 = BDA_LOAD(uint, pc.sorted_morton + j * 8 + 4);
        return 32 + (31 - firstbithigh(idx1 ^ idx2));
    }
    return 31 - firstbithigh(key1 ^ key2);
}

float2 determine_range(uint n, int i) {
    int d = sign((float)(common_prefix(n, i, i + 1) - common_prefix(n, i, i - 1)));
    int min_p = common_prefix(n, i, i - d), l_max = 2;
    while (common_prefix(n, i, i + l_max * d) > min_p) l_max *= 2;
    int l = 0, t = l_max / 2;
    while (t >= 1) { if (common_prefix(n, i, i + (l + t) * d) > min_p) l += t; t /= 2; }
    return float2(min(i, i + l * d), max(i, i + l * d));
}

int find_split(uint n, int first, int last) {
    int common_node = common_prefix(n, first, last), split = first, step = last - first;
    do {
        step = (step + 1) >> 1; int new_split = split + step;
        if (new_split < last && common_prefix(n, first, new_split) > common_node) split = new_split;
    } while (step > 1);
    return split;
}

#define NODE_SIZE 2832
#define NODE_MIN_X 0
#define NODE_MAX_X 128
#define NODE_MIN_Y 256
#define NODE_MAX_Y 384
#define NODE_MIN_Z 512
#define NODE_MAX_Z 640
#define NODE_CHILD 768
#define NODE_METADATA 896
#define NODE_MASSES 1024
#define NODE_COM_X 1152
#define NODE_COM_Y 1280
#define NODE_COM_Z 1408
#define NODE_VALID_MASK 1792
#define NODE_PARENT_IDX 1800

#define BVH_LOAD_UINT(bvh, node, offset, is_r) BDA_LOAD(uint, (bvh) + (node) * NODE_SIZE + (offset) + (is_r) * 4)
#define BVH_STORE_UINT(bvh, node, offset, is_r, val) BDA_STORE(uint, (bvh) + (node) * NODE_SIZE + (offset) + (is_r) * 4, val)
#define BVH_LOAD_FLOAT(bvh, node, offset, is_r) BDA_LOAD(float, (bvh) + (node) * NODE_SIZE + (offset) + (is_r) * 4)
#define BVH_STORE_FLOAT(bvh, node, offset, is_r, val) BDA_STORE(float, (bvh) + (node) * NODE_SIZE + (offset) + (is_r) * 4, val)

#ifdef KERNEL_lbvh_build
void lbvh_build(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint idx = DispatchThreadID.x, n = pc.num_primitives;
    if (idx >= n) return;
    uint num_internal_nodes = n - 1;

    if (idx < num_internal_nodes) {
        float2 range = determine_range(n, int(idx));
        int split = find_split(n, int(range.x), int(range.y));
        uint left_child = (split == int(range.x)) ? (num_internal_nodes + split) : uint(split);
        uint right_child = (split + 1 == int(range.y)) ? (num_internal_nodes + split + 1) : uint(split + 1);

        BVH_STORE_UINT(pc.bvh, idx, NODE_CHILD, 0, left_child);
        BVH_STORE_UINT(pc.bvh, idx, NODE_CHILD, 1, right_child);
        BDA_STORE(uint2, pc.bvh + idx * NODE_SIZE + NODE_VALID_MASK, uint2(3u, 0u));
        BDA_STORE(uint, pc.bvh + left_child * NODE_SIZE + NODE_PARENT_IDX, idx);
        BDA_STORE(uint, pc.bvh + right_child * NODE_SIZE + NODE_PARENT_IDX, idx);
    }

    uint leaf_idx = num_internal_nodes + idx;
    uint p_id = BDA_LOAD(uint, pc.sorted_morton + idx * 8 + 4);
    uint base = (p_id / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (p_id % SUBGROUP_SIZE);

    float3 pos = float3(
        BDA_LOAD(float, pc.particles + (base+0)*4),
        BDA_LOAD(float, pc.particles + (base+1*SUBGROUP_SIZE)*4),
        BDA_LOAD(float, pc.particles + (base+2*SUBGROUP_SIZE)*4)
    );
    float3 vel = float3(
        BDA_LOAD(float, pc.particles + (base+3*SUBGROUP_SIZE)*4),
        BDA_LOAD(float, pc.particles + (base+4*SUBGROUP_SIZE)*4),
        BDA_LOAD(float, pc.particles + (base+5*SUBGROUP_SIZE)*4)
    );
    float mass = BDA_LOAD(float, pc.particles + (base+6*SUBGROUP_SIZE)*4);
    float r = pc.particle_radius;

    float3 p1 = pos + vel * pc.dt;
    float3 l_min = min(pos - float3(r, r, r), p1 - float3(r, r, r));
    float3 l_max = max(pos + float3(r, r, r), p1 + float3(r, r, r));

    uint current = BDA_LOAD(uint, pc.bvh + leaf_idx * NODE_SIZE + NODE_PARENT_IDX);
    uint is_right = (BVH_LOAD_UINT(pc.bvh, current, NODE_CHILD, 1) == leaf_idx) ? 1 : 0;

    BVH_STORE_FLOAT(pc.bvh, current, NODE_MIN_X, is_right, l_min.x); BVH_STORE_FLOAT(pc.bvh, current, NODE_MAX_X, is_right, l_max.x);
    BVH_STORE_FLOAT(pc.bvh, current, NODE_MIN_Y, is_right, l_min.y); BVH_STORE_FLOAT(pc.bvh, current, NODE_MAX_Y, is_right, l_max.y);
    BVH_STORE_FLOAT(pc.bvh, current, NODE_MIN_Z, is_right, l_min.z); BVH_STORE_FLOAT(pc.bvh, current, NODE_MAX_Z, is_right, l_max.z);
    BVH_STORE_FLOAT(pc.bvh, current, NODE_MASSES, is_right, mass);
    BVH_STORE_FLOAT(pc.bvh, current, NODE_COM_X, is_right, pos.x); BVH_STORE_FLOAT(pc.bvh, current, NODE_COM_Y, is_right, pos.y); BVH_STORE_FLOAT(pc.bvh, current, NODE_COM_Z, is_right, pos.z);
    BVH_STORE_UINT(pc.bvh, current, NODE_METADATA, is_right, bvh_pack_metadata(true, BVH_FRAME_MICRO, BVH_SHAPE_AABB, p_id));

    DeviceMemoryBarrierWithGroupSync();

    while (current != 0xFFFFFFFFu) {
        uint counter_addr = pc.counters + current * 4;
        uint old_val = spvAtomicIAdd(counter_addr, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, 1);
        if (old_val == 0) break;

        float3 c_l_min = float3(BVH_LOAD_FLOAT(pc.bvh, current, NODE_MIN_X, 0), BVH_LOAD_FLOAT(pc.bvh, current, NODE_MIN_Y, 0), BVH_LOAD_FLOAT(pc.bvh, current, NODE_MIN_Z, 0));
        float3 c_l_max = float3(BVH_LOAD_FLOAT(pc.bvh, current, NODE_MAX_X, 0), BVH_LOAD_FLOAT(pc.bvh, current, NODE_MAX_Y, 0), BVH_LOAD_FLOAT(pc.bvh, current, NODE_MAX_Z, 0));
        float l_m = BVH_LOAD_FLOAT(pc.bvh, current, NODE_MASSES, 0);
        float3 l_com = float3(BVH_LOAD_FLOAT(pc.bvh, current, NODE_COM_X, 0), BVH_LOAD_FLOAT(pc.bvh, current, NODE_COM_Y, 0), BVH_LOAD_FLOAT(pc.bvh, current, NODE_COM_Z, 0));

        float3 c_r_min = float3(BVH_LOAD_FLOAT(pc.bvh, current, NODE_MIN_X, 1), BVH_LOAD_FLOAT(pc.bvh, current, NODE_MIN_Y, 1), BVH_LOAD_FLOAT(pc.bvh, current, NODE_MIN_Z, 1));
        float3 c_r_max = float3(BVH_LOAD_FLOAT(pc.bvh, current, NODE_MAX_X, 1), BVH_LOAD_FLOAT(pc.bvh, current, NODE_MAX_Y, 1), BVH_LOAD_FLOAT(pc.bvh, current, NODE_MAX_Z, 1));
        float r_m = BVH_LOAD_FLOAT(pc.bvh, current, NODE_MASSES, 1);
        float3 r_com = float3(BVH_LOAD_FLOAT(pc.bvh, current, NODE_COM_X, 1), BVH_LOAD_FLOAT(pc.bvh, current, NODE_COM_Y, 1), BVH_LOAD_FLOAT(pc.bvh, current, NODE_COM_Z, 1));

        float3 c_min = min(c_l_min, c_r_min), c_max = max(c_l_max, c_r_max);
        float c_mass = l_m + r_m;
        float3 c_com = c_mass > 0.0 ? (l_com * l_m + r_com * r_m) / c_mass : (l_com + r_com) * 0.5;

        uint parent = BDA_LOAD(uint, pc.bvh + current * NODE_SIZE + NODE_PARENT_IDX);
        if (parent != 0xFFFFFFFFu) {
            uint is_r = (BVH_LOAD_UINT(pc.bvh, parent, NODE_CHILD, 1) == current) ? 1 : 0;
            BVH_STORE_FLOAT(pc.bvh, parent, NODE_MIN_X, is_r, c_min.x); BVH_STORE_FLOAT(pc.bvh, parent, NODE_MAX_X, is_r, c_max.x);
            BVH_STORE_FLOAT(pc.bvh, parent, NODE_MIN_Y, is_r, c_min.y); BVH_STORE_FLOAT(pc.bvh, parent, NODE_MAX_Y, is_r, c_max.y);
            BVH_STORE_FLOAT(pc.bvh, parent, NODE_MIN_Z, is_r, c_min.z); BVH_STORE_FLOAT(pc.bvh, parent, NODE_MAX_Z, is_r, c_max.z);
            BVH_STORE_FLOAT(pc.bvh, parent, NODE_MASSES, is_r, c_mass);
            BVH_STORE_FLOAT(pc.bvh, parent, NODE_COM_X, is_r, c_com.x); BVH_STORE_FLOAT(pc.bvh, parent, NODE_COM_Y, is_r, c_com.y); BVH_STORE_FLOAT(pc.bvh, parent, NODE_COM_Z, is_r, c_com.z);
            BVH_STORE_UINT(pc.bvh, parent, NODE_METADATA, is_r, bvh_pack_metadata(false, BVH_FRAME_MICRO, BVH_SHAPE_AABB, current));
        }
        DeviceMemoryBarrierWithGroupSync();
        current = parent;
    }
}
#endif


// --- hlsl_lbvh_collapse.txt ---
#define SUBGROUP_SIZE 32

// BDA Memory Access Macros
// ------------------------------------------------------------------
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

#define OFFSET_MIN_X 0
#define OFFSET_MAX_X (SUBGROUP_SIZE * 4)
#define OFFSET_MIN_Y (SUBGROUP_SIZE * 8)
#define OFFSET_MAX_Y (SUBGROUP_SIZE * 12)
#define OFFSET_MIN_Z (SUBGROUP_SIZE * 16)
#define OFFSET_MAX_Z (SUBGROUP_SIZE * 20)
#define OFFSET_CHILD_INDICES (SUBGROUP_SIZE * 24)
#define OFFSET_METADATA (SUBGROUP_SIZE * 28)
#define OFFSET_MASSES (SUBGROUP_SIZE * 32)
#define OFFSET_COM_X (SUBGROUP_SIZE * 36)
#define OFFSET_COM_Y (SUBGROUP_SIZE * 40)
#define OFFSET_COM_Z (SUBGROUP_SIZE * 44)
#define OFFSET_PARTICLE_START (SUBGROUP_SIZE * 48)
#define OFFSET_PARTICLE_COUNT (SUBGROUP_SIZE * 52)
#define OFFSET_VALID_MASK (SUBGROUP_SIZE * 56)
#define OFFSET_PARENT_IDX (OFFSET_VALID_MASK + 8)
#define OFFSET_PAD (OFFSET_VALID_MASK + 12)
#define OFFSET_PERMUTATIONS (OFFSET_VALID_MASK + 16)
#define SIZEOF_MULTI_BVH_NODE (OFFSET_PERMUTATIONS + 8 * SUBGROUP_SIZE * 4)

#define BVH_FRAME_MACRO  0u
#define BVH_FRAME_MICRO  1u
#define BVH_SHAPE_AABB   0u
#define BVH_SHAPE_OBB    1u
#define BVH_SHAPE_SPHERE 2u

bool bvh_is_leaf(uint meta)   { return (meta & 0x80000000u) != 0u; }
uint bvh_get_index(uint meta) { return meta & 0x07FFFFFFu; }
uint bvh_pack_metadata(bool is_leaf, uint frame, uint shape, uint index) {
    uint meta = index & 0x07FFFFFFu;
    meta |= (shape & 0x3u) << 27;
    meta |= (frame & 0x3u) << 29;
    if (is_leaf) meta |= 0x80000000u;
    return meta;
}

struct PushConstants {
    uint64_t binary_bvh;
    uint64_t multi_bvh;
    uint64_t collapse_map;
    uint num_multi_nodes;
};

[[vk::push_constant]]
PushConstants pc;

[numthreads(SUBGROUP_SIZE, 1, 1)]
void lbvh_collapse(uint3 GroupID : SV_GroupID, uint3 DispatchThreadID : SV_DispatchThreadID, uint GroupIndex : SV_GroupIndex) {
    uint multi_node_idx = GroupID.x;
    if (multi_node_idx >= pc.num_multi_nodes) return;

    uint lane = GroupIndex;
    uint binary_idx = BDA_LOAD(uint, pc.collapse_map + multi_node_idx * 4);
    
    bool is_leaf = false;
    uint payload = 0;
    uint f_parent = 0;
    uint f_dir = 0;

    int depth = firstbithigh(SUBGROUP_SIZE) - 1;
    for (int d = depth; d >= 0; d--) {
        uint dir = (lane >> d) & 1u;
        uint64_t node_addr = pc.binary_bvh + binary_idx * SIZEOF_MULTI_BVH_NODE;
        uint meta = BDA_LOAD(uint, node_addr + OFFSET_METADATA + dir * 4);
        
        is_leaf = bvh_is_leaf(meta);
        uint next_idx = bvh_get_index(meta);

        f_parent = binary_idx;
        f_dir = dir;
        if (is_leaf) { payload = next_idx; break; }
        binary_idx = next_idx;
    }

    if (!is_leaf) {
        payload = binary_idx;
        uint64_t node_addr = pc.binary_bvh + binary_idx * SIZEOF_MULTI_BVH_NODE;
        f_parent = BDA_LOAD(uint, node_addr + OFFSET_PARENT_IDX);
        uint64_t parent_addr = pc.binary_bvh + f_parent * SIZEOF_MULTI_BVH_NODE;
        uint child_1 = BDA_LOAD(uint, parent_addr + OFFSET_CHILD_INDICES + 1 * 4);
        f_dir = (child_1 == binary_idx) ? 1u : 0u;
    }

    uint64_t f_parent_addr = pc.binary_bvh + f_parent * SIZEOF_MULTI_BVH_NODE;
    uint64_t multi_node_addr = pc.multi_bvh + multi_node_idx * SIZEOF_MULTI_BVH_NODE;

    float p_min_x = BDA_LOAD(float, f_parent_addr + OFFSET_MIN_X + f_dir * 4);
    BDA_STORE(float, multi_node_addr + OFFSET_MIN_X + lane * 4, p_min_x);
    
    float p_max_x = BDA_LOAD(float, f_parent_addr + OFFSET_MAX_X + f_dir * 4);
    BDA_STORE(float, multi_node_addr + OFFSET_MAX_X + lane * 4, p_max_x);
    
    float p_min_y = BDA_LOAD(float, f_parent_addr + OFFSET_MIN_Y + f_dir * 4);
    BDA_STORE(float, multi_node_addr + OFFSET_MIN_Y + lane * 4, p_min_y);
    
    float p_max_y = BDA_LOAD(float, f_parent_addr + OFFSET_MAX_Y + f_dir * 4);
    BDA_STORE(float, multi_node_addr + OFFSET_MAX_Y + lane * 4, p_max_y);
    
    float p_min_z = BDA_LOAD(float, f_parent_addr + OFFSET_MIN_Z + f_dir * 4);
    BDA_STORE(float, multi_node_addr + OFFSET_MIN_Z + lane * 4, p_min_z);
    
    float p_max_z = BDA_LOAD(float, f_parent_addr + OFFSET_MAX_Z + f_dir * 4);
    BDA_STORE(float, multi_node_addr + OFFSET_MAX_Z + lane * 4, p_max_z);
    
    BDA_STORE(uint, multi_node_addr + OFFSET_CHILD_INDICES + lane * 4, payload);
    
    uint packed_meta = bvh_pack_metadata(is_leaf, BVH_FRAME_MICRO, BVH_SHAPE_AABB, payload);
    BDA_STORE(uint, multi_node_addr + OFFSET_METADATA + lane * 4, packed_meta);
    
    float p_mass = BDA_LOAD(float, f_parent_addr + OFFSET_MASSES + f_dir * 4);
    BDA_STORE(float, multi_node_addr + OFFSET_MASSES + lane * 4, p_mass);
    
    float p_com_x = BDA_LOAD(float, f_parent_addr + OFFSET_COM_X + f_dir * 4);
    BDA_STORE(float, multi_node_addr + OFFSET_COM_X + lane * 4, p_com_x);
    
    float p_com_y = BDA_LOAD(float, f_parent_addr + OFFSET_COM_Y + f_dir * 4);
    BDA_STORE(float, multi_node_addr + OFFSET_COM_Y + lane * 4, p_com_y);
    
    float p_com_z = BDA_LOAD(float, f_parent_addr + OFFSET_COM_Z + f_dir * 4);
    BDA_STORE(float, multi_node_addr + OFFSET_COM_Z + lane * 4, p_com_z);

    if (lane == 0) {
        uint mask_x = (SUBGROUP_SIZE >= 32) ? 0xFFFFFFFFu : ((1u << SUBGROUP_SIZE) - 1u);
        uint mask_y = 0u;
        if (SUBGROUP_SIZE > 32) mask_y = (SUBGROUP_SIZE >= 64) ? 0xFFFFFFFFu : ((1u << (SUBGROUP_SIZE - 32)) - 1u);
        
        BDA_STORE(uint, multi_node_addr + OFFSET_VALID_MASK, mask_x);
        BDA_STORE(uint, multi_node_addr + OFFSET_VALID_MASK + 4, mask_y);
        
        for (uint i = 0; i < 8; ++i) {
            for (uint j = 0; j < SUBGROUP_SIZE; ++j) {
                BDA_STORE(uint, multi_node_addr + OFFSET_PERMUTATIONS + (i * SUBGROUP_SIZE + j) * 4, j);
            }
        }
    }
}


// --- hlsl_lbvh_prepass.txt ---


// BDA Memory Access Macros
#ifndef BDA_LOAD
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)
#endif

#ifdef KERNEL_LBVH_PREPASS

struct PushConstants_lbvh_prepass {
    uint64_t bvh;
    uint64_t counters;
    uint num_internal_nodes;
};

[[vk::push_constant]]
PushConstants_lbvh_prepass pc_lbvh_prepass;

[numthreads(256, 1, 1)]
void lbvh_prepass(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint idx = DispatchThreadID.x;
    if (idx >= pc_lbvh_prepass.num_internal_nodes) return;
    
    // pc.counters.counts[idx] = 0;
    uint64_t count_addr = pc_lbvh_prepass.counters + (idx * 4);
    BDA_STORE(uint, count_addr, 0u);
    
    if (idx == 0u) {
        // pc.bvh.nodes[0].parent_idx = 0xFFFFFFFFu;
        // MultiBvhNode's parent_idx offset is 1800 bytes
        uint64_t node_addr = pc_lbvh_prepass.bvh;
        BDA_STORE(uint, node_addr + 1800, 0xFFFFFFFFu);
    }
}

#endif // KERNEL_LBVH_PREPASS


// --- hlsl_morton_encode.txt ---

#include "../debug_utils.glsl"

// BDA Memory Access Macros
// ------------------------------------------------------------------
#ifndef BDA_LOAD
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#endif

#ifndef BDA_STORE
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)
#endif

// ------------------------------------------------------------------
// SPIR-V Atomic Intrinsics for 64-bit BDA Pointers
// ------------------------------------------------------------------
#ifndef SPV_SCOPE_DEVICE
#define SPV_SCOPE_DEVICE 1
#endif
#ifndef SPV_SEMANTICS_RELAXED
#define SPV_SEMANTICS_RELAXED 0
#endif

#ifndef SPV_ATOMIC_DECLARATIONS
#define SPV_ATOMIC_DECLARATIONS
[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(237)]] uint spvAtomicUMin([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(230)]] uint spvAtomicCompareExchange([[vk::ext_reference]] uint64_t ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);
#endif

#ifndef KERNEL_morton_encode_SUBGROUP_SIZE
#define KERNEL_morton_encode_SUBGROUP_SIZE 32
#endif

struct PushConstants_morton_encode {
    uint64_t morton_out;
    uint64_t particles;
    uint num_particles;
    float3 scene_min;
    float3 scene_max;
};

[[vk::push_constant]]
PushConstants_morton_encode pc_morton_encode;

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
uint morton_encode_expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint morton_encode_morton3D(float3 norm_pos) {
    norm_pos = clamp(norm_pos, 0.0f, 1.0f);
    uint x = (uint)(norm_pos.x * 1023.0f);
    uint y = (uint)(norm_pos.y * 1023.0f);
    uint z = (uint)(norm_pos.z * 1023.0f);
    return (morton_encode_expandBits(x) << 2) | (morton_encode_expandBits(y) << 1) | morton_encode_expandBits(z);
}

[numthreads(KERNEL_morton_encode_SUBGROUP_SIZE, 1, 1)]
void morton_encode(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint idx = DispatchThreadID.x;
    if (idx >= pc_morton_encode.num_particles) return;

    // AOSOA unpacking matching your particle structure
    uint block_idx = idx / KERNEL_morton_encode_SUBGROUP_SIZE;
    uint local_idx = idx % KERNEL_morton_encode_SUBGROUP_SIZE;
    uint base = block_idx * (10 * KERNEL_morton_encode_SUBGROUP_SIZE) + local_idx;

    uint p_x_uint = BDA_LOAD(uint, pc_morton_encode.particles + (base + 0 * KERNEL_morton_encode_SUBGROUP_SIZE) * 4);
    uint p_y_uint = BDA_LOAD(uint, pc_morton_encode.particles + (base + 1 * KERNEL_morton_encode_SUBGROUP_SIZE) * 4);
    uint p_z_uint = BDA_LOAD(uint, pc_morton_encode.particles + (base + 2 * KERNEL_morton_encode_SUBGROUP_SIZE) * 4);

    float3 pos = float3(asfloat(p_x_uint), asfloat(p_y_uint), asfloat(p_z_uint));

    // Normalize relative to scene bounds
    float3 extents = pc_morton_encode.scene_max - pc_morton_encode.scene_min;
    float3 norm_pos = (pos - pc_morton_encode.scene_min) / max(extents, float3(1e-5f, 1e-5f, 1e-5f));

    uint m_code = morton_encode_morton3D(norm_pos);

    BDA_STORE(uint2, pc_morton_encode.morton_out + idx * 8, uint2(m_code, idx));
}

// --- hlsl_radix_sort.txt ---
#include "physics_core.hlsl"

struct PushConstants_radix_sort {
    uint64_t input_keys;
    uint64_t output_keys;
    uint64_t histograms;
    uint num_particles;
    uint shift;
    uint stage;
    uint num_blocks;
};

#ifndef KERNEL_radix_sort
[[vk::push_constant]]
PushConstants_radix_sort pc;
#endif

#define STAGE_COUNT   0
#define STAGE_SCAN    1
#define STAGE_SCATTER 2

#define RADIX 16
#define ELEMENTS_PER_BLOCK 4096

groupshared uint s_counts[RADIX];
groupshared uint s_offsets[RADIX];
groupshared uint s_sg_counts[64]; // Supports up to 64 subgroups (down to a subgroup size of 4)
groupshared uint s_bin_sums[RADIX];

[numthreads(256, 1, 1)]
void radix_sort(
    uint3 DispatchThreadID : SV_DispatchThreadID,
    uint3 GroupID : SV_GroupID,
    uint3 GroupThreadID : SV_GroupThreadID,
    uint GroupIndex : SV_GroupIndex
) {
    uint lid = GroupThreadID.x;
    uint wid = GroupID.x;
    uint sg_id = WaveGetLaneIndex();
    
    uint num_subgroups = (256 + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    uint sg_group_id = lid / WaveGetLaneCount();

    if (pc.stage == STAGE_COUNT) {
        if (lid < RADIX) s_counts[lid] = 0;
        GroupMemoryBarrierWithGroupSync();

        uint block_start = wid * ELEMENTS_PER_BLOCK;
        uint block_end = min(block_start + ELEMENTS_PER_BLOCK, pc.num_particles);

        for (uint i = block_start + lid; i < block_end; i += 256) {
            uint2 entry = BDA_LOAD(uint2, pc.input_keys + i * 8);
            uint key = (entry.x >> pc.shift) & 0xFu;
            InterlockedAdd(s_counts[key], 1);
        }
        GroupMemoryBarrierWithGroupSync();

        if (lid < RADIX) {
            BDA_STORE(uint, pc.histograms + (lid * pc.num_blocks + wid) * 4, s_counts[lid]);
        }
    }
    else if (pc.stage == STAGE_SCAN) {
        if (lid < RADIX) {
            uint bin_sum = 0;
            for (uint w = 0; w < pc.num_blocks; ++w) {
                bin_sum += BDA_LOAD(uint, pc.histograms + (lid * pc.num_blocks + w) * 4);
            }
            s_bin_sums[lid] = bin_sum;
        }
        GroupMemoryBarrierWithGroupSync();

        if (lid == 0) {
            uint global_offset = 0;
            for (uint i = 0; i < RADIX; ++i) {
                uint val = s_bin_sums[i];
                s_bin_sums[i] = global_offset;
                global_offset += val;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (lid < RADIX) {
            uint running_offset = s_bin_sums[lid];
            for (uint w = 0; w < pc.num_blocks; ++w) {
                uint addr = pc.histograms + (lid * pc.num_blocks + w) * 4;
                uint val = BDA_LOAD(uint, addr);
                BDA_STORE(uint, addr, running_offset);
                running_offset += val;
            }
        }
    }
    else if (pc.stage == STAGE_SCATTER) {
        if (lid < RADIX) {
            s_offsets[lid] = BDA_LOAD(uint, pc.histograms + (lid * pc.num_blocks + wid) * 4);
        }
        GroupMemoryBarrierWithGroupSync();

        uint block_start = wid * ELEMENTS_PER_BLOCK;
        uint block_end = min(block_start + ELEMENTS_PER_BLOCK, pc.num_particles);

        for (uint chunk_start = block_start; chunk_start < block_end; chunk_start += 256) {
            uint i = chunk_start + lid;
            bool valid = (i < block_end);
            
            uint2 my_entry = uint2(0, 0);
            if (valid) {
                my_entry = BDA_LOAD(uint2, pc.input_keys + i * 8);
            }
            uint my_key = valid ? ((my_entry.x >> pc.shift) & 0xFu) : 0xFFFFFFFFu;

            uint local_offset = 0;
            uint my_global_base = 0;

            for (uint b = 0; b < RADIX; ++b) {
                bool match = (my_key == b);

                uint sg_match_count = WaveActiveCountBits(match);
                uint my_sg_offset = WavePrefixCountBits(match);

                if (sg_id == 0) {
                    s_sg_counts[sg_group_id] = sg_match_count;
                }
                GroupMemoryBarrierWithGroupSync();

                if (lid == 0) {
                    uint sum = 0;
                    for (uint sg = 0; sg < num_subgroups; ++sg) {
                        uint c = s_sg_counts[sg];
                        s_sg_counts[sg] = sum;
                        sum += c;
                    }
                    s_counts[b] = sum;
                }
                GroupMemoryBarrierWithGroupSync();

                if (match) {
                    local_offset = s_sg_counts[sg_group_id] + my_sg_offset;
                    my_global_base = s_offsets[b];
                }

                if (lid == 0) {
                    s_offsets[b] += s_counts[b];
                }
                GroupMemoryBarrierWithGroupSync();
            }

            if (valid) {
                uint dest = my_global_base + local_offset;
                BDA_STORE(uint2, pc.output_keys + dest * 8, my_entry);
            }
        }
    }
}

// --- hlsl_motion_bounds.txt ---


#ifndef BDA_LOAD
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#endif
#ifndef BDA_STORE
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)
#endif

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#ifndef PRIMITIVE_TYPE
#define PRIMITIVE_TYPE 0
#endif

#define MULTI_BVH_NODE_SIZE (22 * SUBGROUP_SIZE * 4 + 16)
#define OFFSET_PARENT_IDX (14 * SUBGROUP_SIZE * 4 + 8)
#define OFFSET_CHILD_INDICES (6 * SUBGROUP_SIZE * 4)

struct PushConstants_motion_bounds {
    uint64_t bvh;
    uint64_t primitive_data;
    uint num_primitives;
    float dt;
    float particle_radius;
};

#ifdef KERNEL_motion_bounds
[[vk::push_constant]]
PushConstants_motion_bounds pc;
#endif

[numthreads(256, 1, 1)]
void motion_bounds(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint idx = DispatchThreadID.x;
    if (idx >= pc.num_primitives) return;

    if (PRIMITIVE_TYPE == 0) {
        uint base = (idx / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (idx % SUBGROUP_SIZE);
        
        float pos_x = asfloat(BDA_LOAD(uint, pc.primitive_data + (base + 0) * 4));
        float pos_y = asfloat(BDA_LOAD(uint, pc.primitive_data + (base + 1 * SUBGROUP_SIZE) * 4));
        float pos_z = asfloat(BDA_LOAD(uint, pc.primitive_data + (base + 2 * SUBGROUP_SIZE) * 4));
        float3 pos = float3(pos_x, pos_y, pos_z);
        
        float vel_x = asfloat(BDA_LOAD(uint, pc.primitive_data + (base + 3 * SUBGROUP_SIZE) * 4));
        float vel_y = asfloat(BDA_LOAD(uint, pc.primitive_data + (base + 4 * SUBGROUP_SIZE) * 4));
        float vel_z = asfloat(BDA_LOAD(uint, pc.primitive_data + (base + 5 * SUBGROUP_SIZE) * 4));
        float3 vel = float3(vel_x, vel_y, vel_z);

        float3 p1 = pos + vel * pc.dt;
        float3 min_p = min(pos, p1) - pc.particle_radius;
        float3 max_p = max(pos, p1) + pc.particle_radius;

        uint leaf_idx = (pc.num_primitives - 1) + idx;
        
        uint64_t leaf_addr = pc.bvh + leaf_idx * MULTI_BVH_NODE_SIZE;
        uint parent = BDA_LOAD(uint, leaf_addr + OFFSET_PARENT_IDX);
        
        uint64_t parent_addr = pc.bvh + parent * MULTI_BVH_NODE_SIZE;
        uint child_1 = BDA_LOAD(uint, parent_addr + OFFSET_CHILD_INDICES + 4);
        
        uint is_right = (child_1 == leaf_idx) ? 1 : 0;

        BDA_STORE(float, parent_addr + 0 * SUBGROUP_SIZE * 4 + is_right * 4, min_p.x);
        BDA_STORE(float, parent_addr + 1 * SUBGROUP_SIZE * 4 + is_right * 4, max_p.x);
        BDA_STORE(float, parent_addr + 2 * SUBGROUP_SIZE * 4 + is_right * 4, min_p.y);
        BDA_STORE(float, parent_addr + 3 * SUBGROUP_SIZE * 4 + is_right * 4, max_p.y);
        BDA_STORE(float, parent_addr + 4 * SUBGROUP_SIZE * 4 + is_right * 4, min_p.z);
        BDA_STORE(float, parent_addr + 5 * SUBGROUP_SIZE * 4 + is_right * 4, max_p.z);
    }
}


// --- hlsl_motion_refit.txt ---
#include "../bvh_utils.hlsli"

// BDA Memory Access Macros
// ------------------------------------------------------------------
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

// ------------------------------------------------------------------
// SPIR-V Atomic Intrinsics for 64-bit BDA Pointers
// ------------------------------------------------------------------
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(237)]] uint spvAtomicUMin([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(230)]] uint spvAtomicCompareExchange([[vk::ext_reference]] uint64_t ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);

struct PushConstants {
    uint64_t bvh;
    uint64_t depth_indices;
    uint total_nodes_at_depth;
};

[[vk::push_constant]]
PushConstants pc;

#define MULTI_BVH_NODE_SIZE 2832
#define OFFSET_MIN_X 0
#define OFFSET_MAX_X 128
#define OFFSET_MIN_Y 256
#define OFFSET_MAX_Y 384
#define OFFSET_MIN_Z 512
#define OFFSET_MAX_Z 640
#define OFFSET_CHILD_INDICES 768
#define OFFSET_METADATA 896

[numthreads(256, 1, 1)]
void motion_refit(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint global_id = DispatchThreadID.x;
    if (global_id >= pc.total_nodes_at_depth) return;

    uint node_idx = BDA_LOAD(uint, pc.depth_indices + 4 + (global_id + 4) * 4);
    uint64_t node_addr = pc.bvh + node_idx * MULTI_BVH_NODE_SIZE;

    for (uint i = 0; i < 2; ++i) {
        uint child = BDA_LOAD(uint, node_addr + OFFSET_CHILD_INDICES + i * 4);
        uint metadata = BDA_LOAD(uint, node_addr + OFFSET_METADATA + i * 4);
        uint64_t child_addr = pc.bvh + child * MULTI_BVH_NODE_SIZE;

        if (bvh_is_leaf(metadata)) {
            BDA_STORE(float, node_addr + OFFSET_MIN_X + i * 4, BDA_LOAD(float, child_addr + OFFSET_MIN_X + 0 * 4));
            BDA_STORE(float, node_addr + OFFSET_MAX_X + i * 4, BDA_LOAD(float, child_addr + OFFSET_MAX_X + 0 * 4));
            BDA_STORE(float, node_addr + OFFSET_MIN_Y + i * 4, BDA_LOAD(float, child_addr + OFFSET_MIN_Y + 0 * 4));
            BDA_STORE(float, node_addr + OFFSET_MAX_Y + i * 4, BDA_LOAD(float, child_addr + OFFSET_MAX_Y + 0 * 4));
            BDA_STORE(float, node_addr + OFFSET_MIN_Z + i * 4, BDA_LOAD(float, child_addr + OFFSET_MIN_Z + 0 * 4));
            BDA_STORE(float, node_addr + OFFSET_MAX_Z + i * 4, BDA_LOAD(float, child_addr + OFFSET_MAX_Z + 0 * 4));
        } else {
            BDA_STORE(float, node_addr + OFFSET_MIN_X + i * 4, min(BDA_LOAD(float, child_addr + OFFSET_MIN_X + 0 * 4), BDA_LOAD(float, child_addr + OFFSET_MIN_X + 1 * 4)));
            BDA_STORE(float, node_addr + OFFSET_MAX_X + i * 4, max(BDA_LOAD(float, child_addr + OFFSET_MAX_X + 0 * 4), BDA_LOAD(float, child_addr + OFFSET_MAX_X + 1 * 4)));
            BDA_STORE(float, node_addr + OFFSET_MIN_Y + i * 4, min(BDA_LOAD(float, child_addr + OFFSET_MIN_Y + 0 * 4), BDA_LOAD(float, child_addr + OFFSET_MIN_Y + 1 * 4)));
            BDA_STORE(float, node_addr + OFFSET_MAX_Y + i * 4, max(BDA_LOAD(float, child_addr + OFFSET_MAX_Y + 0 * 4), BDA_LOAD(float, child_addr + OFFSET_MAX_Y + 1 * 4)));
            BDA_STORE(float, node_addr + OFFSET_MIN_Z + i * 4, min(BDA_LOAD(float, child_addr + OFFSET_MIN_Z + 0 * 4), BDA_LOAD(float, child_addr + OFFSET_MIN_Z + 1 * 4)));
            BDA_STORE(float, node_addr + OFFSET_MAX_Z + i * 4, max(BDA_LOAD(float, child_addr + OFFSET_MAX_Z + 0 * 4), BDA_LOAD(float, child_addr + OFFSET_MAX_Z + 1 * 4)));
        }
    }
}


// --- hlsl_reduce_toi.txt ---
// BDA Memory Access Macros
// ------------------------------------------------------------------
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

// ------------------------------------------------------------------
// SPIR-V Atomic Intrinsics for 64-bit BDA Pointers
// ------------------------------------------------------------------
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

// Force DXC to emit explicit SPIR-V atomics mapped to 64-bit PhysicalStorageBuffers
[[vk::ext_instruction(237)]] uint spvAtomicUMin([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);

struct PushConstants {
    uint64_t particles;
    uint64_t collisions;
    uint64_t out_toi;
    float particle_radius;
    float dt;
};

[[vk::push_constant]]
PushConstants pc;

groupshared uint shared_min_toi[4];

[numthreads(128, 1, 1)]
void reduce_toi(
    uint3 DispatchThreadID : SV_DispatchThreadID,
    uint GroupIndex : SV_GroupIndex
) {
    uint global_id = DispatchThreadID.x;
    uint local_id = GroupIndex;
    
    // Wave size can vary (e.g. 32 or 64 on Vulkan), safely handle subgroups
    uint subgroup_id = GroupIndex / WaveGetLaneCount();
    
    float tc = pc.dt; // Default to max time
    
    uint collisions_count = BDA_LOAD(uint, pc.collisions + 12);
    
    if (global_id < collisions_count) {
        // Size of PackedPair is 80 bytes. Toi is offset 16
        uint64_t pair_addr = pc.collisions + 16 + global_id * 80;
        tc = BDA_LOAD(float, pair_addr + 16);
    }
    
    // Subgroup reduction
    float subgroup_min_tc = WaveActiveMin(tc);
    
    if (WaveIsFirstLane()) {
        shared_min_toi[subgroup_id] = asuint(subgroup_min_tc);
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    // Workgroup reduction
    if (local_id == 0) {
        uint num_subgroups = 128 / WaveGetLaneCount();
        uint wg_min_uint = shared_min_toi[0];
        for (uint i = 1; i < num_subgroups; i++) {
            wg_min_uint = min(wg_min_uint, shared_min_toi[i]);
        }
        
        // Global reduction
        spvAtomicUMin(pc.out_toi, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, wg_min_uint);
    }
}


// --- hlsl_graph_coloring.txt ---
// BDA Memory Access Macros
// ------------------------------------------------------------------
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

// ------------------------------------------------------------------
// SPIR-V Atomic Intrinsics for 64-bit BDA Pointers
// ------------------------------------------------------------------
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

// Force DXC to emit explicit SPIR-V atomics mapped to 64-bit PhysicalStorageBuffers
[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(237)]] uint spvAtomicUMin([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(230)]] uint spvAtomicCompareExchange([[vk::ext_reference]] uint64_t ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);

struct PushConstants_graph_coloring {
    uint64_t collisions;
    uint64_t colors;
    uint64_t weights;
    uint total_pairs;
};

#ifdef KERNEL_graph_coloring
[[vk::push_constant]]
PushConstants_graph_coloring pc;
#endif

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

#ifdef KERNEL_graph_coloring
[numthreads(256, 1, 1)]
void graph_coloring(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint idx = DispatchThreadID.x;
    if (idx >= pc.total_pairs) return;

    // 1. Initialize weights
    BDA_STORE(uint, pc.weights + idx * 4, hash(idx + 1));
    BDA_STORE(uint, pc.colors + idx * 4, 0);

    DeviceMemoryBarrierWithGroupSync();

    bool colored = false;
    uint my_color = 1;
    uint my_weight = BDA_LOAD(uint, pc.weights + idx * 4);
    
    // PackedPair layout is 80 bytes. First 16 bytes of collisions is dispatch/count header.
    uint64_t my_pair_addr = pc.collisions + 16 + idx * 80;
    uint my_a = BDA_LOAD(uint, my_pair_addr + 4);
    uint my_b = BDA_LOAD(uint, my_pair_addr + 12);

    for (int iter = 0; iter < 10; ++iter) {
        if (!colored) {
            bool is_max = true;
            
            for (uint j = 0; j < pc.total_pairs; ++j) {
                if (idx == j) continue;
                
                uint64_t other_pair_addr = pc.collisions + 16 + j * 80;
                uint other_a = BDA_LOAD(uint, other_pair_addr + 4);
                uint other_b = BDA_LOAD(uint, other_pair_addr + 12);
                
                if (my_a == other_a || my_a == other_b || my_b == other_a || my_b == other_b) {
                    uint other_color = BDA_LOAD(uint, pc.colors + j * 4);
                    if (other_color == 0 || other_color == my_color) {
                        uint other_weight = BDA_LOAD(uint, pc.weights + j * 4);
                        if (other_weight > my_weight || (other_weight == my_weight && j > idx)) {
                            is_max = false;
                            break;
                        }
                    }
                }
            }
            
            if (is_max) {
                BDA_STORE(uint, pc.colors + idx * 4, my_color);
                colored = true;
            }
        }
        
        DeviceMemoryBarrierWithGroupSync();
        
        if (!colored) {
            my_color++;
        }
    }
}
#endif // KERNEL_graph_coloring


// --- hlsl_convert_particles.txt ---
#include "../debug_utils.hlsl"

// BDA Memory Access Macros
// ------------------------------------------------------------------
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

// ------------------------------------------------------------------
// SPIR-V Atomic Intrinsics for 64-bit BDA Pointers
// ------------------------------------------------------------------
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

[[vk::ext_instruction(234)]] uint spvAtomicIAdd([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(237)]] uint spvAtomicUMin([[vk::ext_reference]] uint64_t ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(230)]] uint spvAtomicCompareExchange([[vk::ext_reference]] uint64_t ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);

struct PushConstants {
    uint64_t aosoa_particles;
    uint64_t mega_particles;
    uint64_t mega_indirect;
    uint64_t atomic_counters;
    uint mega_indirect_index;
    uint mega_particle_offset;
};

[[vk::push_constant]] PushConstants pc;

static const uint SUBGROUP_SIZE = 32;

[numthreads(128, 1, 1)]
void convert_particles(uint3 DispatchThreadID : SV_DispatchThreadID) {
    uint total_particles = BDA_LOAD(uint, pc.atomic_counters + 0);

    // Only thread 0 writes the indirect command
    if (DispatchThreadID.x == 0) {
        uint64_t cmd_offset = pc.mega_indirect + pc.mega_indirect_index * 16;
        BDA_STORE(uint, cmd_offset + 0, 4);
        BDA_STORE(uint, cmd_offset + 4, total_particles);
        BDA_STORE(uint, cmd_offset + 8, 0);
        BDA_STORE(uint, cmd_offset + 12, pc.mega_particle_offset);
    }

    uint idx = DispatchThreadID.x;
    if (idx >= total_particles) {
        return;
    }

    uint in_block = idx / SUBGROUP_SIZE;
    uint in_lane  = idx % SUBGROUP_SIZE;
    uint in_base  = in_block * 10 * SUBGROUP_SIZE + in_lane;

    float3 pos;
    pos.x = BDA_LOAD(float, pc.aosoa_particles + (in_base + 0 * SUBGROUP_SIZE) * 4);
    pos.y = BDA_LOAD(float, pc.aosoa_particles + (in_base + 1 * SUBGROUP_SIZE) * 4);
    pos.z = BDA_LOAD(float, pc.aosoa_particles + (in_base + 2 * SUBGROUP_SIZE) * 4);

    float3 vel;
    vel.x = BDA_LOAD(float, pc.aosoa_particles + (in_base + 3 * SUBGROUP_SIZE) * 4);
    vel.y = BDA_LOAD(float, pc.aosoa_particles + (in_base + 4 * SUBGROUP_SIZE) * 4);
    vel.z = BDA_LOAD(float, pc.aosoa_particles + (in_base + 5 * SUBGROUP_SIZE) * 4);

    float mass = BDA_LOAD(float, pc.aosoa_particles + (in_base + 6 * SUBGROUP_SIZE) * 4);

    uint out_idx = pc.mega_particle_offset + idx;
    uint64_t out_offset = pc.mega_particles + out_idx * 48;

    // We do not have IDs or Age from the physics simulation right now.
    // They could be added in emit_particles.comp in the future.
    BDA_STORE(uint, out_offset + 0, 0);
    BDA_STORE(uint, out_offset + 4, 0);
    BDA_STORE(uint, out_offset + 8, 0);
    BDA_STORE(uint, out_offset + 12, 0);
    BDA_STORE(float3, out_offset + 16, pos);
    BDA_STORE(float, out_offset + 28, mass);
    BDA_STORE(float3, out_offset + 32, vel);
    BDA_STORE(uint, out_offset + 44, 1);
}

// --- hlsl_barnes_hut.txt ---



#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#define SUBGROUPS_PER_WG (256 / SUBGROUP_SIZE)

// ------------------------------------------------------------------
// BDA Memory Access Macros
// ------------------------------------------------------------------
#define BDA_LOAD(T, addr) vk::RawBufferLoad<T>(addr)
#define BDA_STORE(T, addr, val) vk::RawBufferStore<T>(addr, val)

#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0

[[vk::ext_instruction(230)]] 
uint spvAtomicCompareExchange([[vk::ext_reference]] uint64_t ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);

void AtomicAddFloatBDA(uint64_t addr, float val) {
    uint old_val = BDA_LOAD(uint, addr);
    uint assumed_val;
    do {
        assumed_val = old_val;
        uint new_val = asuint(asfloat(assumed_val) + val);
        old_val = spvAtomicCompareExchange(addr, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, SPV_SEMANTICS_RELAXED, new_val, assumed_val);
    } while (assumed_val != old_val);
}

struct PushConstants {
    uint64_t particles;
    uint64_t bvh;
    uint64_t cluster_list;
    uint64_t wrenches;
    uint num_clusters;
    float dt;
    float theta;
    float G;
    float softening_sq;
    uint root_node_idx;
    uint cluster_threshold;
};

[[vk::push_constant]] PushConstants pc;

#define P_READ(addr, idx) asfloat(BDA_LOAD(uint, (addr) + (idx) * 4))

#define OFFSET_MIN_X          (0 * SUBGROUP_SIZE * 4)
#define OFFSET_MAX_X          (1 * SUBGROUP_SIZE * 4)
#define OFFSET_MIN_Y          (2 * SUBGROUP_SIZE * 4)
#define OFFSET_MAX_Y          (3 * SUBGROUP_SIZE * 4)
#define OFFSET_MIN_Z          (4 * SUBGROUP_SIZE * 4)
#define OFFSET_MAX_Z          (5 * SUBGROUP_SIZE * 4)
#define OFFSET_CHILD_INDICES  (6 * SUBGROUP_SIZE * 4)
#define OFFSET_METADATA       (7 * SUBGROUP_SIZE * 4)
#define OFFSET_MASSES         (8 * SUBGROUP_SIZE * 4)
#define OFFSET_COM_X          (9 * SUBGROUP_SIZE * 4)
#define OFFSET_COM_Y          (10 * SUBGROUP_SIZE * 4)
#define OFFSET_COM_Z          (11 * SUBGROUP_SIZE * 4)
#define OFFSET_PARTICLE_START (12 * SUBGROUP_SIZE * 4)
#define OFFSET_PARTICLE_COUNT (13 * SUBGROUP_SIZE * 4)
#define OFFSET_VALID_MASK     (14 * SUBGROUP_SIZE * 4)
#define NODE_STRIDE           (22 * SUBGROUP_SIZE * 4 + 16)

bool bvh_node_is_valid(uint2 valid_mask, uint lane_id) {
    if (lane_id < 32) return (valid_mask.x & (1u << lane_id)) != 0u;
    else return (valid_mask.y & (1u << (lane_id - 32))) != 0u;
}

bool bvh_is_leaf(uint meta) { return (meta & 0x80000000u) != 0u; }

groupshared uint shared_stacks[SUBGROUPS_PER_WG][64];
groupshared uint shared_stack_ptrs[SUBGROUPS_PER_WG];

[numthreads(256, 1, 1)]
void barnes_hut(uint3 gl_WorkGroupID : SV_GroupID,
                uint gl_LocalInvocationIndex : SV_GroupIndex)
{
    uint lane_id = WaveGetLaneIndex();
    uint subgroup_id = gl_LocalInvocationIndex / WaveGetLaneCount();
    uint cluster_job_idx = gl_WorkGroupID.x * SUBGROUPS_PER_WG + subgroup_id;
    
    if (cluster_job_idx >= pc.num_clusters) return;

    uint target_node_idx = BDA_LOAD(uint, pc.cluster_list + cluster_job_idx * 4);
    uint64_t target_node_addr = pc.bvh + target_node_idx * NODE_STRIDE;
    
    uint2 target_valid_mask = BDA_LOAD(uint2, target_node_addr + OFFSET_VALID_MASK);
    bool i_am_valid = bvh_node_is_valid(target_valid_mask, lane_id);
    uint my_p_idx = BDA_LOAD(uint, target_node_addr + OFFSET_CHILD_INDICES + lane_id * 4);

    float3 my_pos = float3(0.0, 0.0, 0.0);
    float my_mass = 0.0;
    
    if (i_am_valid) {
        uint base = (my_p_idx / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (my_p_idx % SUBGROUP_SIZE);
        my_pos = float3(P_READ(pc.particles, base), 
                        P_READ(pc.particles, base + 1 * SUBGROUP_SIZE), 
                        P_READ(pc.particles, base + 2 * SUBGROUP_SIZE));
        my_mass = P_READ(pc.particles, base + 6 * SUBGROUP_SIZE);
    }

    float3 safe_pos = i_am_valid ? my_pos : float3(0.0, 0.0, 0.0);
    float3 min_pos = WaveActiveMin(i_am_valid ? my_pos : float3(1e20, 1e20, 1e20));
    float3 max_pos = WaveActiveMax(i_am_valid ? my_pos : float3(-1e20, -1e20, -1e20));
    float3 cluster_extents = max_pos - min_pos;
    float target_size = max(cluster_extents.x, max(cluster_extents.y, cluster_extents.z));
    float sum_mass = WaveActiveSum(i_am_valid ? my_mass : 0.0);
    float3 target_com = WaveActiveSum(safe_pos * my_mass) / max(sum_mass, 1e-6);

    float3 my_acc = float3(0.0, 0.0, 0.0);
    if (lane_id == 0) { 
        shared_stacks[subgroup_id][0] = pc.root_node_idx; 
        shared_stack_ptrs[subgroup_id] = 1; 
    }

    while (true) {
        GroupMemoryBarrierWithGroupSync();
        uint stack_ptr = shared_stack_ptrs[subgroup_id]; 
        if (stack_ptr == 0) break;
        
        stack_ptr--;
        uint source_node_idx = shared_stacks[subgroup_id][stack_ptr]; 
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr;

        uint64_t source_node_addr = pc.bvh + source_node_idx * NODE_STRIDE;
        
        uint2 s_valid_mask = BDA_LOAD(uint2, source_node_addr + OFFSET_VALID_MASK);
        bool s_valid = bvh_node_is_valid(s_valid_mask, lane_id);
        
        uint s_meta = BDA_LOAD(uint, source_node_addr + OFFSET_METADATA + lane_id * 4);
        bool s_is_leaf = bvh_is_leaf(s_meta);

        float3 s_com = float3(
            asfloat(BDA_LOAD(uint, source_node_addr + OFFSET_COM_X + lane_id * 4)),
            asfloat(BDA_LOAD(uint, source_node_addr + OFFSET_COM_Y + lane_id * 4)),
            asfloat(BDA_LOAD(uint, source_node_addr + OFFSET_COM_Z + lane_id * 4))
        );
        float s_mass = asfloat(BDA_LOAD(uint, source_node_addr + OFFSET_MASSES + lane_id * 4));
        uint s_idx = BDA_LOAD(uint, source_node_addr + OFFSET_CHILD_INDICES + lane_id * 4);
        uint s_start = BDA_LOAD(uint, source_node_addr + OFFSET_PARTICLE_START + lane_id * 4);
        uint s_count = BDA_LOAD(uint, source_node_addr + OFFSET_PARTICLE_COUNT + lane_id * 4);

        float3 s_extents = float3(
            asfloat(BDA_LOAD(uint, source_node_addr + OFFSET_MAX_X + lane_id * 4)) - asfloat(BDA_LOAD(uint, source_node_addr + OFFSET_MIN_X + lane_id * 4)),
            asfloat(BDA_LOAD(uint, source_node_addr + OFFSET_MAX_Y + lane_id * 4)) - asfloat(BDA_LOAD(uint, source_node_addr + OFFSET_MIN_Y + lane_id * 4)),
            asfloat(BDA_LOAD(uint, source_node_addr + OFFSET_MAX_Z + lane_id * 4)) - asfloat(BDA_LOAD(uint, source_node_addr + OFFSET_MIN_Z + lane_id * 4))
        );
        float s_size = max(s_extents.x, max(s_extents.y, s_extents.z));

        bool pass_mac = ((s_size + target_size) / max(length(s_com - target_com), 1e-6)) < pc.theta;
        bool pass_lod_thresh = (s_count <= pc.cluster_threshold) && !((my_p_idx >= s_start) && (my_p_idx < s_start + s_count));
        bool action_accumulate = s_valid && (pass_mac || pass_lod_thresh || s_is_leaf);
        bool action_traverse = s_valid && !action_accumulate;

        uint4 acc_ballot = WaveActiveBallot(action_accumulate);
        for (uint i = 0; i < 4; i++) {
            uint mask = acc_ballot[i];
            while (mask != 0) {
                uint bit = firstbitlow(mask); 
                mask &= ~(1u << bit); 
                uint src_lane = i * 32 + bit;
                
                if (i_am_valid) {
                    float3 k_com = float3(WaveReadLaneAt(s_com.x, src_lane), WaveReadLaneAt(s_com.y, src_lane), WaveReadLaneAt(s_com.z, src_lane));
                    float k_mass = WaveReadLaneAt(s_mass, src_lane); 
                    uint k_idx = WaveReadLaneAt(s_idx, src_lane); 
                    bool k_leaf = WaveReadLaneAt(s_is_leaf, src_lane);

                    if (!(k_leaf && my_p_idx == k_idx)) {
                        float3 p_dir = k_com - my_pos; 
                        float p_dist_sq = dot(p_dir, p_dir);
                        my_acc += (p_dir / max(sqrt(p_dist_sq), 1e-6)) * ((pc.G * k_mass) / (p_dist_sq + pc.softening_sq));
                    }
                }
            }
        }

        uint prefix_count = WavePrefixCountBits(action_traverse);
        if (action_traverse) {
            shared_stacks[subgroup_id][stack_ptr + prefix_count] = s_idx;
        }
        
        uint total_trav = WaveActiveCountBits(action_traverse);
        if (lane_id == 0) {
            shared_stack_ptrs[subgroup_id] = stack_ptr + total_trav;
        }
    }

    if (i_am_valid) {
        float3 g_f = my_acc * my_mass;
        uint64_t w_addr = pc.wrenches + my_p_idx * 24;
        AtomicAddFloatBDA(w_addr + 0, g_f.x);
        AtomicAddFloatBDA(w_addr + 4, g_f.y);
        AtomicAddFloatBDA(w_addr + 8, g_f.z);
    }
}

// --- hlsl_emit_particles.txt ---


#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#ifdef KERNEL_emit_particles
struct PushConstants {
    uint64_t particles;
    uint64_t candidates;
    uint64_t bvh;
    uint64_t counter;
    uint root_index;
    uint num_candidates;
    uint2 pad;
    float3 sun_pos;
};

[[vk::push_constant]]
PushConstants pc;
#endif

#ifndef INTERSECT_RAY_AABB_DEFINED
#define INTERSECT_RAY_AABB_DEFINED
bool intersectRayAABB(float3 rO, float3 rD, float3 invD, float3 mi, float3 mx, float max_t) {
    float3 t0 = (mi - rO) * invD;
    float3 t1 = (mx - rO) * invD;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float tnear = max(max(tmin.x, tmin.y), tmin.z);
    float tfar = min(min(tmax.x, tmax.y), tmax.z);
    return tnear <= tfar && tfar > 0.0f && tnear < max_t;
}
#endif

[numthreads(128, 1, 1)]
void emit_particles(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.num_candidates) return;
    
    uint stride = 10 * SUBGROUP_SIZE;
    uint base = (gid / SUBGROUP_SIZE) * stride + (gid % SUBGROUP_SIZE);

    float pos_x = asfloat(BDA_LOAD(uint, pc.candidates + base * 4));
    float pos_y = asfloat(BDA_LOAD(uint, pc.candidates + (base + SUBGROUP_SIZE) * 4));
    float pos_z = asfloat(BDA_LOAD(uint, pc.candidates + (base + 2 * SUBGROUP_SIZE) * 4));
    float3 pos = float3(pos_x, pos_y, pos_z);
    
    float3 dir = pc.sun_pos - pos;
    float dist = length(dir);
    if (dist < 1e-5f) return;
    dir /= dist;
    float3 invDir = 1.0f / dir;

    bool occluded = false;
    uint stack[64];
    int stackPtr = 0;
    if (pc.root_index != 0xFFFFFFFFu) stack[stackPtr++] = pc.root_index;

    while(stackPtr > 0 && !occluded) {
        uint node = stack[--stackPtr];
        uint64_t node_addr = pc.bvh + node * 2832;
        uint2 valid_mask = BDA_LOAD(uint2, node_addr + 1792);

        for (uint i = 0; i < SUBGROUP_SIZE; ++i) {
            if (!bvh_node_is_valid(valid_mask, i)) continue;
            
            float mn_x = BDA_LOAD(float, node_addr + 0 + i * 4);
            float mn_y = BDA_LOAD(float, node_addr + 256 + i * 4);
            float mn_z = BDA_LOAD(float, node_addr + 512 + i * 4);
            float mx_x = BDA_LOAD(float, node_addr + 128 + i * 4);
            float mx_y = BDA_LOAD(float, node_addr + 384 + i * 4);
            float mx_z = BDA_LOAD(float, node_addr + 640 + i * 4);
            
            float3 mn = float3(mn_x, mn_y, mn_z);
            float3 mx = float3(mx_x, mx_y, mx_z);

            if (intersectRayAABB(pos + dir * 0.1f, dir, invDir, mn, mx, dist)) {
                uint meta = BDA_LOAD(uint, node_addr + 896 + i * 4);
                if (bvh_is_leaf(meta)) { 
                    occluded = true; 
                    break; 
                }
                else if (bvh_get_index(meta) != 0xFFFFFFFFu) {
                    stack[stackPtr++] = bvh_get_index(meta);
                }
            }
        }
    }

    if (!occluded) {
        uint out_idx = spvAtomicIAdd(pc.counter + 0, SPV_SCOPE_DEVICE, SPV_SEMANTICS_RELAXED, 1);
        uint out_base = (out_idx / SUBGROUP_SIZE) * stride + (out_idx % SUBGROUP_SIZE);
        for (int i = 0; i < 10; ++i) {
            uint val = BDA_LOAD(uint, pc.candidates + (base + i * SUBGROUP_SIZE) * 4);
            BDA_STORE(uint, pc.particles + (out_base + i * SUBGROUP_SIZE) * 4, val);
        }
    }
}


