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
    uint old_val = buf[idx]; uint assumed_val;
    do {
        assumed_val = old_val;
        uint new_val = asuint(asfloat(assumed_val) + val);
        InterlockedCompareExchange(buf[idx], assumed_val, new_val, old_val);
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
    uint valid; uint entity_a; uint prim_a; uint entity_b;
    uint prim_b; uint pad1; uint2 bda_a;
    uint2 bda_b; uint2 frame_bda;
    float toi; float penetration_depth; uint pad2; uint pad3;
    float norm_x; float norm_y; float norm_z; uint pad4;
    float pt_x; float pt_y; float pt_z; uint pad5;
};

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
struct EntityHeader { uint type; uint pad[3]; };
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
#ifdef KERNEL_motion_refit
struct PushConstants {
    uint2 bvh;
    uint2 depth_indices;
    uint total_nodes_at_depth;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(256, 1, 1)]
void motion_refit(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint global_id = gl_GlobalInvocationID.x; if (global_id >= pc.total_nodes_at_depth) return;

    vk::BufferPointer<DepthIndicesType> depth_indices = cast_u2_depth(pc.depth_indices);
    vk::BufferPointer<MultiBvhNode> bvh = cast_u2_bvh(pc.bvh);

    uint node_idx = depth_indices[0].indices[global_id + 4];
    for (uint i = 0; i < 2; ++i) {
        uint child = bvh[node_idx].child_indices[i];
        if (bvh_is_leaf(bvh[node_idx].metadata[i])) {
            bvh[node_idx].min_x[i] = bvh[child].min_x[0]; bvh[node_idx].max_x[i] = bvh[child].max_x[0];
            bvh[node_idx].min_y[i] = bvh[child].min_y[0]; bvh[node_idx].max_y[i] = bvh[child].max_y[0];
            bvh[node_idx].min_z[i] = bvh[child].min_z[0]; bvh[node_idx].max_z[i] = bvh[child].max_z[0];
        } else {
            bvh[node_idx].min_x[i] = min(bvh[child].min_x[0], bvh[child].min_x[1]);
            bvh[node_idx].max_x[i] = max(bvh[child].max_x[0], bvh[child].max_x[1]);
            bvh[node_idx].min_y[i] = min(bvh[child].min_y[0], bvh[child].min_y[1]);
            bvh[node_idx].max_y[i] = max(bvh[child].max_y[0], bvh[child].max_y[1]);
            bvh[node_idx].min_z[i] = min(bvh[child].min_z[0], bvh[child].min_z[1]);
            bvh[node_idx].max_z[i] = max(bvh[child].max_z[0], bvh[child].max_z[1]);
        }
    }
}
#endif // KERNEL_motion_refit


// --- ccd ---
#ifdef KERNEL_ccd
struct PushConstants {
    uint2 particle_bvh;
    uint2 output_list;
    uint2 particles;
    uint root_index;
    uint total_particles;
    float particle_radius;
    float dt;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void ccd(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint idx = gl_GlobalInvocationID.x; if (idx >= pc.total_particles) return;

    vk::BufferPointer<float> part_f = cast_u2_f(pc.particles);
    vk::BufferPointer<MultiBvhNode> bvh = cast_u2_bvh(pc.particle_bvh);
    vk::BufferPointer<SparseCollisionsType> out_list = cast_u2_sparse(pc.output_list);

    uint my_prim_id = idx;
    uint bA = (my_prim_id / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (my_prim_id % SUBGROUP_SIZE);

    float3 my_center = float3(part_f[bA + 0], part_f[bA + 1 * SUBGROUP_SIZE], part_f[bA + 2 * SUBGROUP_SIZE]);
    float3 my_vel = float3(part_f[bA + 3 * SUBGROUP_SIZE], part_f[bA + 4 * SUBGROUP_SIZE], part_f[bA + 5 * SUBGROUP_SIZE]);
    float3 p1 = my_center + my_vel * pc.dt;

    AABB my_aabb;
    my_aabb.minBounds = min(my_center - (float3)pc.particle_radius, p1 - (float3)pc.particle_radius);
    my_aabb.maxBounds = max(my_center + (float3)pc.particle_radius, p1 + (float3)pc.particle_radius);

    uint stack[64]; int stackPtr = 0; if (pc.root_index != 0xFFFFFFFFu) stack[stackPtr++] = pc.root_index;
    uint collisions_found = 0;

    while (stackPtr > 0) {
        uint node_idx = stack[--stackPtr];
        for (uint i = 0; i < SUBGROUP_SIZE; ++i) {
            if (!bvh_node_is_valid(bvh[node_idx].valid_mask, i)) continue;

            AABB bound;
            bound.minBounds = float3(bvh[node_idx].min_x[i], bvh[node_idx].min_y[i], bvh[node_idx].min_z[i]);
            bound.maxBounds = float3(bvh[node_idx].max_x[i], bvh[node_idx].max_y[i], bvh[node_idx].max_z[i]);

            if (intersectAABB(my_aabb.minBounds, my_aabb.maxBounds, bound.minBounds, bound.maxBounds)) {
                uint meta = bvh[node_idx].metadata[i];
                uint offset = bvh_get_index(meta);

                if (bvh_is_leaf(meta)) {
                    if (my_prim_id < offset) {
                        float toi = 0.0, depth = 0.0; float3 normal = (float3)0.0, contact = (float3)0.0;
                        uint bB = (offset / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (offset % SUBGROUP_SIZE);
                        float3 other_vel = float3(part_f[bB + 3 * SUBGROUP_SIZE], part_f[bB + 4 * SUBGROUP_SIZE], part_f[bB + 5 * SUBGROUP_SIZE]) * pc.dt;
                        float4x4 transA = float4x4(1.0,0,0,0, 0,1.0,0,0, 0,0,1.0,0, my_center.x, my_center.y, my_center.z, 1.0);
                        float4x4 transB = float4x4(1.0,0,0,0, 0,1.0,0,0, 0,0,1.0,0, bvh[node_idx].com_x[i], bvh[node_idx].com_y[i], bvh[node_idx].com_z[i], 1.0);

                        if (compute_toi_generic(0, float3(pc.particle_radius,0,0), transA, my_vel * pc.dt, 0, float3(pc.particle_radius,0,0), transB, other_vel, 1e-3, 10, toi, normal, contact, depth)) {
                            if (collisions_found < 16) {
                                collisions_found++;
                                uint outIdx; InterlockedAdd(out_list[0].count, 1, outIdx);
                                SparseCollisionData data;
                                data.valid = 1; data.entity_a = 0xFFFFFFFFu; data.prim_a = my_prim_id;
                                data.entity_b = 0xFFFFFFFFu; data.prim_b = offset;
                                data.toi = toi;
                                data.norm_x = normal.x; data.norm_y = normal.y; data.norm_z = normal.z;
                                data.pt_x = contact.x; data.pt_y = contact.y; data.pt_z = contact.z;
                                data.penetration_depth = depth;
                                out_list[0].pairs[outIdx] = data;
                            }
                        }
                    }
                } else if (offset != 0xFFFFFFFFu) stack[stackPtr++] = offset;
            }
        }
    }
}
#endif // KERNEL_ccd


// --- lbvh_build ---
#ifdef KERNEL_lbvh_build
struct PushConstants {
    uint2 bvh;
    uint2 sorted_morton;
    uint2 counters;
    uint2 particles;
    uint num_primitives;
    float particle_radius;
    float dt;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

int common_prefix(uint n, int i, int j, vk::BufferPointer<uint2> morton) {
    if (j < 0 || j >= n) return -1;
    uint key1 = morton[i].x; uint key2 = morton[j].x;
    if (key1 == key2) {
        uint idx1 = morton[i].y; uint idx2 = morton[j].y;
        return 32 + (31 - firstbithigh(idx1 ^ idx2));
    }
    return 31 - firstbithigh(key1 ^ key2);
}

float2 determine_range(uint n, int i, vk::BufferPointer<uint2> morton) {
    int d = sign((int)common_prefix(n, i, i + 1, morton) - (int)common_prefix(n, i, i - 1, morton));
    int min_p = common_prefix(n, i, i - d, morton), l_max = 2;
    while (common_prefix(n, i, i + l_max * d, morton) > min_p) l_max *= 2;
    int l = 0, t = l_max / 2;
    while (t >= 1) { if (common_prefix(n, i, i + (l + t) * d, morton) > min_p) l += t; t /= 2; }
    return float2(min(i, i + l * d), max(i, i + l * d));
}

int find_split(uint n, int first, int last, vk::BufferPointer<uint2> morton) {
    int common_node = common_prefix(n, first, last, morton), split = first, step = last - first;
    do { step = (step + 1) >> 1; int new_split = split + step; if (new_split < last && common_prefix(n, first, new_split, morton) > common_node) split = new_split; } while (step > 1);
    return split;
}

[numthreads(128, 1, 1)]
void lbvh_build(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint idx = gl_GlobalInvocationID.x, n = pc.num_primitives;
    if (idx >= n) return;

    vk::BufferPointer<MultiBvhNode> bvh = cast_u2_bvh(pc.bvh);
    vk::BufferPointer<uint2> morton = cast_u2_u2(pc.sorted_morton);
    vk::BufferPointer<uint> counters = cast_u2_u(pc.counters);
    vk::BufferPointer<float> part_f = cast_u2_f(pc.particles);

    uint num_internal_nodes = n - 1;

    if (idx < num_internal_nodes) {
        float2 range = determine_range(n, int(idx), morton);
        int split = find_split(n, int(range.x), int(range.y), morton);
        uint left_child = (split == int(range.x)) ? (num_internal_nodes + split) : uint(split);
        uint right_child = (split + 1 == int(range.y)) ? (num_internal_nodes + split + 1) : uint(split + 1);

        bvh[idx].child_indices[0] = left_child; bvh[idx].child_indices[1] = right_child;
        bvh[idx].valid_mask = uint2(3u, 0u);
        bvh[left_child].parent_idx = idx; bvh[right_child].parent_idx = idx;
    }

    uint leaf_idx = num_internal_nodes + idx, p_id = morton[idx].y;
    uint base = (p_id / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (p_id % SUBGROUP_SIZE);

    float3 pos = float3(part_f[base+0], part_f[base+1*SUBGROUP_SIZE], part_f[base+2*SUBGROUP_SIZE]);
    float3 vel = float3(part_f[base+3*SUBGROUP_SIZE], part_f[base+4*SUBGROUP_SIZE], part_f[base+5*SUBGROUP_SIZE]);
    float mass = part_f[base+6*SUBGROUP_SIZE], r = pc.particle_radius;

    float3 p1 = pos + vel * pc.dt;
    float3 l_min = min(pos - (float3)r, p1 - (float3)r), l_max = max(pos + (float3)r, p1 + (float3)r);

    uint current = bvh[leaf_idx].parent_idx;
    uint is_right = (bvh[current].child_indices[1] == leaf_idx) ? 1 : 0;

    bvh[current].min_x[is_right] = l_min.x; bvh[current].max_x[is_right] = l_max.x;
    bvh[current].min_y[is_right] = l_min.y; bvh[current].max_y[is_right] = l_max.y;
    bvh[current].min_z[is_right] = l_min.z; bvh[current].max_z[is_right] = l_max.z;
    bvh[current].masses[is_right] = mass;
    bvh[current].com_x[is_right] = pos.x; bvh[current].com_y[is_right] = pos.y; bvh[current].com_z[is_right] = pos.z;
    bvh[current].metadata[is_right] = bvh_pack_metadata(true, BVH_FRAME_MICRO, BVH_SHAPE_AABB, p_id);

    DeviceMemoryBarrier();

    while (current != 0xFFFFFFFFu) {
        uint _old_atomic; InterlockedAdd(counters[current], 1, _old_atomic); if (_old_atomic == 0) break;

        float3 c_l_min = float3(bvh[current].min_x[0], bvh[current].min_y[0], bvh[current].min_z[0]);
        float3 c_l_max = float3(bvh[current].max_x[0], bvh[current].max_y[0], bvh[current].max_z[0]);
        float l_m = bvh[current].masses[0];
        float3 l_com = float3(bvh[current].com_x[0], bvh[current].com_y[0], bvh[current].com_z[0]);

        float3 c_r_min = float3(bvh[current].min_x[1], bvh[current].min_y[1], bvh[current].min_z[1]);
        float3 c_r_max = float3(bvh[current].max_x[1], bvh[current].max_y[1], bvh[current].max_z[1]);
        float r_m = bvh[current].masses[1];
        float3 r_com = float3(bvh[current].com_x[1], bvh[current].com_y[1], bvh[current].com_z[1]);

        float3 c_min = min(c_l_min, c_r_min), c_max = max(c_l_max, c_r_max);
        float c_mass = l_m + r_m;
        float3 c_com = c_mass > 0.0 ? (l_com * l_m + r_com * r_m) / c_mass : (l_com + r_com) * 0.5;

        uint parent = bvh[current].parent_idx;
        if (parent != 0xFFFFFFFFu) {
            uint is_r = (bvh[parent].child_indices[1] == current) ? 1 : 0;
            bvh[parent].min_x[is_r] = c_min.x; bvh[parent].max_x[is_r] = c_max.x;
            bvh[parent].min_y[is_r] = c_min.y; bvh[parent].max_y[is_r] = c_max.y;
            bvh[parent].min_z[is_r] = c_min.z; bvh[parent].max_z[is_r] = c_max.z;
            bvh[parent].masses[is_r] = c_mass;
            bvh[parent].com_x[is_r] = c_com.x; bvh[parent].com_y[is_r] = c_com.y; bvh[parent].com_z[is_r] = c_com.z;
            bvh[parent].metadata[is_r] = bvh_pack_metadata(false, BVH_FRAME_MICRO, BVH_SHAPE_AABB, current);
        }
        DeviceMemoryBarrier();
        current = parent;
    }
}
#endif // KERNEL_lbvh_build


// --- lcp_solver ---
#ifdef KERNEL_lcp_solver
struct PushConstants {
    uint2 particles;
    uint2 collisions;
    uint2 outputs;
    uint total_clusters;
    uint2 rigid_bodies;
    float dt;
    float restitution;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

#define MAX_BODIES_PER_ISLAND 32
groupshared uint shared_v_x[MAX_BODIES_PER_ISLAND]; groupshared uint shared_v_y[MAX_BODIES_PER_ISLAND]; groupshared uint shared_v_z[MAX_BODIES_PER_ISLAND];
groupshared uint shared_w_x[MAX_BODIES_PER_ISLAND]; groupshared uint shared_w_y[MAX_BODIES_PER_ISLAND]; groupshared uint shared_w_z[MAX_BODIES_PER_ISLAND];
groupshared float accumulated_normal[128]; groupshared float accumulated_t1[128]; groupshared float accumulated_t2[128];

void generate_tangents_local(float3 normal, out float3 t1, out float3 t2) {
    if (abs(normal.x) >= 0.57735) t1 = normalize(float3(normal.y, -normal.x, 0.0));
    else t1 = normalize(float3(0.0, normal.z, -normal.y));
    t2 = cross(normal, t1);
}

float compute_effective_mass_local(float3 dir, float3 rA, float3 rB, float invMA, float invMB, float3 invIA, float3 invIB, float4 qA, float4 qB) {
    float3 I_crossA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, dir)));
    float3 I_crossB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, dir)));
    return 1.0 / max(invMA + invMB + dot(I_crossA, cross(rA, dir)) + dot(I_crossB, cross(rB, dir)), 1e-6);
}

[numthreads(128, 1, 1)]
void lcp_solver(uint3 gl_LocalInvocationID : SV_GroupThreadID, uint3 gl_WorkGroupID : SV_GroupID) {
    vk::BufferPointer<PackedCollisionsType> cols = cast_u2_packed(pc.collisions);
    vk::BufferPointer<RigidBody> rbs = cast_u2_rb(pc.rigid_bodies);
    vk::BufferPointer<float> part_f = cast_u2_f(pc.particles);
    vk::BufferPointer<float> out_f = cast_u2_f(pc.outputs);

    uint local_id = gl_LocalInvocationID.x, contact_idx = gl_WorkGroupID.x * 128 + local_id;
    bool valid = (contact_idx < cols[0].count);

    accumulated_normal[local_id] = 0.0; accumulated_t1[local_id] = 0.0; accumulated_t2[local_id] = 0.0;

    if (local_id < MAX_BODIES_PER_ISLAND) {
        RigidBody rb = rbs[local_id];
        shared_v_x[local_id] = asuint(rb.lin_vel_x); shared_v_y[local_id] = asuint(rb.lin_vel_y); shared_v_z[local_id] = asuint(rb.lin_vel_z);
        shared_w_x[local_id] = asuint(rb.ang_vel_x); shared_w_y[local_id] = asuint(rb.ang_vel_y); shared_w_z[local_id] = asuint(rb.ang_vel_z);
    }
    GroupMemoryBarrierWithGroupSync();
    if (!valid) return;

    PackedPair pair = cols[0].pairs[contact_idx];
    bool is_partA = (pair.a.entity_id == 0xFFFFFFFFu), is_partB = (pair.b.entity_id == 0xFFFFFFFFu);
    uint idA = pair.a.primitive_index, idB = pair.b.primitive_index;

    float invMA = 0.0, invMB = 0.0; float3 invIA = (float3)0.0, invIB = (float3)0.0; float4 qA = float4(0,0,0,1), qB = float4(0,0,0,1);
    float3 posA = (float3)0.0, posB = (float3)0.0, vA_init = (float3)0.0, wA_init = (float3)0.0, vB_init = (float3)0.0, wB_init = (float3)0.0;

    if (is_partA) {
        uint base = (idA / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (idA % SUBGROUP_SIZE);
        posA = float3(part_f[base+0], part_f[base+1*SUBGROUP_SIZE], part_f[base+2*SUBGROUP_SIZE]);
        vA_init = float3(part_f[base+3*SUBGROUP_SIZE], part_f[base+4*SUBGROUP_SIZE], part_f[base+5*SUBGROUP_SIZE]);
        float mass = part_f[base+6*SUBGROUP_SIZE]; invMA = (mass > 0.0) ? 1.0 / mass : 0.0;
    } else {
        RigidBody rbA = rbs[idA]; invMA = rbA.mass > 0.0 ? 1.0 / rbA.mass : 0.0;
        invIA = float3(rbA.inv_inertia_x, rbA.inv_inertia_y, rbA.inv_inertia_z); qA = float4(rbA.orient_x, rbA.orient_y, rbA.orient_z, rbA.orient_w);
        posA = float3(rbA.pos_x, rbA.pos_y, rbA.pos_z); vA_init = float3(rbA.lin_vel_x, rbA.lin_vel_y, rbA.lin_vel_z); wA_init = float3(rbA.ang_vel_x, rbA.ang_vel_y, rbA.ang_vel_z);
    }

    if (is_partB) {
        uint base = (idB / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (idB % SUBGROUP_SIZE);
        posB = float3(part_f[base+0], part_f[base+1*SUBGROUP_SIZE], part_f[base+2*SUBGROUP_SIZE]);
        vB_init = float3(part_f[base+3*SUBGROUP_SIZE], part_f[base+4*SUBGROUP_SIZE], part_f[base+5*SUBGROUP_SIZE]);
        float mass = part_f[base+6*SUBGROUP_SIZE]; invMB = (mass > 0.0) ? 1.0 / mass : 0.0;
    } else {
        RigidBody rbB = rbs[idB]; invMB = rbB.mass > 0.0 ? 1.0 / rbB.mass : 0.0;
        invIB = float3(rbB.inv_inertia_x, rbB.inv_inertia_y, rbB.inv_inertia_z); qB = float4(rbB.orient_x, rbB.orient_y, rbB.orient_z, rbB.orient_w);
        posB = float3(rbB.pos_x, rbB.pos_y, rbB.pos_z); vB_init = float3(rbB.lin_vel_x, rbB.lin_vel_y, rbB.lin_vel_z); wB_init = float3(rbB.ang_vel_x, rbB.ang_vel_y, rbB.ang_vel_z);
    }

    float3 n = float3(pair.norm_x, pair.norm_y, pair.norm_z), t1, t2; generate_tangents_local(n, t1, t2);
    float3 rA = float3(pair.pt_x, pair.pt_y, pair.pt_z) - posA, rB = float3(pair.pt_x, pair.pt_y, pair.pt_z) - posB;
    float eff_m_n = compute_effective_mass_local(n, rA, rB, invMA, invMB, invIA, invIB, qA, qB);
    float eff_m_t1 = compute_effective_mass_local(t1, rA, rB, invMA, invMB, invIA, invIB, qA, qB);
    float eff_m_t2 = compute_effective_mass_local(t2, rA, rB, invMA, invMB, invIA, invIB, qA, qB);

    float3 v_rel_init = (vB_init + cross(wB_init, rB)) - (vA_init + cross(wA_init, rA));
    float bounce = dot(v_rel_init, n) < -0.1 ? -pc.restitution * dot(v_rel_init, n) : 0.0;
    float target_v_n = bounce + ((0.2 / max(pc.dt, 1e-6)) * max(pair.penetration_depth - 0.01, 0.0));

    for (int iter = 0; iter < 20; ++iter) {
        GroupMemoryBarrierWithGroupSync();

        float3 vA = vA_init, wA = wA_init, vB = vB_init, wB = wB_init;
        if (!is_partA && idA < MAX_BODIES_PER_ISLAND) { vA = float3(asfloat(shared_v_x[idA]), asfloat(shared_v_y[idA]), asfloat(shared_v_z[idA])); wA = float3(asfloat(shared_w_x[idA]), asfloat(shared_w_y[idA]), asfloat(shared_w_z[idA])); }
        if (!is_partB && idB < MAX_BODIES_PER_ISLAND) { vB = float3(asfloat(shared_v_x[idB]), asfloat(shared_v_y[idB]), asfloat(shared_v_z[idB])); wB = float3(asfloat(shared_w_x[idB]), asfloat(shared_w_y[idB]), asfloat(shared_w_z[idB])); }

        float3 v_rel = (vB + cross(wB, rB)) - (vA + cross(wA, rA));
        float jn_delta = eff_m_n * (-dot(v_rel, n) + target_v_n), old_jn = accumulated_normal[local_id], new_jn = max(old_jn + jn_delta, 0.0);
        jn_delta = new_jn - old_jn; accumulated_normal[local_id] = new_jn;
        float3 P_n = jn_delta * n;

        if (!is_partA && invMA > 0.0 && idA < MAX_BODIES_PER_ISLAND) {
            SHARED_ATOMIC_ADD_FLOAT(shared_v_x[idA], -P_n.x * invMA); SHARED_ATOMIC_ADD_FLOAT(shared_v_y[idA], -P_n.y * invMA); SHARED_ATOMIC_ADD_FLOAT(shared_v_z[idA], -P_n.z * invMA);
            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -P_n)));
            SHARED_ATOMIC_ADD_FLOAT(shared_w_x[idA], dwA.x); SHARED_ATOMIC_ADD_FLOAT(shared_w_y[idA], dwA.y); SHARED_ATOMIC_ADD_FLOAT(shared_w_z[idA], dwA.z);
        }
        if (!is_partB && invMB > 0.0 && idB < MAX_BODIES_PER_ISLAND) {
            SHARED_ATOMIC_ADD_FLOAT(shared_v_x[idB], P_n.x * invMB); SHARED_ATOMIC_ADD_FLOAT(shared_v_y[idB], P_n.y * invMB); SHARED_ATOMIC_ADD_FLOAT(shared_v_z[idB], P_n.z * invMB);
            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, P_n)));
            SHARED_ATOMIC_ADD_FLOAT(shared_w_x[idB], dwB.x); SHARED_ATOMIC_ADD_FLOAT(shared_w_y[idB], dwB.y); SHARED_ATOMIC_ADD_FLOAT(shared_w_z[idB], dwB.z);
        }

        GroupMemoryBarrierWithGroupSync();

        if (!is_partA && idA < MAX_BODIES_PER_ISLAND) { vA = float3(asfloat(shared_v_x[idA]), asfloat(shared_v_y[idA]), asfloat(shared_v_z[idA])); wA = float3(asfloat(shared_w_x[idA]), asfloat(shared_w_y[idA]), asfloat(shared_w_z[idA])); }
        if (!is_partB && idB < MAX_BODIES_PER_ISLAND) { vB = float3(asfloat(shared_v_x[idB]), asfloat(shared_v_y[idB]), asfloat(shared_v_z[idB])); wB = float3(asfloat(shared_w_x[idB]), asfloat(shared_w_y[idB]), asfloat(shared_w_z[idB])); }
        v_rel = (vB + cross(wB, rB)) - (vA + cross(wA, rA));

        float max_fric = 0.5 * accumulated_normal[local_id];
        float jt1_delta = eff_m_t1 * (-dot(v_rel, t1));
        float old_jt1 = accumulated_t1[local_id]; float new_jt1 = clamp(old_jt1 + jt1_delta, -max_fric, max_fric);
        jt1_delta = new_jt1 - old_jt1; accumulated_t1[local_id] = new_jt1;

        float jt2_delta = eff_m_t2 * (-dot(v_rel, t2));
        float old_jt2 = accumulated_t2[local_id]; float new_jt2 = clamp(old_jt2 + jt2_delta, -max_fric, max_fric);
        jt2_delta = new_jt2 - old_jt2; accumulated_t2[local_id] = new_jt2;

        float3 P_t = jt1_delta * t1 + jt2_delta * t2;

        if (!is_partA && invMA > 0.0 && idA < MAX_BODIES_PER_ISLAND) {
            SHARED_ATOMIC_ADD_FLOAT(shared_v_x[idA], -P_t.x * invMA); SHARED_ATOMIC_ADD_FLOAT(shared_v_y[idA], -P_t.y * invMA); SHARED_ATOMIC_ADD_FLOAT(shared_v_z[idA], -P_t.z * invMA);
            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -P_t)));
            SHARED_ATOMIC_ADD_FLOAT(shared_w_x[idA], dwA.x); SHARED_ATOMIC_ADD_FLOAT(shared_w_y[idA], dwA.y); SHARED_ATOMIC_ADD_FLOAT(shared_w_z[idA], dwA.z);
        }
        if (!is_partB && invMB > 0.0 && idB < MAX_BODIES_PER_ISLAND) {
            SHARED_ATOMIC_ADD_FLOAT(shared_v_x[idB], P_t.x * invMB); SHARED_ATOMIC_ADD_FLOAT(shared_v_y[idB], P_t.y * invMB); SHARED_ATOMIC_ADD_FLOAT(shared_v_z[idB], P_t.z * invMB);
            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, P_t)));
            SHARED_ATOMIC_ADD_FLOAT(shared_w_x[idB], dwB.x); SHARED_ATOMIC_ADD_FLOAT(shared_w_y[idB], dwB.y); SHARED_ATOMIC_ADD_FLOAT(shared_w_z[idB], dwB.z);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    float3 out_impulse = accumulated_normal[local_id] * n + accumulated_t1[local_id] * t1 + accumulated_t2[local_id] * t2;
    out_f[contact_idx * 4 + 0] = out_impulse.x;
    out_f[contact_idx * 4 + 1] = out_impulse.y;
    out_f[contact_idx * 4 + 2] = out_impulse.z;
}
#endif // KERNEL_lcp_solver


// --- bp_classify ---
#ifdef KERNEL_bp_classify
struct PushConstants { uint2 raw_pairs; uint2 out_rb_rb; uint2 out_rb_ps; uint2 out_ps_ps; uint max_pairs; uint num_rigid_bodies; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(256, 1, 1)]
void bp_classify(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    vk::BufferPointer<PairBufferType> raw_pairs = cast_u2_pair(pc.raw_pairs);
    uint id = gl_GlobalInvocationID.x;
    if (id >= raw_pairs[0].count) return;

    uint2 pair = raw_pairs[0].pairs[id];
    uint ent_A = pair.x, ent_B = pair.y;

    uint type_A = (ent_A < pc.num_rigid_bodies) ? TYPE_RIGID_BODY : TYPE_PARTICLE_SYSTEM;
    uint type_B = (ent_B < pc.num_rigid_bodies) ? TYPE_RIGID_BODY : TYPE_PARTICLE_SYSTEM;

    if (type_A > type_B) { uint temp = ent_A; ent_A = ent_B; ent_B = temp; temp = type_A; type_A = type_B; type_B = temp; }

    if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_PARTICLE_SYSTEM) {
        if (pc.out_ps_ps.x != 0 || pc.out_ps_ps.y != 0) { vk::BufferPointer<PairBufferType> b = cast_u2_pair(pc.out_ps_ps); uint out_idx; InterlockedAdd(b[0].count, 1, out_idx); if (out_idx < pc.max_pairs) b[0].pairs[out_idx] = uint2(ent_A, ent_B); }
    } else if (type_A == TYPE_RIGID_BODY && type_B == TYPE_PARTICLE_SYSTEM) {
        if (pc.out_rb_ps.x != 0 || pc.out_rb_ps.y != 0) { vk::BufferPointer<PairBufferType> b = cast_u2_pair(pc.out_rb_ps); uint out_idx; InterlockedAdd(b[0].count, 1, out_idx); if (out_idx < pc.max_pairs) b[0].pairs[out_idx] = uint2(ent_A, ent_B); }
    } else if (type_A == TYPE_RIGID_BODY && type_B == TYPE_RIGID_BODY) {
        if (pc.out_rb_rb.x != 0 || pc.out_rb_rb.y != 0) { vk::BufferPointer<PairBufferType> b = cast_u2_pair(pc.out_rb_rb); uint out_idx; InterlockedAdd(b[0].count, 1, out_idx); if (out_idx < pc.max_pairs) b[0].pairs[out_idx] = uint2(ent_A, ent_B); }
    }
}
#endif


// --- radix_sort ---
#ifdef KERNEL_radix_sort
struct PushConstants { uint2 input_keys; uint2 output_keys; uint2 histograms; uint num_particles; uint shift; uint stage; uint num_blocks; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

#define STAGE_COUNT   0
#define STAGE_SCAN    1
#define STAGE_SCATTER 2
#define RADIX 16
#define ELEMENTS_PER_BLOCK 4096

groupshared uint s_counts[RADIX]; groupshared uint s_offsets[RADIX]; groupshared uint s_sg_counts[64]; groupshared uint s_bin_sums[RADIX];

[numthreads(256, 1, 1)]
void radix_sort(uint3 gl_LocalInvocationID : SV_GroupThreadID, uint3 gl_WorkGroupID : SV_GroupID, uint gl_LocalInvocationIndex : SV_GroupIndex) {
    uint gl_SubgroupID = gl_LocalInvocationIndex / WaveGetLaneCount(), gl_NumSubgroups = (256 + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    uint lid = gl_LocalInvocationID.x, wid = gl_WorkGroupID.x, sg_id = WaveGetLaneIndex();

    vk::BufferPointer<uint2> in_k = cast_u2_u2(pc.input_keys);
    vk::BufferPointer<uint2> out_k = cast_u2_u2(pc.output_keys);
    vk::BufferPointer<uint> hist = cast_u2_u(pc.histograms);

    if (pc.stage == STAGE_COUNT) {
        if (lid < RADIX) s_counts[lid] = 0; GroupMemoryBarrierWithGroupSync();
        uint block_start = wid * ELEMENTS_PER_BLOCK, block_end = min(block_start + ELEMENTS_PER_BLOCK, pc.num_particles);
        for (uint i = block_start + lid; i < block_end; i += 256) { uint key = (in_k[i].x >> pc.shift) & 0xFu; InterlockedAdd(s_counts[key], 1); }
        GroupMemoryBarrierWithGroupSync();
        if (lid < RADIX) hist[lid * pc.num_blocks + wid] = s_counts[lid];
    }
    else if (pc.stage == STAGE_SCAN) {
        if (lid < RADIX) { uint bin_sum = 0; for (uint w = 0; w < pc.num_blocks; ++w) bin_sum += hist[lid * pc.num_blocks + w]; s_bin_sums[lid] = bin_sum; }
        GroupMemoryBarrierWithGroupSync();
        if (lid == 0) { uint global_offset = 0; for (uint i = 0; i < RADIX; ++i) { uint val = s_bin_sums[i]; s_bin_sums[i] = global_offset; global_offset += val; } }
        GroupMemoryBarrierWithGroupSync();
        if (lid < RADIX) {
            uint running_offset = s_bin_sums[lid];
            for (uint w = 0; w < pc.num_blocks; ++w) { uint val = hist[lid * pc.num_blocks + w]; hist[lid * pc.num_blocks + w] = running_offset; running_offset += val; }
        }
    }
    else if (pc.stage == STAGE_SCATTER) {
        if (lid < RADIX) s_offsets[lid] = hist[lid * pc.num_blocks + wid]; GroupMemoryBarrierWithGroupSync();
        uint block_start = wid * ELEMENTS_PER_BLOCK, block_end = min(block_start + ELEMENTS_PER_BLOCK, pc.num_particles);

        for (uint chunk_start = block_start; chunk_start < block_end; chunk_start += 256) {
            uint i = chunk_start + lid; bool valid = (i < block_end); uint2 raw_key = valid ? in_k[i] : uint2(0xFFFFFFFFu, 0);
            uint my_key = valid ? ((raw_key.x >> pc.shift) & 0xFu) : 0xFFFFFFFFu, local_offset = 0, my_global_base = 0;

            for (uint b = 0; b < RADIX; ++b) {
                bool match = (my_key == b); uint4 ballot = WaveActiveBallot(match); uint sg_match_count = get_ballot_count(ballot), my_sg_offset = get_ballot_prefix(ballot, WaveGetLaneIndex());
                if (sg_id == 0) s_sg_counts[gl_SubgroupID] = sg_match_count; GroupMemoryBarrierWithGroupSync();
                if (lid == 0) { uint sum = 0; for (uint sg = 0; sg < gl_NumSubgroups; ++sg) { uint c = s_sg_counts[sg]; s_sg_counts[sg] = sum; sum += c; } s_counts[b] = sum; }
                GroupMemoryBarrierWithGroupSync();
                if (match) { local_offset = s_sg_counts[gl_SubgroupID] + my_sg_offset; my_global_base = s_offsets[b]; }
                if (lid == 0) s_offsets[b] += s_counts[b]; GroupMemoryBarrierWithGroupSync();
            }
            if (valid) out_k[my_global_base + local_offset] = raw_key;
        }
    }
}
#endif


// --- stream_compact ---
#ifdef KERNEL_stream_compact
struct PushConstants { uint2 sparse_in; uint2 packed_out; uint total_elements; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void stream_compact(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    vk::BufferPointer<SparseCollisionsType> s_in = cast_u2_sparse(pc.sparse_in);
    vk::BufferPointer<PackedCollisionsType> p_out = cast_u2_packed(pc.packed_out);

    uint id = gl_GlobalInvocationID.x, in_count = s_in[0].count;

    if (id == 0) {
        p_out[0].count = in_count;
        p_out[0].dispatch_x = (in_count + 127) / 128; p_out[0].dispatch_y = 1; p_out[0].dispatch_z = 1;
    }
    if (id < in_count) {
        SparseCollisionData in_data = s_in[0].pairs[id]; PackedPair p;
        p.a.entity_id = in_data.entity_a; p.a.primitive_index = in_data.prim_a; p.b.entity_id = in_data.entity_b; p.b.primitive_index = in_data.prim_b;
        p.toi = in_data.toi; p.norm_x = in_data.norm_x; p.norm_y = in_data.norm_y; p.norm_z = in_data.norm_z;
        p.pt_x = in_data.pt_x; p.pt_y = in_data.pt_y; p.pt_z = in_data.pt_z; p.penetration_depth = in_data.penetration_depth;
        p_out[0].pairs[id] = p;
    }
}
#endif


// --- narrow_ccd ---
#ifdef KERNEL_narrow_ccd
struct PushConstants {
    uint2 scene_entities; uint2 output_list; uint2 cross_output_list; uint2 particles;
    uint2 pair_buffer; uint2 cross_pair_buffer; uint2 lca_entities; float dt; float particle_radius; uint space_type;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(256, 1, 1)]
void narrow_ccd(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint pair_idx = gl_GlobalInvocationID.x, idA, idB, lca_id; bool is_partA = false;

    vk::BufferPointer<CrossPairBufferType> cp_buf = cast_u2_cpair(pc.cross_pair_buffer);
    vk::BufferPointer<PairBufferType> p_buf = cast_u2_pair(pc.pair_buffer);
    vk::BufferPointer<RigidBody> rbs = cast_u2_rb(pc.scene_entities);
    vk::BufferPointer<LcaEntity> lcas = cast_u2_lca(pc.lca_entities);

    if (pc.space_type == 1) {
        if (pair_idx >= cp_buf[0].count) return;
        CrossPair pair = cp_buf[0].pairs[pair_idx]; idA = pair.macro_id; idB = pair.micro_id; lca_id = pair.lca_id;
    } else {
        if (pair_idx >= p_buf[0].count) return;
        uint2 pair = p_buf[0].pairs[pair_idx]; idA = pair.x; idB = pair.y;
    }

    if (idA == 0xFFFFFFFFu) is_partA = true;

    RigidBody ent_A = rbs[idA]; RigidBody ent_B = rbs[idB];
    uint shape_A = ent_A.shape_type; float3 extents_A = float3(ent_A.shape_x, ent_A.shape_y, ent_A.shape_z); float4 orient_A = float4(ent_A.orient_x, ent_A.orient_y, ent_A.orient_z, ent_A.orient_w); float3 pos_A = float3(ent_A.pos_x, ent_A.pos_y, ent_A.pos_z); float3 vel_A = float3(ent_A.lin_vel_x, ent_A.lin_vel_y, ent_A.lin_vel_z);
    uint shape_B = ent_B.shape_type; float3 extents_B = float3(ent_B.shape_x, ent_B.shape_y, ent_B.shape_z); float4 orient_B = float4(ent_B.orient_x, ent_B.orient_y, ent_B.orient_z, ent_B.orient_w); float3 pos_B = float3(ent_B.pos_x, ent_B.pos_y, ent_B.pos_z); float3 vel_B = float3(ent_B.lin_vel_x, ent_B.lin_vel_y, ent_B.lin_vel_z);

    float4x4 trans_A = float4x4(1.0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1); float4x4 trans_B = float4x4(1.0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);

    if (pc.space_type == 1) {
        LcaEntity lca = lcas[lca_id]; float3 macro_rel_vel_au = vel_A - float3(lca.lin_x, lca.lin_y, lca.lin_z);
        pos_A = mul(lca.inv_transform, float4(pos_A, 1.0)).xyz * AU_TO_KM; vel_A = mul((float3x3)lca.inv_transform, macro_rel_vel_au) * AU_TO_KM; extents_A *= AU_TO_KM; trans_A = lca.inv_transform;
    } else { float3x3 rotA = quat_to_mat3(orient_A); trans_A = float4x4(float4(rotA[0],0), float4(rotA[1],0), float4(rotA[2],0), float4(pos_A,1.0)); }
    float3x3 rotB = quat_to_mat3(orient_B); trans_B = float4x4(float4(rotB[0],0), float4(rotB[1],0), float4(rotB[2],0), float4(pos_B,1.0));

    float toi, depth; float3 normal, contact;
    if (compute_toi_generic(shape_A, extents_A, trans_A, vel_A, shape_B, extents_B, trans_B, vel_B, 1e-3, 10, toi, normal, contact, depth)) {
        if (pc.space_type == 1) {
            vk::BufferPointer<CrossSparseCollisionsType> cout = cast_u2_cdata(pc.cross_output_list);
            uint count; InterlockedAdd(cout[0].count, 1u, count);
            if (count < 4000u) {
                CrossCollisionData ccd; ccd.valid = 1u; ccd.macro_id = idA; ccd.micro_id = idB; ccd.lca_id = lca_id; ccd.toi = toi; ccd.norm_x = normal.x; ccd.norm_y = normal.y; ccd.norm_z = normal.z; ccd.pt_x = contact.x; ccd.pt_y = contact.y; ccd.pt_z = contact.z; ccd.penetration_depth = depth;
                cout[0].pairs[count] = ccd;
            }
        } else {
            vk::BufferPointer<SparseCollisionsType> out = cast_u2_sparse(pc.output_list);
            uint count; InterlockedAdd(out[0].count, 1u, count);
            if (count < 4000u) {
                SparseCollisionData scd; scd.valid = 1u; scd.entity_a = idA; scd.prim_a = idA; scd.entity_b = idB; scd.prim_b = idB; scd.toi = toi; scd.norm_x = normal.x; scd.norm_y = normal.y; scd.norm_z = normal.z; scd.pt_x = contact.x; scd.pt_y = contact.y; scd.pt_z = contact.z; scd.penetration_depth = depth;
                out[0].pairs[count] = scd;
            }
        }
    }
}
#endif


// --- emit_particles ---
#ifdef KERNEL_emit_particles
struct PushConstants { uint2 particles; uint2 candidates; uint2 bvh; uint2 counter; uint root_index; uint num_candidates; float3 sun_pos; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

bool intersectRayAABB(float3 rO, float3 rD, float3 invD, float3 mi, float3 mx, float max_t) {
    float3 t0 = (mi - rO) * invD, t1 = (mx - rO) * invD, tmin = min(t0, t1), tmax = max(t0, t1); float tnear = max(max(tmin.x, tmin.y), tmin.z), tfar = min(min(tmax.x, tmax.y), tmax.z); return tnear <= tfar && tfar > 0.0 && tnear < max_t;
}

[numthreads(128, 1, 1)]
void emit_particles(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint gid = gl_GlobalInvocationID.x; if (gid >= pc.num_candidates) return;

    vk::BufferPointer<float> cand_f = cast_u2_f(pc.candidates);
    vk::BufferPointer<float> part_f = cast_u2_f(pc.particles);
    vk::BufferPointer<MultiBvhNode> bvh = cast_u2_bvh(pc.bvh);
    vk::BufferPointer<uint> counter = cast_u2_u(pc.counter);

    uint stride_u = 10 * SUBGROUP_SIZE;
    uint cand_base = (gid / SUBGROUP_SIZE) * stride_u + (gid % SUBGROUP_SIZE);

    float3 pos = float3(cand_f[cand_base+0], cand_f[cand_base+1*SUBGROUP_SIZE], cand_f[cand_base+2*SUBGROUP_SIZE]);
    float3 dir = pc.sun_pos - pos; float dist = length(dir); if (dist < 1e-5) return; dir /= dist; float3 invDir = 1.0 / dir;

    bool occluded = false; uint stack[64]; int stackPtr = 0; if (pc.root_index != 0xFFFFFFFFu) stack[stackPtr++] = pc.root_index;

    while(stackPtr > 0 && !occluded) {
        uint node = stack[--stackPtr]; uint2 vmask = bvh[node].valid_mask;
        for (uint i = 0; i < SUBGROUP_SIZE; ++i) {
            if (!bvh_node_is_valid(vmask, i)) continue;
            float3 mn = float3(bvh[node].min_x[i], bvh[node].min_y[i], bvh[node].min_z[i]);
            float3 mx = float3(bvh[node].max_x[i], bvh[node].max_y[i], bvh[node].max_z[i]);

            if (intersectRayAABB(pos + dir * 0.1, dir, invDir, mn, mx, dist)) {
                uint meta = bvh[node].metadata[i];
                if (bvh_is_leaf(meta)) { occluded = true; break; }
                else if (bvh_get_index(meta) != 0xFFFFFFFFu) stack[stackPtr++] = bvh_get_index(meta);
            }
        }
    }

    if (!occluded) {
        uint out_idx; InterlockedAdd(counter[0], 1u, out_idx);
        uint out_base = (out_idx / SUBGROUP_SIZE) * stride_u + (out_idx % SUBGROUP_SIZE);
        for (int i = 0; i < 10; ++i) part_f[out_base + i * SUBGROUP_SIZE] = cand_f[cand_base + i * SUBGROUP_SIZE];
    }
}
#endif


// --- morton_encode ---
#ifdef KERNEL_morton_encode
struct PushConstants { uint2 morton_out; uint2 particles; uint num_particles; float3 scene_min; float3 scene_max; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

uint expandBits(uint v) { v = (v * 0x00010001u) & 0xFF0000FFu; v = (v * 0x00000101u) & 0x0F00F00Fu; v = (v * 0x00000011u) & 0xC30C30C3u; v = (v * 0x00000005u) & 0x49249249u; return v; }

[numthreads(256, 1, 1)]
void morton_encode(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint idx = gl_GlobalInvocationID.x; if (idx >= pc.num_particles) return;
    vk::BufferPointer<float> part_f = cast_u2_f(pc.particles);
    vk::BufferPointer<uint2> mout = cast_u2_u2(pc.morton_out);

    uint base = (idx / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (idx % SUBGROUP_SIZE);
    float3 pos = float3(part_f[base + 0], part_f[base + 1*SUBGROUP_SIZE], part_f[base + 2*SUBGROUP_SIZE]);
    float3 extents = pc.scene_max - pc.scene_min, norm_pos = (pos - pc.scene_min) / max(extents, float3(1e-5, 1e-5, 1e-5));
    uint x = uint(clamp(norm_pos.x, 0.0, 1.0) * 1023.0), y = uint(clamp(norm_pos.y, 0.0, 1.0) * 1023.0), z = uint(clamp(norm_pos.z, 0.0, 1.0) * 1023.0);
    mout[idx] = uint2((expandBits(x) << 2) | (expandBits(y) << 1) | expandBits(z), idx);
}
#endif


// --- graph_coloring ---
#ifdef KERNEL_graph_coloring
struct PushConstants { uint2 collisions; uint2 colors; uint2 weights; uint total_pairs; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

uint hash_l(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }

[numthreads(256, 1, 1)]
void graph_coloring(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint idx = gl_GlobalInvocationID.x; if (idx >= pc.total_pairs) return;

    vk::BufferPointer<PackedCollisionsType> cols = cast_u2_packed(pc.collisions);
    vk::BufferPointer<uint> weights = cast_u2_u(pc.weights);
    vk::BufferPointer<uint> colors = cast_u2_u(pc.colors);

    weights[idx] = hash_l(idx + 1); colors[idx] = 0; DeviceMemoryBarrier();

    bool colored = false; uint my_color = 1, my_weight = weights[idx];
    PackedPair my_pair = cols[0].pairs[idx]; uint my_a = my_pair.a.primitive_index, my_b = my_pair.b.primitive_index;

    for (int iter = 0; iter < 10; ++iter) {
        if (!colored) {
            bool is_max = true;
            for (uint j = 0; j < pc.total_pairs; ++j) {
                if (idx == j) continue;
                PackedPair other_pair = cols[0].pairs[j]; uint other_a = other_pair.a.primitive_index, other_b = other_pair.b.primitive_index;
                if (my_a == other_a || my_a == other_b || my_b == other_a || my_b == other_b) {
                    uint other_color = colors[j];
                    if (other_color == 0 || other_color == my_color) {
                        uint other_weight = weights[j];
                        if (other_weight > my_weight || (other_weight == my_weight && j > idx)) { is_max = false; break; }
                    }
                }
            }
            if (is_max) { colors[idx] = my_color; colored = true; }
        }
        DeviceMemoryBarrier(); if (!colored) my_color++;
    }
}
#endif


// --- bp_cross_lca ---
#ifdef KERNEL_bp_cross_lca
struct PushConstants { uint2 lca_entities; uint2 macro_leaves; uint2 entity_headers; uint2 lca_query_pairs; uint2 out_rb_rb; uint2 out_rb_ps; uint2 out_ps_ps; uint2 out_cross_pairs; uint total_queries; uint max_pairs; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint NUM_STACKS_CROSS = 64;
groupshared uint shared_stacks[NUM_STACKS_CROSS][32]; groupshared uint shared_stack_ptrs[NUM_STACKS_CROSS]; groupshared uint2 shared_lca_bvh_addr[NUM_STACKS_CROSS];

void transform_aabb_macro_to_micro(float4x4 lca_inv_transform, float3 macro_center_au, float3 macro_extents_au, out float3 out_min, out float3 out_max) {
    float3 center_km = macro_center_au * AU_TO_KM, extents_km = macro_extents_au * AU_TO_KM;
    float3 corners[8] = {
        float3(center_km.x - extents_km.x, center_km.y - extents_km.y, center_km.z - extents_km.z), float3(center_km.x + extents_km.x, center_km.y - extents_km.y, center_km.z - extents_km.z),
        float3(center_km.x - extents_km.x, center_km.y + extents_km.y, center_km.z - extents_km.z), float3(center_km.x + extents_km.x, center_km.y + extents_km.y, center_km.z - extents_km.z),
        float3(center_km.x - extents_km.x, center_km.y - extents_km.y, center_km.z + extents_km.z), float3(center_km.x + extents_km.x, center_km.y - extents_km.y, center_km.z + extents_km.z),
        float3(center_km.x - extents_km.x, center_km.y + extents_km.y, center_km.z + extents_km.z), float3(center_km.x + extents_km.x, center_km.y + extents_km.y, center_km.z + extents_km.z)
    };
    out_min = float3(1e20, 1e20, 1e20); out_max = float3(-1e20, -1e20, -1e20);
    for (int i = 0; i < 8; i++) { float3 local_p = mul(lca_inv_transform, float4(corners[i], 1.0)).xyz; out_min = min(out_min, local_p); out_max = max(out_max, local_p); }
}

[numthreads(256, 1, 1)]
void bp_cross_lca(uint3 gl_WorkGroupID : SV_GroupID, uint gl_LocalInvocationIndex : SV_GroupIndex) {
    uint subgroup_id = gl_LocalInvocationIndex / WaveGetLaneCount(), lane_id = WaveGetLaneIndex(), query_idx = gl_WorkGroupID.x * (256 / WaveGetLaneCount()) + subgroup_id;

    vk::BufferPointer<PairBufferType> queries = cast_u2_pair(pc.lca_query_pairs);
    if (query_idx >= pc.total_queries || query_idx >= queries[0].count) return;

    vk::BufferPointer<LcaEntity> lcas = cast_u2_lca(pc.lca_entities);
    vk::BufferPointer<TLASLeaf> macros = cast_u2_leaf(pc.macro_leaves);

    uint2 query = queries[0].pairs[query_idx]; uint macro_ent_id = query.x, lca_ent_id = query.y; float3 query_min, query_max;

    if (lane_id == 0) {
        LcaEntity l_ent = lcas[lca_ent_id]; shared_lca_bvh_addr[subgroup_id] = l_ent.bvh;
        TLASLeaf m_leaf = macros[macro_ent_id];
        transform_aabb_macro_to_micro(l_ent.inv_transform, float3((m_leaf.min_x + m_leaf.max_x)*0.5, (m_leaf.min_y + m_leaf.max_y)*0.5, (m_leaf.min_z + m_leaf.max_z)*0.5), float3((m_leaf.max_x - m_leaf.min_x)*0.5, (m_leaf.max_y - m_leaf.min_y)*0.5, (m_leaf.max_z - m_leaf.min_z)*0.5), query_min, query_max);
        shared_stacks[subgroup_id][0] = l_ent.root_index; shared_stack_ptrs[subgroup_id] = 1;
    }

    GroupMemoryBarrierWithGroupSync();
    query_min = float3(WaveReadLaneAt(query_min.x, 0), WaveReadLaneAt(query_min.y, 0), WaveReadLaneAt(query_min.z, 0));
    query_max = float3(WaveReadLaneAt(query_max.x, 0), WaveReadLaneAt(query_max.y, 0), WaveReadLaneAt(query_max.z, 0));
    macro_ent_id = WaveReadLaneAt(macro_ent_id, 0);

    vk::BufferPointer<MultiBvhNode> tlas = cast_u2_bvh(shared_lca_bvh_addr[subgroup_id]);

    while (true) {
        GroupMemoryBarrierWithGroupSync(); uint stack_ptr = shared_stack_ptrs[subgroup_id]; if (stack_ptr == 0) break;
        stack_ptr--; uint node_idx = shared_stacks[subgroup_id][stack_ptr]; if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr;

        uint meta = tlas[node_idx].metadata[lane_id]; bool valid = bvh_node_is_valid(tlas[node_idx].valid_mask, lane_id);
        float3 c_min = float3(tlas[node_idx].min_x[lane_id], tlas[node_idx].min_y[lane_id], tlas[node_idx].min_z[lane_id]);
        float3 c_max = float3(tlas[node_idx].max_x[lane_id], tlas[node_idx].max_y[lane_id], tlas[node_idx].max_z[lane_id]);
        uint child_payload = tlas[node_idx].child_indices[lane_id];

        bool hit = valid && intersectAABB(query_min, query_max, c_min, c_max), is_leaf = bvh_is_leaf(meta), hit_leaf = hit && is_leaf, hit_node = hit && !is_leaf;
        uint4 leaf_ballot = WaveActiveBallot(hit_leaf); uint leaf_count = get_ballot_count(leaf_ballot), leaf_offset = get_ballot_prefix(leaf_ballot, WaveGetLaneIndex());

        if (leaf_count > 0) {
            uint base_idx = 0;
            vk::BufferPointer<CrossPairBufferType> cp = cast_u2_cpair(pc.out_cross_pairs);
            if (lane_id == 0) InterlockedAdd(cp[0].count, leaf_count, base_idx);
            base_idx = WaveReadLaneAt(base_idx, 0);

            if (hit_leaf && (base_idx + leaf_offset) < pc.max_pairs) {
                CrossPair cx; cx.macro_id = macro_ent_id; cx.micro_id = bvh_get_index(meta); cx.lca_id = lca_ent_id; cx.pad = 0;
                cp[0].pairs[base_idx + leaf_offset] = cx;
            }
        }

        vk::BufferPointer<EntityHeader> headers = cast_u2_header(pc.entity_headers);

        for (uint i = 0; i < 4; i++) {
            uint m = leaf_ballot[i];
            while (m != 0) {
                uint bit = firstbitlow(m); m &= ~(1u << bit); uint src_lane = i * 32 + bit;
                uint micro_ent_id = bvh_get_index(WaveReadLaneAt(meta, src_lane));

                if (lane_id == 0) {
                    uint type_A = headers[macro_ent_id].type, type_B = headers[micro_ent_id].type;
                    uint ent_A = macro_ent_id, ent_B = micro_ent_id; if (type_A > type_B) { uint temp = ent_A; ent_A = ent_B; ent_B = temp; temp = type_A; type_A = type_B; type_B = temp; }

                    if (type_A == TYPE_RIGID_BODY && type_B == TYPE_RIGID_BODY && pc.out_rb_rb.x != 0) { vk::BufferPointer<PairBufferType> pb = cast_u2_pair(pc.out_rb_rb); uint out_idx; InterlockedAdd(pb[0].count, 1u, out_idx); if (out_idx < pc.max_pairs) pb[0].pairs[out_idx] = uint2(ent_A, ent_B); }
                    else if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_RIGID_BODY && pc.out_rb_ps.x != 0) { vk::BufferPointer<PairBufferType> pb = cast_u2_pair(pc.out_rb_ps); uint out_idx; InterlockedAdd(pb[0].count, 1u, out_idx); if (out_idx < pc.max_pairs) pb[0].pairs[out_idx] = uint2(ent_B, ent_A); }
                    else if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_PARTICLE_SYSTEM && pc.out_ps_ps.x != 0) { vk::BufferPointer<PairBufferType> pb = cast_u2_pair(pc.out_ps_ps); uint out_idx; InterlockedAdd(pb[0].count, 1u, out_idx); if (out_idx < pc.max_pairs) pb[0].pairs[out_idx] = uint2(ent_A, ent_B); }
                }
            }
        }

        uint4 node_ballot = WaveActiveBallot(hit_node); if (hit_node) shared_stacks[subgroup_id][stack_ptr + get_ballot_prefix(node_ballot, WaveGetLaneIndex())] = child_payload;
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr + get_ballot_count(node_ballot);
    }
}
#endif


// --- bp_scene ---
#ifdef KERNEL_bp_scene
struct PushConstants { uint2 tlas_bvh; uint2 query_leaves; uint2 overlapping_pairs; uint tlas_root_index; uint total_queries; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint NUM_STACKS_SCENE = 64;
groupshared uint shared_stacks_s[NUM_STACKS_SCENE][32]; groupshared uint shared_stack_ptrs_s[NUM_STACKS_SCENE];

[numthreads(256, 1, 1)]
void bp_scene(uint3 gl_WorkGroupID : SV_GroupID, uint gl_LocalInvocationIndex : SV_GroupIndex) {
    uint subgroup_id = gl_LocalInvocationIndex / WaveGetLaneCount(), lane_id = WaveGetLaneIndex(), query_idx = gl_WorkGroupID.x * (256 / WaveGetLaneCount()) + subgroup_id;
    if (query_idx >= pc.total_queries) return;

    vk::BufferPointer<TLASLeaf> leaves = cast_u2_leaf(pc.query_leaves);
    vk::BufferPointer<MultiBvhNode> tlas = cast_u2_bvh(pc.tlas_bvh);
    vk::BufferPointer<PairBufferType> overlaps = cast_u2_pair(pc.overlapping_pairs);

    float3 my_min, my_max; uint my_ent_id;
    if (lane_id == 0) {
        TLASLeaf q_leaf = leaves[query_idx];
        my_min = float3(q_leaf.min_x, q_leaf.min_y, q_leaf.min_z); my_max = float3(q_leaf.max_x, q_leaf.max_y, q_leaf.max_z); my_ent_id = q_leaf.entity_idx;
        shared_stacks_s[subgroup_id][0] = pc.tlas_root_index; shared_stack_ptrs_s[subgroup_id] = 1;
    }

    my_min = float3(WaveReadLaneAt(my_min.x, 0), WaveReadLaneAt(my_min.y, 0), WaveReadLaneAt(my_min.z, 0));
    my_max = float3(WaveReadLaneAt(my_max.x, 0), WaveReadLaneAt(my_max.y, 0), WaveReadLaneAt(my_max.z, 0));
    my_ent_id = WaveReadLaneAt(my_ent_id, 0);

    while (true) {
        GroupMemoryBarrierWithGroupSync(); uint stack_ptr = shared_stack_ptrs_s[subgroup_id]; if (stack_ptr == 0) break;
        stack_ptr--; uint node_idx = shared_stacks_s[subgroup_id][stack_ptr]; if (lane_id == 0) shared_stack_ptrs_s[subgroup_id] = stack_ptr;

        uint meta = tlas[node_idx].metadata[lane_id]; bool valid = bvh_node_is_valid(tlas[node_idx].valid_mask, lane_id);
        float3 c_min = float3(tlas[node_idx].min_x[lane_id], tlas[node_idx].min_y[lane_id], tlas[node_idx].min_z[lane_id]);
        float3 c_max = float3(tlas[node_idx].max_x[lane_id], tlas[node_idx].max_y[lane_id], tlas[node_idx].max_z[lane_id]);
        uint child_payload = tlas[node_idx].child_indices[lane_id], entity_id = bvh_get_index(meta);

        bool hit = valid && intersectAABB(my_min, my_max, c_min, c_max), is_leaf = bvh_is_leaf(meta), hit_leaf = hit && is_leaf && (my_ent_id < entity_id), hit_node = hit && !is_leaf;
        uint4 leaf_ballot = WaveActiveBallot(hit_leaf); uint leaf_count = get_ballot_count(leaf_ballot), leaf_offset = get_ballot_prefix(leaf_ballot, WaveGetLaneIndex());

        if (leaf_count > 0) {
            uint base_idx = 0; if (lane_id == 0) InterlockedAdd(overlaps[0].count, leaf_count, base_idx);
            base_idx = WaveReadLaneAt(base_idx, 0);
            if (hit_leaf && base_idx + leaf_offset < 100000u) overlaps[0].pairs[base_idx + leaf_offset] = uint2(my_ent_id, entity_id);
        }

        uint4 node_ballot = WaveActiveBallot(hit_node); if (hit_node) shared_stacks_s[subgroup_id][stack_ptr + get_ballot_prefix(node_ballot, WaveGetLaneIndex())] = child_payload;
        if (lane_id == 0) shared_stack_ptrs_s[subgroup_id] = stack_ptr + get_ballot_count(node_ballot);
    }
}
#endif


// --- integrate_bodies_p3 ---
#ifdef KERNEL_integrate_bodies_p3
struct PushConstants { uint2 rigid_bodies; uint2 wrenches; uint2 emitters; float dt; uint n_bodies; uint n_iterations; uint num_emitters; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(32, 1, 1)]
void integrate_bodies_p3(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint id = gl_GlobalInvocationID.x; if (id >= pc.n_bodies) return;

    vk::BufferPointer<RigidBody> rbs = cast_u2_rb(pc.rigid_bodies);
    vk::BufferPointer<Wrench> ws = cast_u2_wrench(pc.wrenches);
    vk::BufferPointer<ForceEmitter> ems = cast_u2_emitter(pc.emitters);

    RigidBody body = rbs[id];
    float mass = body.mass, inv_m = (mass > 0.0) ? 1.0 / mass : 0.0;
    float3 I_inv = float3(body.inv_inertia_x, body.inv_inertia_y, body.inv_inertia_z);
    float3 I_fwd = float3((I_inv.x > 1e-14) ? 1.0 / I_inv.x : 0.0, (I_inv.y > 1e-14) ? 1.0 / I_inv.y : 0.0, (I_inv.z > 1e-14) ? 1.0 / I_inv.z : 0.0);

    float3 pos_n = float3(body.pos_x, body.pos_y, body.pos_z), v_n = float3(body.lin_vel_x, body.lin_vel_y, body.lin_vel_z), w_n = float3(body.ang_vel_x, body.ang_vel_y, body.ang_vel_z);
    float4 q_n = float4(body.orient_x, body.orient_y, body.orient_z, body.orient_w);
    Wrench wrench = ws[body.wrench_idx];
    float3 f_n = float3(asfloat(wrench.force_x), asfloat(wrench.force_y), asfloat(wrench.force_z)), t_n = float3(asfloat(wrench.torque_x), asfloat(wrench.torque_y), asfloat(wrench.torque_z));

    for (uint e = 0; e < pc.num_emitters; ++e) {
        ForceEmitter em = ems[e];
        float3 em_pos = float3(em.pos_x, em.pos_y, em.pos_z);
        if (em.type_id == 0) {
            float3 r = em_pos - pos_n; float s_dist_sq = dot(r, r) * em.scale_factor * em.scale_factor;
            if (s_dist_sq > 1e-6) {
                float s_dist = sqrt(s_dist_sq); float softening = 1.0 - exp(-(s_dist_sq * s_dist * s_dist_sq));
                f_n += normalize(r) * ((em.mu * mass * softening) / s_dist_sq);
            }
        } else if (em.type_id == 1) {
            float dist = dot(pos_n - em_pos, float3(em.norm_x, em.norm_y, em.norm_z)); if (dist >= 0.0 && dist <= em.trunc_distance) f_n += float3(em.norm_x, em.norm_y, em.norm_z) * em.mu;
        }
    }

    float half_dt = 0.5 * pc.dt; float3 a_lin = f_n * inv_m, v_mid = v_n + half_dt * a_lin, pos_next = pos_n + pc.dt * v_mid, v_next = v_n + pc.dt * a_lin;
    float3 t_local = quat_rotate_inv(q_n, t_n), w_n_local = quat_rotate_inv(q_n, w_n), w_mid_local = w_n_local;

    for (uint iter = 0u; iter < pc.n_iterations; ++iter) { float3 gyro = cross(w_mid_local, I_fwd * w_mid_local); w_mid_local = w_n_local + half_dt * (I_inv * (t_local - gyro)); }

    float3 w_next_local = 2.0 * w_mid_local - w_n_local, w_next = quat_rotate(q_n, w_next_local), w_mid_world = quat_rotate(q_n, w_mid_local);
    float4 q_next = normalize(q_n + half_dt * quat_mult(float4(w_mid_world, 0.0), q_n));

    body.pos_x = pos_next.x; body.pos_y = pos_next.y; body.pos_z = pos_next.z;
    body.orient_x = q_next.x; body.orient_y = q_next.y; body.orient_z = q_next.z; body.orient_w = q_next.w;
    body.lin_vel_x = v_next.x; body.lin_vel_y = v_next.y; body.lin_vel_z = v_next.z;
    body.ang_vel_x = w_next.x; body.ang_vel_y = w_next.y; body.ang_vel_z = w_next.z;

    rbs[id] = body;
    ws[body.wrench_idx].force_x = 0; ws[body.wrench_idx].force_y = 0; ws[body.wrench_idx].force_z = 0;
    ws[body.wrench_idx].torque_x = 0; ws[body.wrench_idx].torque_y = 0; ws[body.wrench_idx].torque_z = 0;
}
#endif


// --- apply_impulses ---
#ifdef KERNEL_apply_impulses
struct PushConstants { uint2 particles; uint2 collisions; uint2 impulses; uint2 rigid_bodies; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void apply_impulses(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    vk::BufferPointer<PackedCollisionsType> cols = cast_u2_packed(pc.collisions);
    uint global_id = gl_GlobalInvocationID.x; if (global_id >= cols[0].count) return;

    vk::BufferPointer<float> imps = cast_u2_f(pc.impulses);
    vk::BufferPointer<uint> rbs_u = cast_u2_u(pc.rigid_bodies);
    vk::BufferPointer<uint> parts_u = cast_u2_u(pc.particles);
    vk::BufferPointer<RigidBody> rbs = cast_u2_rb(pc.rigid_bodies);
    vk::BufferPointer<float> part_f = cast_u2_f(pc.particles);

    PackedPair pair = cols[0].pairs[global_id];
    float3 impulse = float3(imps[global_id * 4 + 0], imps[global_id * 4 + 1], imps[global_id * 4 + 2]);
    if (length(impulse) < 1e-6) return;

    uint pA_id = pair.a.primitive_index, pB_id = pair.b.primitive_index;
    bool is_rb_a = (pair.a.entity_id != 0xFFFFFFFFu), is_rb_b = (pair.b.entity_id != 0xFFFFFFFFu);

    if (is_rb_a) {
        RigidBody rbA = rbs[pA_id]; float mass = rbA.mass, invMA = mass > 0.0 ? 1.0 / mass : 0.0;
        if (invMA > 0.0) {
            float3 dvA = -impulse * invMA, dwA = quat_rotate(float4(rbA.orient_x, rbA.orient_y, rbA.orient_z, rbA.orient_w), float3(rbA.inv_inertia_x, rbA.inv_inertia_y, rbA.inv_inertia_z) * quat_rotate_inv(float4(rbA.orient_x, rbA.orient_y, rbA.orient_z, rbA.orient_w), cross(float3(pair.pt_x, pair.pt_y, pair.pt_z) - float3(rbA.pos_x, rbA.pos_y, rbA.pos_z), -impulse)));
            bda_atomic_add_float(rbs_u, pA_id * 28 + 8, dvA.x); bda_atomic_add_float(rbs_u, pA_id * 28 + 9, dvA.y); bda_atomic_add_float(rbs_u, pA_id * 28 + 10, dvA.z);
            bda_atomic_add_float(rbs_u, pA_id * 28 + 12, dwA.x); bda_atomic_add_float(rbs_u, pA_id * 28 + 13, dwA.y); bda_atomic_add_float(rbs_u, pA_id * 28 + 14, dwA.z);
        }
    } else {
        uint base = (pA_id / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (pA_id % SUBGROUP_SIZE); float mass = part_f[base + 6 * SUBGROUP_SIZE], invMA = mass > 0.0 ? 1.0 / mass : 0.0;
        if (invMA > 0.0) {
            float3 dvA = -impulse * invMA;
            bda_atomic_add_float(parts_u, base + 3 * SUBGROUP_SIZE, dvA.x); bda_atomic_add_float(parts_u, base + 4 * SUBGROUP_SIZE, dvA.y); bda_atomic_add_float(parts_u, base + 5 * SUBGROUP_SIZE, dvA.z);
        }
    }

    if (is_rb_b) {
        RigidBody rbB = rbs[pB_id]; float mass = rbB.mass, invMB = mass > 0.0 ? 1.0 / mass : 0.0;
        if (invMB > 0.0) {
            float3 dvB = impulse * invMB, dwB = quat_rotate(float4(rbB.orient_x, rbB.orient_y, rbB.orient_z, rbB.orient_w), float3(rbB.inv_inertia_x, rbB.inv_inertia_y, rbB.inv_inertia_z) * quat_rotate_inv(float4(rbB.orient_x, rbB.orient_y, rbB.orient_z, rbB.orient_w), cross(float3(pair.pt_x, pair.pt_y, pair.pt_z) - float3(rbB.pos_x, rbB.pos_y, rbB.pos_z), impulse)));
            bda_atomic_add_float(rbs_u, pB_id * 28 + 8, dvB.x); bda_atomic_add_float(rbs_u, pB_id * 28 + 9, dvB.y); bda_atomic_add_float(rbs_u, pB_id * 28 + 10, dvB.z);
            bda_atomic_add_float(rbs_u, pB_id * 28 + 12, dwB.x); bda_atomic_add_float(rbs_u, pB_id * 28 + 13, dwB.y); bda_atomic_add_float(rbs_u, pB_id * 28 + 14, dwB.z);
        }
    } else {
        uint base = (pB_id / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (pB_id % SUBGROUP_SIZE); float mass = part_f[base + 6 * SUBGROUP_SIZE], invMB = mass > 0.0 ? 1.0 / mass : 0.0;
        if (invMB > 0.0) {
            float3 dvB = impulse * invMB;
            bda_atomic_add_float(parts_u, base + 3 * SUBGROUP_SIZE, dvB.x); bda_atomic_add_float(parts_u, base + 4 * SUBGROUP_SIZE, dvB.y); bda_atomic_add_float(parts_u, base + 5 * SUBGROUP_SIZE, dvB.z);
        }
    }
}
#endif


// --- lbvh_collapse ---
#ifdef KERNEL_lbvh_collapse
struct PushConstants { uint2 binary_bvh; uint2 multi_bvh; uint2 collapse_map; uint num_multi_nodes; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(256, 1, 1)]
void lbvh_collapse(uint3 gl_WorkGroupID : SV_GroupID, uint gl_LocalInvocationIndex : SV_GroupIndex) {
    uint multi_node_idx = gl_WorkGroupID.x; if (multi_node_idx >= pc.num_multi_nodes) return;

    vk::BufferPointer<uint> cmap = cast_u2_u(pc.collapse_map);
    vk::BufferPointer<MultiBvhNode> bin_bvh = cast_u2_bvh(pc.binary_bvh);
    vk::BufferPointer<MultiBvhNode> mul_bvh = cast_u2_bvh(pc.multi_bvh);

    uint lane = WaveGetLaneIndex(), binary_idx = cmap[multi_node_idx];
    bool is_leaf = false; uint payload = 0, f_parent = 0, f_dir = 0;

    int depth = firstbithigh(SUBGROUP_SIZE) - 1;
    for (int d = depth; d >= 0; d--) {
        uint dir = (lane >> d) & 1u, meta = bin_bvh[binary_idx].metadata[dir];
        is_leaf = bvh_is_leaf(meta); uint next_idx = bvh_get_index(meta); f_parent = binary_idx; f_dir = dir;
        if (is_leaf) { payload = next_idx; break; } binary_idx = next_idx;
    }

    if (!is_leaf) { payload = binary_idx; f_parent = bin_bvh[binary_idx].parent_idx; f_dir = (bin_bvh[f_parent].child_indices[1] == binary_idx) ? 1 : 0; }

    mul_bvh[multi_node_idx].min_x[lane] = bin_bvh[f_parent].min_x[f_dir]; mul_bvh[multi_node_idx].max_x[lane] = bin_bvh[f_parent].max_x[f_dir];
    mul_bvh[multi_node_idx].min_y[lane] = bin_bvh[f_parent].min_y[f_dir]; mul_bvh[multi_node_idx].max_y[lane] = bin_bvh[f_parent].max_y[f_dir];
    mul_bvh[multi_node_idx].min_z[lane] = bin_bvh[f_parent].min_z[f_dir]; mul_bvh[multi_node_idx].max_z[lane] = bin_bvh[f_parent].max_z[f_dir];
    mul_bvh[multi_node_idx].child_indices[lane] = payload; mul_bvh[multi_node_idx].metadata[lane] = bvh_pack_metadata(is_leaf, BVH_FRAME_MICRO, BVH_SHAPE_AABB, payload);
    mul_bvh[multi_node_idx].masses[lane] = bin_bvh[f_parent].masses[f_dir];
    mul_bvh[multi_node_idx].com_x[lane] = bin_bvh[f_parent].com_x[f_dir]; mul_bvh[multi_node_idx].com_y[lane] = bin_bvh[f_parent].com_y[f_dir]; mul_bvh[multi_node_idx].com_z[lane] = bin_bvh[f_parent].com_z[f_dir];

    if (lane == 0) {
        uint mask_x = (SUBGROUP_SIZE >= 32) ? 0xFFFFFFFFu : ((1u << SUBGROUP_SIZE) - 1u), mask_y = 0u; if (SUBGROUP_SIZE > 32) mask_y = (SUBGROUP_SIZE >= 64) ? 0xFFFFFFFFu : ((1u << (SUBGROUP_SIZE - 32)) - 1u);
        mul_bvh[multi_node_idx].valid_mask = uint2(mask_x, mask_y);
        for (uint i = 0; i < 8; ++i) for (uint j = 0; j < SUBGROUP_SIZE; ++j) mul_bvh[multi_node_idx].permutations[i][j] = j;
    }
}
#endif


// --- bp_particle_self ---
#ifdef KERNEL_bp_particle_self
struct PushConstants { uint2 bvh; uint2 particles; uint2 wrench_buffer; uint root_index; uint total_particles; float particle_radius; float stiffness; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint NUM_STACKS_SELF = 64;
groupshared uint shared_stacks_self[NUM_STACKS_SELF][32]; groupshared uint shared_stack_ptrs_self[NUM_STACKS_SELF];

[numthreads(256, 1, 1)]
void bp_particle_self(uint3 gl_WorkGroupID : SV_GroupID, uint gl_LocalInvocationIndex : SV_GroupIndex) {
    uint subgroup_id = gl_LocalInvocationIndex / WaveGetLaneCount(), lane_id = WaveGetLaneIndex(), my_p_idx = gl_WorkGroupID.x * (256 / WaveGetLaneCount()) + subgroup_id;
    if (my_p_idx >= pc.total_particles) return;

    vk::BufferPointer<float> part_f = cast_u2_f(pc.particles);
    vk::BufferPointer<MultiBvhNode> bvh = cast_u2_bvh(pc.bvh);
    vk::BufferPointer<uint> w_u = cast_u2_u(pc.wrench_buffer);

    float3 my_pos, my_min, my_max; float my_radius = pc.particle_radius;

    if (lane_id == 0) {
        uint base = (my_p_idx / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (my_p_idx % SUBGROUP_SIZE);
        my_pos = float3(part_f[base + 0], part_f[base + 1 * SUBGROUP_SIZE], part_f[base + 2 * SUBGROUP_SIZE]);
        my_min = my_pos - (float3)my_radius; my_max = my_pos + (float3)my_radius;
        shared_stacks_self[subgroup_id][0] = pc.root_index; shared_stack_ptrs_self[subgroup_id] = 1;
    }

    my_pos = float3(WaveReadLaneAt(my_pos.x, 0), WaveReadLaneAt(my_pos.y, 0), WaveReadLaneAt(my_pos.z, 0));
    my_min = float3(WaveReadLaneAt(my_min.x, 0), WaveReadLaneAt(my_min.y, 0), WaveReadLaneAt(my_min.z, 0));
    my_max = float3(WaveReadLaneAt(my_max.x, 0), WaveReadLaneAt(my_max.y, 0), WaveReadLaneAt(my_max.z, 0));
    my_p_idx = WaveReadLaneAt(my_p_idx, 0);

    float3 local_repulsive_force = (float3)0.0;

    while (true) {
        GroupMemoryBarrierWithGroupSync(); uint stack_ptr = shared_stack_ptrs_self[subgroup_id]; if (stack_ptr == 0) break;
        stack_ptr--; uint node_idx = shared_stacks_self[subgroup_id][stack_ptr]; if (lane_id == 0) shared_stack_ptrs_self[subgroup_id] = stack_ptr;

        uint meta = bvh[node_idx].metadata[lane_id]; bool valid = bvh_node_is_valid(bvh[node_idx].valid_mask, lane_id);
        float3 c_min = float3(bvh[node_idx].min_x[lane_id], bvh[node_idx].min_y[lane_id], bvh[node_idx].min_z[lane_id]);
        float3 c_max = float3(bvh[node_idx].max_x[lane_id], bvh[node_idx].max_y[lane_id], bvh[node_idx].max_z[lane_id]);
        uint child_payload = bvh[node_idx].child_indices[lane_id];

        bool hit_aabb = valid && intersectAABB(my_min, my_max, c_min, c_max), is_leaf = bvh_is_leaf(meta), hit_node = hit_aabb && !is_leaf, hit_leaf = hit_aabb && is_leaf && (my_p_idx != child_payload);

        uint4 leaf_ballot = WaveActiveBallot(hit_leaf);
        for (uint i = 0; i < 4; i++) {
            uint mask = leaf_ballot[i];
            while (mask != 0) {
                uint bit = firstbitlow(mask); mask &= ~(1u << bit);
                uint other_idx = WaveReadLaneAt(child_payload, i * 32 + bit), base = (other_idx / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (other_idx % SUBGROUP_SIZE);
                float3 other_pos = float3(part_f[base + 0], part_f[base + 1 * SUBGROUP_SIZE], part_f[base + 2 * SUBGROUP_SIZE]);
                float3 diff = my_pos - other_pos; float dist_sq = dot(diff, diff), min_dist = my_radius * 2.0;

                if (dist_sq > 1e-12 && dist_sq < min_dist * min_dist) { float dist = sqrt(dist_sq); local_repulsive_force += (diff / dist) * (pc.stiffness * (min_dist - dist)); }
            }
        }

        uint4 node_ballot = WaveActiveBallot(hit_node); if (hit_node) shared_stacks_self[subgroup_id][stack_ptr + get_ballot_prefix(node_ballot, WaveGetLaneIndex())] = child_payload;
        if (lane_id == 0) shared_stack_ptrs_self[subgroup_id] = stack_ptr + get_ballot_count(node_ballot);
    }

    local_repulsive_force.x = WaveActiveSum(local_repulsive_force.x); local_repulsive_force.y = WaveActiveSum(local_repulsive_force.y); local_repulsive_force.z = WaveActiveSum(local_repulsive_force.z);
    if (lane_id == 0 && dot(local_repulsive_force, local_repulsive_force) > 0.0) {
        bda_atomic_add_float(w_u, my_p_idx * 6 + 0, local_repulsive_force.x);
        bda_atomic_add_float(w_u, my_p_idx * 6 + 1, local_repulsive_force.y);
        bda_atomic_add_float(w_u, my_p_idx * 6 + 2, local_repulsive_force.z);
    }
}
#endif


// --- integrate_particles_p4_5 ---
#ifdef KERNEL_integrate_particles_p4_5
struct PushConstants { uint2 particles; uint2 clock; float dt; uint total_particles; uint dt_us_lo; uint dt_us_hi; uint current_time_lo; uint current_time_hi; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void integrate_particles_p4_5(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint gid = gl_GlobalInvocationID.x;

    if (gid == 0u) {
        vk::BufferPointer<ClockBufferType> clock = cast_u2_clock(pc.clock);
        clock[0].global_time_us = add64(uint2(pc.current_time_lo, pc.current_time_hi), uint2(pc.dt_us_lo, pc.dt_us_hi));
    }
    if (gid >= pc.total_particles) return;

    vk::BufferPointer<float> part_f = cast_u2_f(pc.particles);
    vk::BufferPointer<uint> part_u = cast_u2_u(pc.particles);

    uint base = (gid / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (gid % SUBGROUP_SIZE);
    float mass = part_f[base + 6 * SUBGROUP_SIZE]; if (mass <= 0.0) return;

    float3 v_half = float3(part_f[base + 3 * SUBGROUP_SIZE], part_f[base + 4 * SUBGROUP_SIZE], part_f[base + 5 * SUBGROUP_SIZE]);
    float3 f_next = float3(part_f[base + 7 * SUBGROUP_SIZE], part_f[base + 8 * SUBGROUP_SIZE], part_f[base + 9 * SUBGROUP_SIZE]);
    float3 v_next = v_half + f_next * (1.0 / mass) * (0.5 * pc.dt);

    part_u[base + 3 * SUBGROUP_SIZE] = asuint(v_next.x);
    part_u[base + 4 * SUBGROUP_SIZE] = asuint(v_next.y);
    part_u[base + 5 * SUBGROUP_SIZE] = asuint(v_next.z);
}
#endif


// --- convert_particles ---
#ifdef KERNEL_convert_particles
struct PushConstants { uint2 aosoa_particles; uint2 mega_particles; uint2 mega_indirect; uint2 atomic_counters; uint mega_indirect_index; uint mega_particle_offset; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void convert_particles(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    vk::BufferPointer<uint> counters = cast_u2_u(pc.atomic_counters);
    uint total_particles = counters[0];

    if (gl_GlobalInvocationID.x == 0) {
        vk::BufferPointer<DrawIndirectCommand> cmds = cast_u2_indirect(pc.mega_indirect);
        DrawIndirectCommand cmd; cmd.vertexCount = 4; cmd.instanceCount = total_particles; cmd.firstVertex = 0; cmd.firstInstance = pc.mega_particle_offset;
        cmds[pc.mega_indirect_index] = cmd;
    }

    uint idx = gl_GlobalInvocationID.x; if (idx >= total_particles) return;

    vk::BufferPointer<float> in_f = cast_u2_f(pc.aosoa_particles);
    vk::BufferPointer<MegaParticleData> out_p = cast_u2_mega(pc.mega_particles);

    uint in_base = (idx / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (idx % SUBGROUP_SIZE);

    MegaParticleData pd; pd.id_low = 0; pd.id_high = 0; pd.age_low = 0; pd.age_high = 0; pd.is_active = 1;
    pd.pos_x = in_f[in_base + 0 * SUBGROUP_SIZE]; pd.pos_y = in_f[in_base + 1 * SUBGROUP_SIZE]; pd.pos_z = in_f[in_base + 2 * SUBGROUP_SIZE];
    pd.vel_x = in_f[in_base + 3 * SUBGROUP_SIZE]; pd.vel_y = in_f[in_base + 4 * SUBGROUP_SIZE]; pd.vel_z = in_f[in_base + 5 * SUBGROUP_SIZE];
    pd.mass = in_f[in_base + 6 * SUBGROUP_SIZE];

    out_p[pc.mega_particle_offset + idx] = pd;
}
#endif


// --- lbvh_prepass ---
#ifdef KERNEL_lbvh_prepass
struct PushConstants { uint2 bvh; uint2 counters; uint num_internal_nodes; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(256, 1, 1)]
void lbvh_prepass(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint idx = gl_GlobalInvocationID.x; if (idx >= pc.num_internal_nodes) return;
    vk::BufferPointer<uint> c = cast_u2_u(pc.counters);
    vk::BufferPointer<MultiBvhNode> bvh = cast_u2_bvh(pc.bvh);
    c[idx] = 0; if (idx == 0) bvh[0].parent_idx = 0xFFFFFFFFu;
}
#endif


// --- bp_clear ---
#ifdef KERNEL_bp_clear
struct PushConstants { uint2 raw_scene_pairs; uint2 out_rb_rb; uint2 out_rb_ps; uint2 out_rb_lca; uint2 internal_pairs; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(1, 1, 1)]
void bp_clear() {
    if (pc.raw_scene_pairs.x != 0 || pc.raw_scene_pairs.y != 0) cast_u2_u(pc.raw_scene_pairs)[0] = 0u;
    if (pc.out_rb_rb.x != 0 || pc.out_rb_rb.y != 0) cast_u2_u(pc.out_rb_rb)[0] = 0u;
    if (pc.out_rb_ps.x != 0 || pc.out_rb_ps.y != 0) cast_u2_u(pc.out_rb_ps)[0] = 0u;
    if (pc.out_rb_lca.x != 0 || pc.out_rb_lca.y != 0) cast_u2_u(pc.out_rb_lca)[0] = 0u;
    if (pc.internal_pairs.x != 0 || pc.internal_pairs.y != 0) cast_u2_u(pc.internal_pairs)[0] = 0u;
}
#endif


// --- reduce_toi ---
#ifdef KERNEL_reduce_toi
struct PushConstants { uint2 particles; uint2 collisions; uint2 out_toi; float particle_radius; float dt; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
groupshared uint shared_min_toi[64];

[numthreads(128, 1, 1)]
void reduce_toi(uint3 gl_GlobalInvocationID : SV_DispatchThreadID, uint3 gl_LocalInvocationID : SV_GroupThreadID, uint gl_LocalInvocationIndex : SV_GroupIndex) {
    uint global_id = gl_GlobalInvocationID.x, local_id = gl_LocalInvocationID.x, subgroup_id = gl_LocalInvocationIndex / WaveGetLaneCount();

    vk::BufferPointer<PackedCollisionsType> cols = cast_u2_packed(pc.collisions);
    vk::BufferPointer<uint> toi = cast_u2_u(pc.out_toi);

    float tc = pc.dt; if (global_id < cols[0].count) tc = cols[0].pairs[global_id].toi;

    if (WaveGetLaneIndex() == 0) shared_min_toi[subgroup_id] = asuint(WaveActiveMin(tc)); GroupMemoryBarrierWithGroupSync();

    if (local_id == 0) {
        uint wg_min_uint = shared_min_toi[0]; for (uint i = 1; i < (128 / WaveGetLaneCount()); i++) wg_min_uint = min(wg_min_uint, shared_min_toi[i]);
        uint original_val; InterlockedMin(toi[0], wg_min_uint, original_val);
    }
}
#endif


// --- barnes_hut ---
#ifdef KERNEL_barnes_hut
struct PushConstants { uint2 particles; uint2 bvh; uint2 cluster_list; uint2 wrenches; uint num_clusters; float dt; float theta; float G; float softening_sq; uint root_node_idx; uint cluster_threshold; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint NUM_STACKS_BH = 64;
groupshared uint shared_stacks_bh[NUM_STACKS_BH][64]; groupshared uint shared_stack_ptrs_bh[NUM_STACKS_BH];

[numthreads(256, 1, 1)]
void barnes_hut(uint3 gl_WorkGroupID : SV_GroupID, uint gl_LocalInvocationIndex : SV_GroupIndex) {
    uint subgroup_id = gl_LocalInvocationIndex / WaveGetLaneCount(), lane_id = WaveGetLaneIndex(), cluster_job_idx = gl_WorkGroupID.x * (256 / WaveGetLaneCount()) + subgroup_id;
    if (cluster_job_idx >= pc.num_clusters) return;

    vk::BufferPointer<uint> cl_list = cast_u2_u(pc.cluster_list);
    vk::BufferPointer<MultiBvhNode> bvh = cast_u2_bvh(pc.bvh);
    vk::BufferPointer<float> part_f = cast_u2_f(pc.particles);

    uint target_node_idx = cl_list[cluster_job_idx];
    bool i_am_valid = bvh_node_is_valid(bvh[target_node_idx].valid_mask, lane_id);
    uint my_p_idx   = bvh[target_node_idx].child_indices[lane_id];

    float3 my_pos = (float3)0.0; float my_mass = 0.0;
    if (i_am_valid) {
        uint base = (my_p_idx / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (my_p_idx % SUBGROUP_SIZE);
        my_pos = float3(part_f[base + 0], part_f[base + 1*SUBGROUP_SIZE], part_f[base + 2*SUBGROUP_SIZE]);
        my_mass = part_f[base + 6*SUBGROUP_SIZE];
    }

    float3 safe_pos = i_am_valid ? my_pos : (float3)0.0, min_pos = WaveActiveMin(i_am_valid ? my_pos : float3(1e20, 1e20, 1e20)), max_pos = WaveActiveMax(i_am_valid ? my_pos : float3(-1e20, -1e20, -1e20));
    float3 cluster_extents = max_pos - min_pos; float target_size = max(cluster_extents.x, max(cluster_extents.y, cluster_extents.z)), sum_mass = WaveActiveSum(i_am_valid ? my_mass : 0.0);
    float3 target_com = WaveActiveSum(safe_pos * my_mass) / max(sum_mass, 1e-6), my_acc = (float3)0.0;

    if (lane_id == 0) { shared_stacks_bh[subgroup_id][0] = pc.root_node_idx; shared_stack_ptrs_bh[subgroup_id] = 1; }

    while (true) {
        GroupMemoryBarrierWithGroupSync(); uint stack_ptr = shared_stack_ptrs_bh[subgroup_id]; if (stack_ptr == 0) break;
        uint source_node_idx = shared_stacks_bh[subgroup_id][--stack_ptr]; if (lane_id == 0) shared_stack_ptrs_bh[subgroup_id] = stack_ptr;

        bool s_valid = bvh_node_is_valid(bvh[source_node_idx].valid_mask, lane_id), s_is_leaf = bvh_is_leaf(bvh[source_node_idx].metadata[lane_id]);
        float3 s_com = float3(bvh[source_node_idx].com_x[lane_id], bvh[source_node_idx].com_y[lane_id], bvh[source_node_idx].com_z[lane_id]);
        float s_mass = bvh[source_node_idx].masses[lane_id];
        uint s_idx = bvh[source_node_idx].child_indices[lane_id], s_start = bvh[source_node_idx].particle_start[lane_id], s_count = bvh[source_node_idx].particle_count[lane_id];
        float3 s_extents = float3(bvh[source_node_idx].max_x[lane_id] - bvh[source_node_idx].min_x[lane_id], bvh[source_node_idx].max_y[lane_id] - bvh[source_node_idx].min_y[lane_id], bvh[source_node_idx].max_z[lane_id] - bvh[source_node_idx].min_z[lane_id]);
        float s_size = max(s_extents.x, max(s_extents.y, s_extents.z));

        bool pass_mac = ((s_size + target_size) / max(length(s_com - target_com), 1e-6)) < pc.theta, pass_lod_thresh = (s_count <= pc.cluster_threshold) && !((my_p_idx >= s_start) && (my_p_idx < s_start + s_count));
        bool action_accumulate = s_valid && (pass_mac || pass_lod_thresh || s_is_leaf), action_traverse = s_valid && !action_accumulate;

        uint4 acc_ballot = WaveActiveBallot(action_accumulate);
        for (uint i = 0; i < 4; i++) {
            uint mask = acc_ballot[i];
            while (mask != 0) {
                uint bit = firstbitlow(mask); mask &= ~(1u << bit); uint src_lane = i * 32 + bit;
                if (i_am_valid) {
                    float3 k_com = float3(WaveReadLaneAt(s_com.x, src_lane), WaveReadLaneAt(s_com.y, src_lane), WaveReadLaneAt(s_com.z, src_lane));
                    float k_mass = WaveReadLaneAt(s_mass, src_lane); uint k_idx = WaveReadLaneAt(s_idx, src_lane); bool k_leaf = WaveReadLaneAt(s_is_leaf, src_lane);

                    if (!(k_leaf && my_p_idx == k_idx)) {
                        float3 p_dir = k_com - my_pos; float p_dist_sq = dot(p_dir, p_dir);
                        my_acc += (p_dir / max(sqrt(p_dist_sq), 1e-6)) * ((pc.G * k_mass) / (p_dist_sq + pc.softening_sq));
                    }
                }
            }
        }

        uint4 trav_ballot = WaveActiveBallot(action_traverse); if (action_traverse) shared_stacks_bh[subgroup_id][stack_ptr + get_ballot_prefix(trav_ballot, WaveGetLaneIndex())] = s_idx;
        if (lane_id == 0) shared_stack_ptrs_bh[subgroup_id] = stack_ptr + get_ballot_count(trav_ballot);
    }

    if (i_am_valid) {
        vk::BufferPointer<uint> w_u = cast_u2_u(pc.wrenches);
        float3 g_f = my_acc * my_mass;
        bda_atomic_add_float(w_u, my_p_idx * 6 + 0, g_f.x);
        bda_atomic_add_float(w_u, my_p_idx * 6 + 1, g_f.y);
        bda_atomic_add_float(w_u, my_p_idx * 6 + 2, g_f.z);
    }
}
#endif // KERNEL_barnes_hut

// --- motion_bounds ---
#ifdef KERNEL_motion_bounds
struct PushConstants { uint2 bvh; uint2 primitive_data; uint num_primitives; float dt; float particle_radius; };
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(256, 1, 1)]
void motion_bounds(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint idx = gl_GlobalInvocationID.x; if (idx >= pc.num_primitives) return;

    if (PRIMITIVE_TYPE == 0) {
        vk::BufferPointer<float> part_f = cast_u2_f(pc.primitive_data);
        vk::BufferPointer<MultiBvhNode> bvh = cast_u2_bvh(pc.bvh);

        uint base = (idx / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (idx % SUBGROUP_SIZE);
        float3 pos = float3(part_f[base+0], part_f[base+1*SUBGROUP_SIZE], part_f[base+2*SUBGROUP_SIZE]);
        float3 vel = float3(part_f[base+3*SUBGROUP_SIZE], part_f[base+4*SUBGROUP_SIZE], part_f[base+5*SUBGROUP_SIZE]);

        float3 p1 = pos + vel * pc.dt;
        float3 min_p = min(pos, p1) - (float3)pc.particle_radius, max_p = max(pos, p1) + (float3)pc.particle_radius;

        uint leaf_idx = (pc.num_primitives - 1) + idx;
        uint parent = bvh[leaf_idx].parent_idx;
        uint is_right = (bvh[parent].child_indices[1] == leaf_idx) ? 1 : 0;

        bvh[parent].min_x[is_right] = min_p.x; bvh[parent].max_x[is_right] = max_p.x;
        bvh[parent].min_y[is_right] = min_p.y; bvh[parent].max_y[is_right] = max_p.y;
        bvh[parent].min_z[is_right] = min_p.z; bvh[parent].max_z[is_right] = max_p.z;
    }
}
#endif