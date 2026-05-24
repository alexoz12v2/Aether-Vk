#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.glsl"
#include "imex_math.glsl"
#include "physics_core.glsl"

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

[[kernel]]
void narrow_ccd(
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    constant PushConstants& pc [[buffer(0)]]
) {
    uint pair_idx = thread_position_in_grid.x;
    
    uint idA, idB, lca_id;
    bool is_partA = false, is_partB = false;

    if (pc.space_type == 1) { // Cross
        device atomic_uint* cross_pairs_count_ptr = (device atomic_uint*)pc.cross_pair_buffer;
        uint cross_pairs_count = atomic_load_explicit(cross_pairs_count_ptr, memory_order_relaxed);
        if (pair_idx >= cross_pairs_count) return;
        device CrossPair* pairs = (device CrossPair*)(pc.cross_pair_buffer + 16);
        CrossPair pair = pairs[pair_idx];
        idA = pair.macro_id;
        idB = pair.micro_id;
        lca_id = pair.lca_id;
    } else { // Standard
        device atomic_uint* pair_buffer_count_ptr = (device atomic_uint*)pc.pair_buffer;
        uint pair_buffer_count = atomic_load_explicit(pair_buffer_count_ptr, memory_order_relaxed);
        if (pair_idx >= pair_buffer_count) return;
        device uint2* pairs = (device uint2*)(pc.pair_buffer + 8);
        uint2 pair = pairs[pair_idx];
        idA = pair.x;
        idB = pair.y;
    }

    float3 pos_A, vel_A, extents_A;
    uint shape_A;
    float4 orient_A = float4(0, 0, 0, 1);

    if (idA == 0xFFFFFFFFu) { 
        is_partA = true;
    }
    
    device RigidBody* bodies = (device RigidBody*)pc.scene_entities;
    RigidBody ent_A = bodies[idA];
    RigidBody ent_B = bodies[idB];
    
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

    float4x4 trans_A = float4x4(1.0);
    float4x4 trans_B = float4x4(1.0);
    
    if (pc.space_type == 1) {
        device LcaEntity* lca_entities = (device LcaEntity*)pc.lca_entities;
        LcaEntity lca = lca_entities[lca_id];
        float3 macro_rel_vel_au = vel_A - lca.linear_velocity;
        pos_A = (lca.inv_transform * float4(pos_A, 1.0)).xyz * AU_TO_KM;
        float3x3 lca_inv_trans_3x3 = float3x3(lca.inv_transform.columns[0].xyz, lca.inv_transform.columns[1].xyz, lca.inv_transform.columns[2].xyz);
        vel_A = (lca_inv_trans_3x3 * macro_rel_vel_au) * AU_TO_KM;
        extents_A *= AU_TO_KM;
        trans_A = lca.inv_transform; 
    }
    
    float3x3 rotA = quat_to_mat3(orient_A);
    trans_A = float4x4(
        float4(rotA.columns[0], 0),
        float4(rotA.columns[1], 0),
        float4(rotA.columns[2], 0),
        float4(pos_A, 1.0)
    );
    
    float3x3 rotB = quat_to_mat3(orient_B);
    trans_B = float4x4(
        float4(rotB.columns[0], 0),
        float4(rotB.columns[1], 0),
        float4(rotB.columns[2], 0),
        float4(pos_B, 1.0)
    );

    float toi, depth;
    float3 normal, contact;
    
    if (compute_toi_generic(shape_A, extents_A, trans_A, vel_A, shape_B, extents_B, trans_B, vel_B, 1e-3, 10, toi, normal, contact, depth)) {
        if (pc.space_type == 1) {
            device atomic_uint* count_ptr = (device atomic_uint*)pc.cross_output_list;
            uint count = atomic_fetch_add_explicit(count_ptr, 1, memory_order_relaxed);
            if (count < 4000u) {
                device CrossPair* pairs = (device CrossPair*)(pc.cross_output_list + 16);
                pairs[count].valid = 1u;
                pairs[count].macro_id = idA;
                pairs[count].micro_id = idB;
                pairs[count].lca_id = lca_id;
                pairs[count].toi = toi;
                pairs[count].contact_normal = float4(normal, 0.0);
                pairs[count].contact_point = float4(contact, 1.0);
                pairs[count].penetration_depth = depth;
            }
        } else {
            device atomic_uint* count_ptr = (device atomic_uint*)pc.output_list;
            uint count = atomic_fetch_add_explicit(count_ptr, 1, memory_order_relaxed);
            if (count < 4000u) {
                device SparseCollisionPair* pairs = (device SparseCollisionPair*)(pc.output_list + 16);
                pairs[count].entity_a = idA;
                pairs[count].prim_a = idA;
                pairs[count].entity_b = idB;
                pairs[count].prim_b = idB;
                pairs[count].toi = toi;
                pairs[count].contact_normal = float4(normal, 0.0);
                pairs[count].contact_point = float4(contact, 1.0);
                pairs[count].penetration_depth = depth;
                pairs[count].bda_a = pc.scene_entities + idA * 128u;
                pairs[count].bda_b = pc.scene_entities + idB * 128u;
                pairs[count].frame_bda = 0; 
                pairs[count].valid = 1u;
            }
        }
    }
}
