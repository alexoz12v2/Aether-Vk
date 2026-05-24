#include <metal_stdlib>
using namespace metal;

#include "../debug_utils.metal"
#include "../bvh_utils.metal"

struct PushConstants {
    device LcaEntity* lca_entities;
    device TLASLeaf* macro_leaves;
    device EntityHeader* entity_headers;
    device PairBuffer* lca_query_pairs;
    device PairBuffer* out_rb_rb;
    device PairBuffer* out_rb_ps;
    device PairBuffer* out_ps_ps;
    device CrossPairBuffer* out_cross_pairs;
    device MultiBvhNode* tlas_bvh;
    uint total_queries;
    uint max_pairs;
};

#define AU_TO_KM 149597870.7

void transform_aabb_macro_to_micro(float3 lca_center, float lca_scale, float3 macro_center_au, float3 macro_extents_au, thread float3& out_min, thread float3& out_max) {
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
    out_min = float3(1e20); out_max = float3(-1e20);
    for (int i = 0; i < 8; i++) {
        float3 local_p = (corners[i] - lca_center) / lca_scale;
        out_min = min(out_min, local_p);
        out_max = max(out_max, local_p);
    }
}

[[kernel]]
void bp_cross_lca(
    constant PushConstants& pc [[buffer(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]
) {
    uint lane_id = thread_index_in_simdgroup;
    uint subgroup_id = thread_index_in_threadgroup / threads_per_simdgroup;
    uint query_idx = threadgroup_position_in_grid.x * (256 / threads_per_simdgroup) + subgroup_id;

    if (query_idx >= pc.total_queries || query_idx >= pc.lca_query_pairs->count) return;

    uint2 query = pc.lca_query_pairs->pairs[query_idx];
    uint macro_ent_id = query.x;
    uint lca_ent_id = query.y;
    float3 query_min, query_max;

    threadgroup uint shared_stacks[8][32]; // Max subgroups is 256/32 = 8
    threadgroup uint shared_stack_ptrs[8];
    threadgroup device MultiBvhNode* shared_lca_bvh_addr[8];

    if (lane_id == 0) {
        LcaEntity l_ent = pc.lca_entities[lca_ent_id];
        shared_lca_bvh_addr[subgroup_id] = pc.tlas_bvh;
        
        TLASLeaf macro_leaf = pc.macro_leaves[macro_ent_id];
        float3 macro_min = macro_leaf.min_bound;
        float3 macro_max = macro_leaf.max_bound;

        float3 center_au = (macro_min + macro_max) * 0.5;
        float3 extents_au = (macro_max - macro_min) * 0.5;

        transform_aabb_macro_to_micro(l_ent.center_pos, l_ent.scale, center_au, extents_au, query_min, query_max);

        shared_stacks[subgroup_id][0] = l_ent.bvh_root_index;
        shared_stack_ptrs[subgroup_id] = 1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    query_min = simd_broadcast(query_min, 0);
    query_max = simd_broadcast(query_max, 0);
    macro_ent_id = simd_broadcast(macro_ent_id, 0);

    device MultiBvhNode* tlas = shared_lca_bvh_addr[subgroup_id];

    while (true) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint stack_ptr = shared_stack_ptrs[subgroup_id];
        if (stack_ptr == 0) break;

        stack_ptr--;
        uint node_idx = shared_stacks[subgroup_id][stack_ptr];
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr;

        uint meta = tlas[node_idx].metadata[lane_id];
        bool valid = bvh_node_is_valid(tlas[node_idx].valid_mask, lane_id);
        
        float3 c_min = float3(tlas[node_idx].min_x[lane_id], tlas[node_idx].min_y[lane_id], tlas[node_idx].min_z[lane_id]);
        float3 c_max = float3(tlas[node_idx].max_x[lane_id], tlas[node_idx].max_y[lane_id], tlas[node_idx].max_z[lane_id]);
        uint child_payload = tlas[node_idx].child_indices[lane_id];

        bool hit = valid && intersectAABB(query_min, query_max, c_min, c_max);
        bool is_leaf = bvh_is_leaf(meta);

        bool hit_leaf = hit && is_leaf;
        bool hit_node = hit && !is_leaf;

        uint leaf_count = simd_sum(hit_leaf ? 1 : 0);
        uint leaf_offset = simd_prefix_exclusive_sum(hit_leaf ? 1 : 0);

        if (leaf_count > 0) {
            uint base_idx = 0;
            if (lane_id == 0) {
                base_idx = atomic_fetch_add_explicit((device atomic_uint*)&pc.out_cross_pairs->count, leaf_count, memory_order_relaxed);
            }
            base_idx = simd_broadcast(base_idx, 0);

            if (hit_leaf && (base_idx + leaf_offset) < pc.max_pairs) {
                pc.out_cross_pairs->pairs[base_idx + leaf_offset].macro_id = macro_ent_id;
                pc.out_cross_pairs->pairs[base_idx + leaf_offset].micro_id = bvh_get_index(meta);
                pc.out_cross_pairs->pairs[base_idx + leaf_offset].lca_id = lca_ent_id;
            }
        }

        uint subgroup_size = threads_per_simdgroup;
        for (uint src_lane = 0; src_lane < subgroup_size; src_lane++) {
            bool src_hit_leaf = simd_broadcast(hit_leaf, src_lane);
            if (src_hit_leaf) {
                uint micro_ent_id = bvh_get_index(simd_broadcast(meta, src_lane));

                if (lane_id == 0) {
                    uint type_A = pc.entity_headers[macro_ent_id].ty;
                    uint type_B = pc.entity_headers[micro_ent_id].ty;
                    uint ent_A = macro_ent_id;
                    uint ent_B = micro_ent_id;

                    if (type_A > type_B) {
                        uint temp = ent_A; ent_A = ent_B; ent_B = temp;
                        temp = type_A; type_A = type_B; type_B = temp;
                    }

                    if (type_A == TYPE_RIGID_BODY && type_B == TYPE_RIGID_BODY) {
                        uint out_idx = atomic_fetch_add_explicit((device atomic_uint*)&pc.out_rb_rb->count, 1, memory_order_relaxed);
                        if (out_idx < pc.max_pairs) pc.out_rb_rb->pairs[out_idx] = uint2(ent_A, ent_B);
                    } else if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_RIGID_BODY) {
                        uint out_idx = atomic_fetch_add_explicit((device atomic_uint*)&pc.out_rb_ps->count, 1, memory_order_relaxed);
                        if (out_idx < pc.max_pairs) pc.out_rb_ps->pairs[out_idx] = uint2(ent_B, ent_A);
                    } else if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_PARTICLE_SYSTEM) {
                        uint out_idx = atomic_fetch_add_explicit((device atomic_uint*)&pc.out_ps_ps->count, 1, memory_order_relaxed);
                        if (out_idx < pc.max_pairs) pc.out_ps_ps->pairs[out_idx] = uint2(ent_A, ent_B);
                    }
                }
            }
        }

        uint node_count = simd_sum(hit_node ? 1 : 0);
        uint push_offset = simd_prefix_exclusive_sum(hit_node ? 1 : 0);

        if (hit_node) shared_stacks[subgroup_id][stack_ptr + push_offset] = child_payload;
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr + node_count;
    }
}