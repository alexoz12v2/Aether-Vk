#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.h"
#include "imex_math.h"

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#define SUBGROUPS_PER_WG (256 / SUBGROUP_SIZE)

struct MultiBvhNode {
    float min_x[SUBGROUP_SIZE]; float max_x[SUBGROUP_SIZE];
    float min_y[SUBGROUP_SIZE]; float max_y[SUBGROUP_SIZE];
    float min_z[SUBGROUP_SIZE]; float max_z[SUBGROUP_SIZE];
    uint  child_indices[SUBGROUP_SIZE]; uint metadata[SUBGROUP_SIZE];
    float masses[SUBGROUP_SIZE];
    float com_x[SUBGROUP_SIZE]; float com_y[SUBGROUP_SIZE]; float com_z[SUBGROUP_SIZE];
    uint  particle_start[SUBGROUP_SIZE]; uint particle_count[SUBGROUP_SIZE];
    uint2 valid_mask;
    uint  parent_idx;
    uint  pad;
    uint  permutations[8][SUBGROUP_SIZE];
};

struct Wrench { uint force_x; uint force_y; uint force_z; uint torque_x; uint torque_y; uint torque_z; };

struct PushConstants {
    device uint* particles;
    device MultiBvhNode* bvh;
    device uint* cluster_list;
    device Wrench* wrenches;
    uint num_clusters;
    float dt;
    float theta;
    float G;
    float softening_sq;
    uint root_node_idx;
    uint cluster_threshold;
};

inline void AtomicAddFloat(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(addr, &expected, as_type<uint>(as_type<float>(expected) + val), memory_order_relaxed, memory_order_relaxed)) {
    }
}

inline bool bvh_node_is_valid(uint2 valid_mask, uint lane_id) {
    if (lane_id < 32) return (valid_mask.x & (1u << lane_id)) != 0u;
    else return (valid_mask.y & (1u << (lane_id - 32))) != 0u;
}

inline bool bvh_is_leaf(uint meta) { return (meta & 0x80000000u) != 0u; }

[[kernel]]
void barnes_hut(
    constant PushConstants& pc [[buffer(0)]],
    uint3 gl_WorkGroupID [[threadgroup_position_in_grid]],
    uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint gl_SubgroupID [[simdgroup_index_in_threadgroup]]
) {
    uint cluster_job_idx = gl_WorkGroupID.x * SUBGROUPS_PER_WG + gl_SubgroupID;
    if (cluster_job_idx >= pc.num_clusters) return;

    threadgroup uint shared_stacks[SUBGROUPS_PER_WG][64];
    threadgroup uint shared_stack_ptrs[SUBGROUPS_PER_WG];

    uint target_node_idx = pc.cluster_list[cluster_job_idx];
    device MultiBvhNode& t_node = pc.bvh[target_node_idx];
    bool i_am_valid = bvh_node_is_valid(t_node.valid_mask, lane_id);
    uint my_p_idx = t_node.child_indices[lane_id];

    float3 my_pos = float3(0.0);
    float my_mass = 0.0;
    if (i_am_valid) {
        uint base = (my_p_idx / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (my_p_idx % SUBGROUP_SIZE);
        my_pos = float3(
            as_type<float>(pc.particles[base]),
            as_type<float>(pc.particles[base + 1 * SUBGROUP_SIZE]),
            as_type<float>(pc.particles[base + 2 * SUBGROUP_SIZE])
        );
        my_mass = as_type<float>(pc.particles[base + 6 * SUBGROUP_SIZE]);
    }

    float3 safe_pos = i_am_valid ? my_pos : float3(0.0);
    float3 min_pos = simd_min(i_am_valid ? my_pos : float3(1e20));
    float3 max_pos = simd_max(i_am_valid ? my_pos : float3(-1e20));
    float3 cluster_extents = max_pos - min_pos;
    float target_size = max(cluster_extents.x, max(cluster_extents.y, cluster_extents.z));
    float sum_mass = simd_sum(i_am_valid ? my_mass : 0.0);
    float3 target_com = simd_sum(safe_pos * my_mass) / max(sum_mass, 1e-6f);

    float3 my_acc = float3(0.0);
    if (lane_id == 0) { 
        shared_stacks[gl_SubgroupID][0] = pc.root_node_idx; 
        shared_stack_ptrs[gl_SubgroupID] = 1; 
    }

    while (true) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint stack_ptr = shared_stack_ptrs[gl_SubgroupID]; 
        if (stack_ptr == 0) break;
        
        stack_ptr--;
        uint source_node_idx = shared_stacks[gl_SubgroupID][stack_ptr]; 
        if (lane_id == 0) shared_stack_ptrs[gl_SubgroupID] = stack_ptr;

        device MultiBvhNode& s_node = pc.bvh[source_node_idx];
        bool s_valid = bvh_node_is_valid(s_node.valid_mask, lane_id);
        bool s_is_leaf = bvh_is_leaf(s_node.metadata[lane_id]);

        float3 s_com = float3(s_node.com_x[lane_id], s_node.com_y[lane_id], s_node.com_z[lane_id]);
        float s_mass = s_node.masses[lane_id];
        uint s_idx = s_node.child_indices[lane_id];
        uint s_start = s_node.particle_start[lane_id];
        uint s_count = s_node.particle_count[lane_id];

        float3 s_extents = float3(
            s_node.max_x[lane_id] - s_node.min_x[lane_id],
            s_node.max_y[lane_id] - s_node.min_y[lane_id],
            s_node.max_z[lane_id] - s_node.min_z[lane_id]
        );
        float s_size = max(s_extents.x, max(s_extents.y, s_extents.z));

        bool pass_mac = ((s_size + target_size) / max(length(s_com - target_com), 1e-6f)) < pc.theta;
        bool pass_lod_thresh = (s_count <= pc.cluster_threshold) && !((my_p_idx >= s_start) && (my_p_idx < s_start + s_count));
        bool action_accumulate = s_valid && (pass_mac || pass_lod_thresh || s_is_leaf);
        bool action_traverse = s_valid && !action_accumulate;

        ulong acc_ballot = simd_ballot(action_accumulate);
        while (acc_ballot != 0) {
            uint src_lane = __builtin_ctzll(acc_ballot);
            acc_ballot &= ~(1ul << src_lane); 
            
            if (i_am_valid) {
                float3 k_com = float3(simd_broadcast(s_com.x, src_lane), simd_broadcast(s_com.y, src_lane), simd_broadcast(s_com.z, src_lane));
                float k_mass = simd_broadcast(s_mass, src_lane); 
                uint k_idx = simd_broadcast(s_idx, src_lane); 
                bool k_leaf = simd_broadcast(s_is_leaf, src_lane);

                if (!(k_leaf && my_p_idx == k_idx)) {
                    float3 p_dir = k_com - my_pos; 
                    float p_dist_sq = dot(p_dir, p_dir);
                    my_acc += (p_dir / max(sqrt(p_dist_sq), 1e-6f)) * ((pc.G * k_mass) / (p_dist_sq + pc.softening_sq));
                }
            }
        }

        uint prefix_count = simd_prefix_exclusive_sum(action_traverse ? 1 : 0);
        if (action_traverse) {
            shared_stacks[gl_SubgroupID][stack_ptr + prefix_count] = s_idx;
        }
        
        uint total_trav = simd_sum(action_traverse ? 1 : 0);
        if (lane_id == 0) {
            shared_stack_ptrs[gl_SubgroupID] = stack_ptr + total_trav;
        }
    }

    if (i_am_valid) {
        float3 g_f = my_acc * my_mass;
        device Wrench& w = pc.wrenches[my_p_idx];
        AtomicAddFloat((device atomic_uint*)&w.force_x, g_f.x);
        AtomicAddFloat((device atomic_uint*)&w.force_y, g_f.y);
        AtomicAddFloat((device atomic_uint*)&w.force_z, g_f.z);
    }
}