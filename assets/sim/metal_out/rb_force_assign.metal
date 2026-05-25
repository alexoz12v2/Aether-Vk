#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.msl"
#include "imex_math.msl"

struct PushConstants {
    device RigidBody* rigid_bodies;
    device Wrench* wrenches;
    uint n_bodies;
    uint _pad;
};

inline void atomic_add_float(device uint* addr, float val) {
    device atomic_uint* atomic_addr = reinterpret_cast<device atomic_uint*>(addr);
    uint old_val = atomic_load_explicit(atomic_addr, memory_order_relaxed);
    uint assumed_val, new_val;
    do {
        assumed_val = old_val;
        new_val = as_type<uint>(as_type<float>(assumed_val) + val);
    } while (!atomic_compare_exchange_weak_explicit(atomic_addr, &old_val, new_val, memory_order_relaxed, memory_order_relaxed));
}

[[kernel]]
void rb_force_assign(
    constant PushConstants& pc [[push_constant]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint threads_per_simdgroup [[thread_execution_width]]
) {
    uint body_id = threadgroup_position_in_grid.x;
    if (body_id >= pc.n_bodies) return;

    uint local_id = thread_position_in_threadgroup.x;
    
    device RigidBody& body = pc.rigid_bodies[body_id];
    uint leaf_start = body.leaf_start_idx;
    uint leaf_count = body.leaf_count;
    uint com_wrench = body.wrench_idx;

    float3 acc_f = float3(0.0);
    float3 acc_t = float3(0.0);

    for (uint i = local_id; i < leaf_count; i += 128u) {
        device Wrench& lw = pc.wrenches[leaf_start + i];
        acc_f += float3(as_type<float>(lw.force_x), as_type<float>(lw.force_y), as_type<float>(lw.force_z));
        acc_t += float3(as_type<float>(lw.torque_x), as_type<float>(lw.torque_y), as_type<float>(lw.torque_z));
    }

    acc_f.x = simd_sum(acc_f.x);
    acc_f.y = simd_sum(acc_f.y);
    acc_f.z = simd_sum(acc_f.z);
    
    acc_t.x = simd_sum(acc_t.x);
    acc_t.y = simd_sum(acc_t.y);
    acc_t.z = simd_sum(acc_t.z);

    threadgroup float3 sh_f[32]; // Max 32 subgroups if thread_execution_width is 4
    threadgroup float3 sh_t[32];

    if (simd_lane_id == 0u) {
        sh_f[simd_group_id] = acc_f;
        sh_t[simd_group_id] = acc_t;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0u) {
        float3 total_f = float3(0.0);
        float3 total_t = float3(0.0);
        uint subgroups_per_wg = 128u / threads_per_simdgroup;
        for (uint s = 0u; s < subgroups_per_wg; ++s) {
            total_f += sh_f[s];
            total_t += sh_t[s];
        }
        
        device Wrench& cw = pc.wrenches[com_wrench];
        atomic_add_float(&cw.force_x, total_f.x);
        atomic_add_float(&cw.force_y, total_f.y);
        atomic_add_float(&cw.force_z, total_f.z);
        atomic_add_float(&cw.torque_x, total_t.x);
        atomic_add_float(&cw.torque_y, total_t.y);
        atomic_add_float(&cw.torque_z, total_t.z);
    }
}
