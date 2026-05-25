#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
using namespace metal;

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

// Shared macros
#define P_READ(addr, offset) as_type<float>((addr)[(offset)])
#define P_WRITE(addr, offset, val) ((addr)[(offset)] = as_type<uint>(val))

// Typedefs
typedef device uint* ParticleData;

// Structs
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

struct DepthIndices {
    uint indices[1];
};

struct Wrench { uint force_x; uint force_y; uint force_z; uint torque_x; uint torque_y; uint torque_z; };

// Atomics
inline void AtomicAddFloatBDA(device uint* addr, uint offset, float val) {
    device atomic_uint* a = (device atomic_uint*)(addr + offset);
    uint e = atomic_load_explicit(a, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(a, &e, as_type<uint>(as_type<float>(e) + val), memory_order_relaxed, memory_order_relaxed));
}

// Bvh helpers
inline bool bvh_node_is_valid(uint2 mask, uint lane_id) {
    if (lane_id < 32) return (mask.x & (1u << lane_id)) != 0;
    return (mask.y & (1u << (lane_id - 32))) != 0;
}

inline bool bvh_is_leaf(uint metadata) {
    return (metadata & 0x80000000) != 0;
}

inline uint bvh_leaf_count(uint metadata) {
    return metadata & 0x7FFFFFFF;
}
// @assets/sim/bp_particle_self.comp

struct PushConstants_bp_particle_self {
    ulong bvh;
    ulong particles;
    ulong wrench_buffer;
    uint root_index;
    uint total_particles;
    float particle_radius;
    float stiffness;
};

[[kernel]] void bp_particle_self(
    constant PushConstants_bp_particle_self& pc [[buffer(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]
) {
    uint SUBGROUPS_PER_WG = 8; // 256 / 32
    uint my_p_idx = threadgroup_position_in_grid.x * SUBGROUPS_PER_WG + simdgroup_index_in_threadgroup;
    if (my_p_idx >= pc.total_particles) return;

    threadgroup uint shared_stacks[8][32];
    threadgroup uint shared_stack_ptrs[8];

    float3 my_pos, my_min, my_max;
    
    device atomic_uint* particles_buf = PTR(atomic_uint, pc.particles);

    if (thread_index_in_simdgroup == 0) {
        uint block_idx = my_p_idx / 32;
        uint local_idx = my_p_idx % 32;
        uint base = block_idx * 320 + local_idx;

        my_pos = float3(
            as_type<float>(atomic_load_explicit(&particles_buf[base + 0], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&particles_buf[base + 32], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&particles_buf[base + 64], memory_order_relaxed))
        );

        my_min = my_pos - pc.particle_radius;
        my_max = my_pos + pc.particle_radius;

        shared_stacks[simdgroup_index_in_threadgroup][0] = pc.root_index;
        shared_stack_ptrs[simdgroup_index_in_threadgroup] = 1;
    }

    my_pos = float3(simd_broadcast(my_pos.x, 0), simd_broadcast(my_pos.y, 0), simd_broadcast(my_pos.z, 0));
    my_min = float3(simd_broadcast(my_min.x, 0), simd_broadcast(my_min.y, 0), simd_broadcast(my_min.z, 0));
    my_max = float3(simd_broadcast(my_max.x, 0), simd_broadcast(my_max.y, 0), simd_broadcast(my_max.z, 0));
    my_p_idx = simd_broadcast(my_p_idx, 0);

    float3 local_repulsive_force(0.0f);

    device MultiBvhNode* bvh = PTR(MultiBvhNode, pc.bvh);

    while (true) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint stack_ptr = shared_stack_ptrs[simdgroup_index_in_threadgroup];
        if (stack_ptr == 0) break;

        stack_ptr--;
        uint node_idx = shared_stacks[simdgroup_index_in_threadgroup][stack_ptr];
        if (thread_index_in_simdgroup == 0) shared_stack_ptrs[simdgroup_index_in_threadgroup] = stack_ptr;

        uint meta = bvh[node_idx].met[thread_index_in_simdgroup];
        bool valid = is_vd(bvh[node_idx].vmk, thread_index_in_simdgroup);

        float3 c_min = float3(
            bvh[node_idx].mx[thread_index_in_simdgroup],
            bvh[node_idx].my[thread_index_in_simdgroup],
            bvh[node_idx].mz[thread_index_in_simdgroup]
        );
        float3 c_max = float3(
            bvh[node_idx].mxx[thread_index_in_simdgroup],
            bvh[node_idx].mxy[thread_index_in_simdgroup],
            bvh[node_idx].mxz[thread_index_in_simdgroup]
        );
        uint child_payload = bvh[node_idx].chd[thread_index_in_simdgroup];

        bool hit_aabb = valid && iAABB(my_min, my_max, c_min, c_max);
        bool is_leaf_node = is_lf(meta);

        bool hit_node = hit_aabb && !is_leaf_node;
        bool hit_leaf = hit_aabb && is_leaf_node && (my_p_idx != child_payload);

        ulong leaf_ballot = get_ballot(hit_leaf);

        while (leaf_ballot != 0) {
            uint bit = ctz(leaf_ballot);
            leaf_ballot &= ~(1ul << bit);

            uint other_idx = simd_shuffle(child_payload, bit);
            uint block_idx = other_idx / 32;
            uint local_idx = other_idx % 32;
            uint base_idx = block_idx * 320 + local_idx;

            float3 other_pos = float3(
                as_type<float>(atomic_load_explicit(&particles_buf[base_idx + 0], memory_order_relaxed)),
                as_type<float>(atomic_load_explicit(&particles_buf[base_idx + 32], memory_order_relaxed)),
                as_type<float>(atomic_load_explicit(&particles_buf[base_idx + 64], memory_order_relaxed))
            );

            float3 diff = my_pos - other_pos;
            float dist_sq = dot(diff, diff);
            float min_dist = pc.particle_radius * 2.0f;

            if (dist_sq > 1e-12f && dist_sq < min_dist * min_dist) {
                float dist = sqrt(dist_sq);
                float penetration = min_dist - dist;
                float3 normal = diff / dist;

                float force_mag = pc.stiffness * penetration;
                local_repulsive_force += normal * force_mag;
            }
        }

        ulong node_ballot = get_ballot(hit_node);
        uint node_count = popcount(node_ballot);
        uint push_offset = popcount(node_ballot & ((1ul << thread_index_in_simdgroup) - 1ul));

        if (hit_node) {
            shared_stacks[simdgroup_index_in_threadgroup][stack_ptr + push_offset] = child_payload;
        }
        if (thread_index_in_simdgroup == 0) shared_stack_ptrs[simdgroup_index_in_threadgroup] = stack_ptr + node_count;
    }

    local_repulsive_force = float3(simd_sum(local_repulsive_force.x), simd_sum(local_repulsive_force.y), simd_sum(local_repulsive_force.z));

    if (thread_index_in_simdgroup == 0 && dot(local_repulsive_force, local_repulsive_force) > 0.0f) {
        device Wrench* wr = PTR(Wrench, pc.wrench_buffer);
        atomic_add_f(&wr[my_p_idx].fx, local_repulsive_force.x);
        atomic_add_f(&wr[my_p_idx].fy, local_repulsive_force.y);
        atomic_add_f(&wr[my_p_idx].fz, local_repulsive_force.z);
    }
}
