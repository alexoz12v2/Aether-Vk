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
#include <metal_stdlib>

using namespace metal;

struct PushConstants {
    device MultiBvhNode* tlas_bvh;
    device TLASLeaf* query_leaves;
    device PairBuffer* overlapping_pairs;
    uint tlas_root_index;
    uint total_queries;
};

[[kernel]]
void bp_scene(
    constant PushConstants& pc [[buffer(0)]],
    uint3 gl_WorkGroupID [[threadgroup_position_in_grid]],
    uint gl_SubgroupID [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    uint query_idx = gl_WorkGroupID.x * 8 + gl_SubgroupID;
    if (query_idx >= pc.total_queries) return;

    float3 my_min, my_max;
    uint my_ent_id;

    threadgroup uint shared_stacks[8][32];
    threadgroup uint shared_stack_ptrs[8];

    if (lane_id == 0) {
        my_min = pc.query_leaves[query_idx].min_bound;
        my_max = pc.query_leaves[query_idx].max_bound;
        my_ent_id = pc.query_leaves[query_idx].entity_idx;

        shared_stacks[gl_SubgroupID][0] = pc.tlas_root_index;
        shared_stack_ptrs[gl_SubgroupID] = 1;
    }

    my_min.x = simd_broadcast(my_min.x, 0);
    my_min.y = simd_broadcast(my_min.y, 0);
    my_min.z = simd_broadcast(my_min.z, 0);
    my_max.x = simd_broadcast(my_max.x, 0);
    my_max.y = simd_broadcast(my_max.y, 0);
    my_max.z = simd_broadcast(my_max.z, 0);
    my_ent_id = simd_broadcast(my_ent_id, 0);

    while (true) {
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint stack_ptr = shared_stack_ptrs[gl_SubgroupID];
        if (stack_ptr == 0) break;

        stack_ptr--;
        uint node_idx = shared_stacks[gl_SubgroupID][stack_ptr];
        if (lane_id == 0) shared_stack_ptrs[gl_SubgroupID] = stack_ptr;

        uint meta = pc.tlas_bvh[node_idx].metadata[lane_id];
        uint2 valid_mask = pc.tlas_bvh[node_idx].valid_mask;
        bool valid = bvh_node_is_valid(valid_mask, lane_id);

        float3 c_min = float3(
            pc.tlas_bvh[node_idx].min_x[lane_id],
            pc.tlas_bvh[node_idx].min_y[lane_id],
            pc.tlas_bvh[node_idx].min_z[lane_id]
        );
        float3 c_max = float3(
            pc.tlas_bvh[node_idx].max_x[lane_id],
            pc.tlas_bvh[node_idx].max_y[lane_id],
            pc.tlas_bvh[node_idx].max_z[lane_id]
        );
        uint child_payload = pc.tlas_bvh[node_idx].child_indices[lane_id];

        uint entity_id = bvh_get_index(meta);

        bool hit = valid && intersectAABB(my_min, my_max, c_min, c_max);
        bool is_leaf = bvh_is_leaf(meta);

        bool hit_leaf = hit && is_leaf && (my_ent_id < entity_id);
        bool hit_node = hit && !is_leaf;

        uint leaf_count = simd_sum(hit_leaf ? 1 : 0);
        uint leaf_offset = simd_prefix_exclusive_sum(hit_leaf ? 1 : 0);

        if (leaf_count > 0) {
            uint base_idx = 0;
            if (lane_id == 0) {
                base_idx = atomic_fetch_add_explicit(&pc.overlapping_pairs->count, leaf_count, memory_order_relaxed);
            }
            base_idx = simd_broadcast(base_idx, 0);

            if (hit_leaf && base_idx + leaf_offset < 10000u) {
                pc.overlapping_pairs->pairs[base_idx + leaf_offset] = uint2(my_ent_id, entity_id);
            }
        }

        uint node_count = simd_sum(hit_node ? 1 : 0);
        uint push_offset = simd_prefix_exclusive_sum(hit_node ? 1 : 0);

        if (hit_node) shared_stacks[gl_SubgroupID][stack_ptr + push_offset] = child_payload;
        if (lane_id == 0) shared_stack_ptrs[gl_SubgroupID] = stack_ptr + node_count;
    }
}
