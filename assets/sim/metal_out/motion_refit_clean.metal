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
    device MultiBvhNode* bvh;
    device DepthIndices* depth_indices;
    uint total_nodes_at_depth;
};

[[kernel]]
void motion_refit(
    constant PushConstants& pc [[buffer(0)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= pc.total_nodes_at_depth) return;

    uint node_idx = pc.depth_indices->indices[global_id + 4];
    for (uint i = 0; i < 2; ++i) {
        uint child = pc.bvh[node_idx].child_indices[i];
        if (bvh_is_leaf(pc.bvh[node_idx].metadata[i])) {
            pc.bvh[node_idx].min_x[i] = pc.bvh[child].min_x[0];
            pc.bvh[node_idx].max_x[i] = pc.bvh[child].max_x[0];
            pc.bvh[node_idx].min_y[i] = pc.bvh[child].min_y[0];
            pc.bvh[node_idx].max_y[i] = pc.bvh[child].max_y[0];
            pc.bvh[node_idx].min_z[i] = pc.bvh[child].min_z[0];
            pc.bvh[node_idx].max_z[i] = pc.bvh[child].max_z[0];
        } else {
            pc.bvh[node_idx].min_x[i] = min(pc.bvh[child].min_x[0], pc.bvh[child].min_x[1]);
            pc.bvh[node_idx].max_x[i] = max(pc.bvh[child].max_x[0], pc.bvh[child].max_x[1]);
            pc.bvh[node_idx].min_y[i] = min(pc.bvh[child].min_y[0], pc.bvh[child].min_y[1]);
            pc.bvh[node_idx].max_y[i] = max(pc.bvh[child].max_y[0], pc.bvh[child].max_y[1]);
            pc.bvh[node_idx].min_z[i] = min(pc.bvh[child].min_z[0], pc.bvh[child].min_z[1]);
            pc.bvh[node_idx].max_z[i] = max(pc.bvh[child].max_z[0], pc.bvh[child].max_z[1]);
        }
    }
}
