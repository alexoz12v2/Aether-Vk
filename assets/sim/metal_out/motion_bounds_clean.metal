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


#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#ifndef PRIMITIVE_TYPE
#define PRIMITIVE_TYPE 0
#endif

struct PushConstants_motion_bounds {
    device MultiBvhNode* bvh;
    device uint* primitive_data;
    uint num_primitives;
    float dt;
    float particle_radius;
};

[[kernel]]
void motion_bounds(
    constant PushConstants_motion_bounds& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.num_primitives) return;

    if (PRIMITIVE_TYPE == 0) {
        uint base = (idx / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (idx % SUBGROUP_SIZE);
        
        float pos_x = as_type<float>(pc.primitive_data[base + 0]);
        float pos_y = as_type<float>(pc.primitive_data[base + 1 * SUBGROUP_SIZE]);
        float pos_z = as_type<float>(pc.primitive_data[base + 2 * SUBGROUP_SIZE]);
        float3 pos = float3(pos_x, pos_y, pos_z);
        
        float vel_x = as_type<float>(pc.primitive_data[base + 3 * SUBGROUP_SIZE]);
        float vel_y = as_type<float>(pc.primitive_data[base + 4 * SUBGROUP_SIZE]);
        float vel_z = as_type<float>(pc.primitive_data[base + 5 * SUBGROUP_SIZE]);
        float3 vel = float3(vel_x, vel_y, vel_z);

        float3 p1 = pos + vel * pc.dt;
        float3 min_p = min(pos, p1) - pc.particle_radius;
        float3 max_p = max(pos, p1) + pc.particle_radius;

        uint leaf_idx = (pc.num_primitives - 1) + idx;
        uint parent = pc.bvh[leaf_idx].parent_idx;
        uint is_right = (pc.bvh[parent].child_indices[1] == leaf_idx) ? 1 : 0;

        pc.bvh[parent].min_x[is_right] = min_p.x; 
        pc.bvh[parent].max_x[is_right] = max_p.x;
        pc.bvh[parent].min_y[is_right] = min_p.y; 
        pc.bvh[parent].max_y[is_right] = max_p.y;
        pc.bvh[parent].min_z[is_right] = min_p.z; 
        pc.bvh[parent].max_z[is_right] = max_p.z;
    }
}
