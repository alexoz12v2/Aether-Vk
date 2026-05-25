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

struct ColliderId {
    uint entity_id;
    uint primitive_index;
};

struct PackedPair {
    ColliderId a;
    ColliderId b;
    float toi;
    float4 contact_normal;
    float4 contact_point;
    float penetration_depth;
};

struct PackedCollisions {
    uint dispatch_x;
    uint dispatch_y;
    uint dispatch_z;
    uint count;
    PackedPair pairs[1];
};

struct OutputTOI {
    atomic_uint min_tc_uint;
};

struct PushConstants {
    device void* particles;
    device PackedCollisions* collisions;
    device OutputTOI* out_toi;
    float particle_radius;
    float dt;
};

constant uint MAX_SUBGROUPS = 128 / 32;

[[kernel]]
void reduce_toi(
    constant PushConstants& pc [[buffer(0)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroups_per_threadgroup [[simdgroups_per_threadgroup]]
) {
    threadgroup uint shared_min_toi[MAX_SUBGROUPS];

    float tc = pc.dt; // Default to max time

    if (global_id < pc.collisions->count) {
        tc = pc.collisions->pairs[global_id].toi;
    }

    // Subgroup reduction
    float subgroup_min_tc = simd_min(tc);

    if (thread_index_in_simdgroup == 0) {
        shared_min_toi[simdgroup_index] = as_type<uint>(subgroup_min_tc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Workgroup reduction
    if (local_id == 0) {
        uint wg_min_uint = shared_min_toi[0];
        for (uint i = 1; i < simdgroups_per_threadgroup; i++) {
            wg_min_uint = min(wg_min_uint, shared_min_toi[i]);
        }

        // Global reduction
        atomic_fetch_min_explicit(&pc.out_toi->min_tc_uint, wg_min_uint, memory_order_relaxed);
    }
}
