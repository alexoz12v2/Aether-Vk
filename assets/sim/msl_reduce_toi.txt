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
