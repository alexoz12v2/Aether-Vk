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
// @assets/sim/integrate_particles_p1_p2.comp
//
// Particle Velocity-Verlet Predictor — Phase 1 & 2
// ─────────────────────────────────────────────────
// Frame-start invariant: AOSOA slots 7/8/9 hold F(x_n) from the previous frame.
//
//   v_{n+½} = v_n + (dt/2) · M⁻¹ · F(x_n)     [half-kick]
//   x_{n+1} = x_n + dt · v_{n+½}               [full position leap]
//
// After writing, CLEARS slots 7/8/9 to 0 so the unified force-generation pass
// (barnes_hut, bp_particle_self, narrow-phase) can safely atomicAdd into them.
// The half-kick velocity v_{n+½} is stored temporarily in slots 3/4/5 for
// integrate_particles_p4_5 to complete the VV corrector step.
//
// Target: SPIR-V 1.4 · Vulkan 1.1 · flexible across all hardware subgroup sizes.





struct PushConstants_integrate_particles_p1_p2 {
    ParticleData particles;
    float dt;
    uint total_particles;
};


[[kernel]]
void integrate_particles_p1_p2(constant PushConstants_integrate_particles_p1_p2& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint gid = thread_position_in_grid.x;
    if (gid >= pc.total_particles) return;

    uint base = (gid / SUBGROUP_SIZE) * (10u * SUBGROUP_SIZE) + (gid % SUBGROUP_SIZE);

    float mass = P_READ(pc.particles, base + 6u * SUBGROUP_SIZE);
    if (mass <= 0.0) return;

    float inv_m = 1.0 / mass;
    float half_dt = 0.5 * pc.dt;

    float3 v_n = float3(P_READ(pc.particles, base + 3u * SUBGROUP_SIZE), P_READ(pc.particles, base + 4u * SUBGROUP_SIZE), P_READ(pc.particles, base + 5u * SUBGROUP_SIZE));
    float3 f_n = float3(P_READ(pc.particles, base + 7u * SUBGROUP_SIZE), P_READ(pc.particles, base + 8u * SUBGROUP_SIZE), P_READ(pc.particles, base + 9u * SUBGROUP_SIZE));

    float3 v_half = v_n + f_n * inv_m * half_dt;
    float3 pos_n = float3(P_READ(pc.particles, base + 0u * SUBGROUP_SIZE), P_READ(pc.particles, base + 1u * SUBGROUP_SIZE), P_READ(pc.particles, base + 2u * SUBGROUP_SIZE));
    float3 pos_next = pos_n + v_half * pc.dt;

    P_WRITE(pc.particles, base + 0u * SUBGROUP_SIZE, pos_next.x);
    P_WRITE(pc.particles, base + 1u * SUBGROUP_SIZE, pos_next.y);
    P_WRITE(pc.particles, base + 2u * SUBGROUP_SIZE, pos_next.z);

    P_WRITE(pc.particles, base + 3u * SUBGROUP_SIZE, v_half.x);
    P_WRITE(pc.particles, base + 4u * SUBGROUP_SIZE, v_half.y);
    P_WRITE(pc.particles, base + 5u * SUBGROUP_SIZE, v_half.z);

    P_WRITE(pc.particles, base + 7u * SUBGROUP_SIZE, 0.0);
    P_WRITE(pc.particles, base + 8u * SUBGROUP_SIZE, 0.0);
    P_WRITE(pc.particles, base + 9u * SUBGROUP_SIZE, 0.0);
}
