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
    device RigidBody* rigid_bodies;
    device Wrench* wrenches;
    device ForceEmitter* emitters;
    float dt;
    uint n_bodies;
    uint n_iterations;
    uint num_emitters;
};

[[kernel]]
void integrate_bodies_p3(
    constant PushConstants& pc [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= pc.n_bodies) return;

    device RigidBody& body = pc.rigid_bodies[id];
    float mass = body.position_mass.w;
    float inv_m = (mass > 0.0) ? 1.0 / mass : 0.0;
    
    float3 I_inv = body.inertia_tensor_inv.xyz;
    float3 I_fwd = float3(
        (I_inv.x > 1e-14) ? 1.0 / I_inv.x : 0.0,
        (I_inv.y > 1e-14) ? 1.0 / I_inv.y : 0.0,
        (I_inv.z > 1e-14) ? 1.0 / I_inv.z : 0.0
    );

    float3 pos_n = body.position_mass.xyz;
    float4 q_n = body.orientation;
    float3 v_n = body.linear_vel_drag.xyz;
    float3 w_n = body.angular_vel_drag.xyz;

    uint w_idx = body.wrench_idx;
    device Wrench& wrench = pc.wrenches[w_idx];

    float3 f_n = float3(as_type<float>(wrench.force_x), as_type<float>(wrench.force_y), as_type<float>(wrench.force_z));
    float3 t_n = float3(as_type<float>(wrench.torque_x), as_type<float>(wrench.torque_y), as_type<float>(wrench.torque_z));

    for (uint e = 0; e < pc.num_emitters; ++e) {
        device ForceEmitter& emitter = pc.emitters[e];
        float3 em_pos = emitter.position;
        float em_mu = emitter.mu;
        float3 em_norm = emitter.normal;
        uint em_type = emitter.type_id;
        float em_trunc = emitter.trunc_distance;
        float em_scale = emitter.scale_factor;

        if (em_type == 0) {
            float3 r = em_pos - pos_n;
            float s_dist_sq = dot(r, r) * em_scale * em_scale;
            if (s_dist_sq > 1e-6) {
                float s_dist = sqrt(s_dist_sq);
                float s_dist3 = s_dist_sq * s_dist;
                float s_dist5 = s_dist3 * s_dist_sq;
                float softening = 1.0 - exp(-s_dist5);
                float force_mag = (em_mu * mass * softening) / s_dist_sq;
                f_n += normalize(r) * force_mag;
            }
        } else if (em_type == 1) {
            float dist = dot(pos_n - em_pos, em_norm);
            if (dist >= 0.0 && dist <= em_trunc) {
                f_n += em_norm * em_mu;
            }
        }
    }

    float half_dt = 0.5 * pc.dt;
    float3 a_lin = f_n * inv_m;
    float3 v_mid = v_n + half_dt * a_lin;
    float3 pos_next = pos_n + pc.dt * v_mid;
    float3 v_next = v_n + pc.dt * a_lin;

    float3 t_local = quat_rotate_inv(q_n, t_n);
    float3 w_n_local = quat_rotate_inv(q_n, w_n);
    float3 w_mid_local = w_n_local;

    for (uint iter = 0; iter < pc.n_iterations; ++iter) {
        float3 gyro = cross(w_mid_local, I_fwd * w_mid_local);
        float3 a_ang = I_inv * (t_local - gyro);
        w_mid_local = w_n_local + half_dt * a_ang;
    }

    float3 w_next_local = 2.0 * w_mid_local - w_n_local;
    float3 w_next = quat_rotate(q_n, w_next_local);
    float3 w_mid_world = quat_rotate(q_n, w_mid_local);
    float4 omega_pure = float4(w_mid_world, 0.0);
    float4 q_next = normalize(q_n + half_dt * quat_mult(omega_pure, q_n));

    body.position_mass = float4(pos_next, mass);
    body.orientation = q_next;
    body.linear_vel_drag = float4(v_next, body.linear_vel_drag.w);
    body.angular_vel_drag = float4(w_next, body.angular_vel_drag.w);

    wrench.force_x = 0;
    wrench.force_y = 0;
    wrench.force_z = 0;
    wrench.torque_x = 0;
    wrench.torque_y = 0;
    wrench.torque_z = 0;
}
