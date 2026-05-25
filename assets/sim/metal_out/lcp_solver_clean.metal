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

#define MAX_BODIES_PER_ISLAND 32
#define SUBGROUP_SIZE 32

struct PushConstants {
    device ParticleData* particles;
    device PackedCollisions* collisions;
    device ImpulseOutput* outputs;
    uint total_clusters;
    device RigidBodyArray* rigid_bodies;
    float dt;
    float restitution;
};

void generate_tangents(float3 normal, thread float3& t1, thread float3& t2) {
    if (abs(normal.x) >= 0.57735) {
        t1 = normalize(float3(normal.y, -normal.x, 0.0));
    } else {
        t1 = normalize(float3(0.0, normal.z, -normal.y));
    }
    t2 = cross(normal, t1);
}

float compute_effective_mass(float3 dir, float3 rA, float3 rB, float invMA, float invMB, float3 invIA, float3 invIB, float4 qA, float4 qB) {
    float3 I_crossA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, dir)));
    float3 I_crossB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, dir)));
    return 1.0 / max(invMA + invMB + dot(I_crossA, cross(rA, dir)) + dot(I_crossB, cross(rB, dir)), 1e-6);
}

void AtomicAddFloatShared(threadgroup atomic_uint& dest, float val) {
    uint old_val = atomic_load_explicit(&dest, memory_order_relaxed);
    uint assumed_val;
    uint new_val;
    do {
        assumed_val = old_val;
        new_val = as_type<uint>(as_type<float>(assumed_val) + val);
    } while (!atomic_compare_exchange_weak_explicit(&dest, &old_val, new_val, memory_order_relaxed, memory_order_relaxed));
}

[[kernel]]
void lcp_solver(
    constant PushConstants& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]])
{
    uint local_id = thread_index_in_threadgroup;
    uint contact_idx = thread_position_in_grid.x;
    
    device PackedCollisions* collisions = pc.collisions;
    device RigidBodyArray* rigid_bodies = pc.rigid_bodies;
    device ParticleData* particles = pc.particles;
    device ImpulseOutput* outputs = pc.outputs;
    
    uint collisions_count = collisions->count;
    bool valid = (contact_idx < collisions_count);

    threadgroup atomic_uint shared_v_x[MAX_BODIES_PER_ISLAND];
    threadgroup atomic_uint shared_v_y[MAX_BODIES_PER_ISLAND];
    threadgroup atomic_uint shared_v_z[MAX_BODIES_PER_ISLAND];
    threadgroup atomic_uint shared_w_x[MAX_BODIES_PER_ISLAND];
    threadgroup atomic_uint shared_w_y[MAX_BODIES_PER_ISLAND];
    threadgroup atomic_uint shared_w_z[MAX_BODIES_PER_ISLAND];
    threadgroup float accumulated_normal[128];
    threadgroup float accumulated_t1[128];
    threadgroup float accumulated_t2[128];

    accumulated_normal[local_id] = 0.0;
    accumulated_t1[local_id] = 0.0;
    accumulated_t2[local_id] = 0.0;

    if (local_id < MAX_BODIES_PER_ISLAND) {
        RigidBody rb = rigid_bodies->bodies[local_id];
        atomic_store_explicit(&shared_v_x[local_id], as_type<uint>(rb.linear_vel_drag.x), memory_order_relaxed);
        atomic_store_explicit(&shared_v_y[local_id], as_type<uint>(rb.linear_vel_drag.y), memory_order_relaxed);
        atomic_store_explicit(&shared_v_z[local_id], as_type<uint>(rb.linear_vel_drag.z), memory_order_relaxed);
        atomic_store_explicit(&shared_w_x[local_id], as_type<uint>(rb.angular_vel_drag.x), memory_order_relaxed);
        atomic_store_explicit(&shared_w_y[local_id], as_type<uint>(rb.angular_vel_drag.y), memory_order_relaxed);
        atomic_store_explicit(&shared_w_z[local_id], as_type<uint>(rb.angular_vel_drag.z), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!valid) return;

    PackedPair pair = collisions->pairs[contact_idx];
    bool is_partA = (pair.a.entity_id == 0xFFFFFFFFu);
    bool is_partB = (pair.b.entity_id == 0xFFFFFFFFu);
    uint idA = pair.a.primitive_index;
    uint idB = pair.b.primitive_index;

    float invMA = 0.0;
    float invMB = 0.0;
    float3 invIA = float3(0.0);
    float3 invIB = float3(0.0);
    float4 qA = float4(0, 0, 0, 1);
    float4 qB = float4(0, 0, 0, 1);
    float3 posA = float3(0.0);
    float3 posB = float3(0.0);
    float3 vA_init = float3(0.0);
    float3 wA_init = float3(0.0);
    float3 vB_init = float3(0.0);
    float3 wB_init = float3(0.0);

    if (is_partA) {
        uint baseA = (idA / SUBGROUP_SIZE) * 10u * SUBGROUP_SIZE + (idA % SUBGROUP_SIZE);
        posA = float3(
            as_type<float>(particles->data[baseA]),
            as_type<float>(particles->data[baseA + SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseA + 2 * SUBGROUP_SIZE])
        );
        vA_init = float3(
            as_type<float>(particles->data[baseA + 3 * SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseA + 4 * SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseA + 5 * SUBGROUP_SIZE])
        );
        float mass = as_type<float>(particles->data[baseA + 6u * SUBGROUP_SIZE]);
        invMA = (mass > 0.0) ? 1.0 / mass : 0.0;
    } else {
        RigidBody rbA = rigid_bodies->bodies[idA];
        invMA = rbA.position_mass.w > 0.0 ? 1.0 / rbA.position_mass.w : 0.0;
        invIA = rbA.inertia_tensor_inv.xyz;
        qA = rbA.orientation;
        posA = rbA.position_mass.xyz;
        vA_init = rbA.linear_vel_drag.xyz;
        wA_init = rbA.angular_vel_drag.xyz;
    }

    if (is_partB) {
        uint baseB = (idB / SUBGROUP_SIZE) * 10u * SUBGROUP_SIZE + (idB % SUBGROUP_SIZE);
        posB = float3(
            as_type<float>(particles->data[baseB]),
            as_type<float>(particles->data[baseB + SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseB + 2 * SUBGROUP_SIZE])
        );
        vB_init = float3(
            as_type<float>(particles->data[baseB + 3 * SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseB + 4 * SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseB + 5 * SUBGROUP_SIZE])
        );
        float mass = as_type<float>(particles->data[baseB + 6u * SUBGROUP_SIZE]);
        invMB = (mass > 0.0) ? 1.0 / mass : 0.0;
    } else {
        RigidBody rbB = rigid_bodies->bodies[idB];
        invMB = rbB.position_mass.w > 0.0 ? 1.0 / rbB.position_mass.w : 0.0;
        invIB = rbB.inertia_tensor_inv.xyz;
        qB = rbB.orientation;
        posB = rbB.position_mass.xyz;
        vB_init = rbB.linear_vel_drag.xyz;
        wB_init = rbB.angular_vel_drag.xyz;
    }

    float3 n = pair.contact_normal.xyz;
    float3 t1, t2;
    generate_tangents(n, t1, t2);
    float3 rA = pair.contact_point.xyz - posA;
    float3 rB = pair.contact_point.xyz - posB;
    
    float eff_m_n = compute_effective_mass(n, rA, rB, invMA, invMB, invIA, invIB, qA, qB);
    float eff_m_t1 = compute_effective_mass(t1, rA, rB, invMA, invMB, invIA, invIB, qA, qB);
    float eff_m_t2 = compute_effective_mass(t2, rA, rB, invMA, invMB, invIA, invIB, qA, qB);

    float3 v_rel_init = (vB_init + cross(wB_init, rB)) - (vA_init + cross(wA_init, rA));
    float bounce = dot(v_rel_init, n) < -0.1 ? -pc.restitution * dot(v_rel_init, n) : 0.0;
    float target_v_n = bounce + ((0.2 / max(pc.dt, 1e-6)) * max(pair.penetration_depth - 0.01, 0.0));

    for (int iter = 0; iter < 20; ++iter) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float3 vA = vA_init;
        float3 wA = wA_init;
        if (!is_partA && idA < MAX_BODIES_PER_ISLAND) {
            vA = float3(as_type<float>(atomic_load_explicit(&shared_v_x[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_y[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_z[idA], memory_order_relaxed)));
            wA = float3(as_type<float>(atomic_load_explicit(&shared_w_x[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_y[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_z[idA], memory_order_relaxed)));
        }

        float3 vB = vB_init;
        float3 wB = wB_init;
        if (!is_partB && idB < MAX_BODIES_PER_ISLAND) {
            vB = float3(as_type<float>(atomic_load_explicit(&shared_v_x[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_y[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_z[idB], memory_order_relaxed)));
            wB = float3(as_type<float>(atomic_load_explicit(&shared_w_x[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_y[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_z[idB], memory_order_relaxed)));
        }

        float3 v_rel = (vB + cross(wB, rB)) - (vA + cross(wA, rA));
        float jn_delta = eff_m_n * (-dot(v_rel, n) + target_v_n);
        float old_jn = accumulated_normal[local_id];
        float new_jn = max(old_jn + jn_delta, 0.0);
        jn_delta = new_jn - old_jn;
        accumulated_normal[local_id] = new_jn;
        float3 P_n = jn_delta * n;

        if (!is_partA && invMA > 0.0 && idA < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idA], -P_n.x * invMA);
            AtomicAddFloatShared(shared_v_y[idA], -P_n.y * invMA);
            AtomicAddFloatShared(shared_v_z[idA], -P_n.z * invMA);
            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -P_n)));
            AtomicAddFloatShared(shared_w_x[idA], dwA.x);
            AtomicAddFloatShared(shared_w_y[idA], dwA.y);
            AtomicAddFloatShared(shared_w_z[idA], dwA.z);
        }
        if (!is_partB && invMB > 0.0 && idB < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idB], P_n.x * invMB);
            AtomicAddFloatShared(shared_v_y[idB], P_n.y * invMB);
            AtomicAddFloatShared(shared_v_z[idB], P_n.z * invMB);
            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, P_n)));
            AtomicAddFloatShared(shared_w_x[idB], dwB.x);
            AtomicAddFloatShared(shared_w_y[idB], dwB.y);
            AtomicAddFloatShared(shared_w_z[idB], dwB.z);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!is_partA && idA < MAX_BODIES_PER_ISLAND) {
            vA = float3(as_type<float>(atomic_load_explicit(&shared_v_x[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_y[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_z[idA], memory_order_relaxed)));
            wA = float3(as_type<float>(atomic_load_explicit(&shared_w_x[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_y[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_z[idA], memory_order_relaxed)));
        }
        if (!is_partB && idB < MAX_BODIES_PER_ISLAND) {
            vB = float3(as_type<float>(atomic_load_explicit(&shared_v_x[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_y[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_z[idB], memory_order_relaxed)));
            wB = float3(as_type<float>(atomic_load_explicit(&shared_w_x[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_y[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_z[idB], memory_order_relaxed)));
        }
        v_rel = (vB + cross(wB, rB)) - (vA + cross(wA, rA));

        float max_fric = 0.5 * accumulated_normal[local_id];
        float jt1_delta = eff_m_t1 * (-dot(v_rel, t1));
        float old_jt1 = accumulated_t1[local_id];
        float new_jt1 = clamp(old_jt1 + jt1_delta, -max_fric, max_fric);
        jt1_delta = new_jt1 - old_jt1;
        accumulated_t1[local_id] = new_jt1;

        float jt2_delta = eff_m_t2 * (-dot(v_rel, t2));
        float old_jt2 = accumulated_t2[local_id];
        float new_jt2 = clamp(old_jt2 + jt2_delta, -max_fric, max_fric);
        jt2_delta = new_jt2 - old_jt2;
        accumulated_t2[local_id] = new_jt2;

        float3 P_t = jt1_delta * t1 + jt2_delta * t2;

        if (!is_partA && invMA > 0.0 && idA < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idA], -P_t.x * invMA);
            AtomicAddFloatShared(shared_v_y[idA], -P_t.y * invMA);
            AtomicAddFloatShared(shared_v_z[idA], -P_t.z * invMA);
            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -P_t)));
            AtomicAddFloatShared(shared_w_x[idA], dwA.x);
            AtomicAddFloatShared(shared_w_y[idA], dwA.y);
            AtomicAddFloatShared(shared_w_z[idA], dwA.z);
        }
        if (!is_partB && invMB > 0.0 && idB < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idB], P_t.x * invMB);
            AtomicAddFloatShared(shared_v_y[idB], P_t.y * invMB);
            AtomicAddFloatShared(shared_v_z[idB], P_t.z * invMB);
            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, P_t)));
            AtomicAddFloatShared(shared_w_x[idB], dwB.x);
            AtomicAddFloatShared(shared_w_y[idB], dwB.y);
            AtomicAddFloatShared(shared_w_z[idB], dwB.z);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    outputs->impulses[contact_idx] = accumulated_normal[local_id] * n + accumulated_t1[local_id] * t1 + accumulated_t2[local_id] * t2;
}
