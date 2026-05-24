#include <metal_stdlib>
using namespace metal;

inline void atomic_add_float(device atomic_uint* ptr, float val) {
    uint old_val = atomic_load_explicit(ptr, memory_order_relaxed);
    uint assumed_val;
    do {
        assumed_val = old_val;
        uint new_val = as_type<uint>(as_type<float>(assumed_val) + val);
    } while (!atomic_compare_exchange_weak_explicit(ptr, &old_val, new_val, memory_order_relaxed, memory_order_relaxed));
}

float4 quat_conj(float4 q) {
    return float4(-q.xyz, q.w);
}

float3 quat_rotate(float4 q, float3 v) {
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float3 quat_rotate_inv(float4 q, float3 v) {
    return quat_rotate(quat_conj(q), v);
}

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

struct PushConstants {
    device atomic_uint* particles;
    device PackedCollisions* collisions;
    device float3* impulses; // float3 in MSL has 16-byte size/alignment matching std430 vec3
    device atomic_uint* rigid_bodies;
};

constant uint SUBGROUP_SIZE = 32;

kernel void apply_impulses(
    constant PushConstants& pc [[buffer(0)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= pc.collisions->count) return;

    PackedPair pair = pc.collisions->pairs[global_id];
    float3 impulse = pc.impulses[global_id];
    if (length(impulse) < 1e-6) return;

    uint pA_id = pair.a.primitive_index;
    uint pB_id = pair.b.primitive_index;

    bool is_rb_a = (pair.a.entity_id != 0xFFFFFFFFu);
    bool is_rb_b = (pair.b.entity_id != 0xFFFFFFFFu);

    if (is_rb_a) {
        uint base = pA_id * 28;
        float mass = as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 3], memory_order_relaxed));
        float invMA = mass > 0.0 ? 1.0 / mass : 0.0;

        float3 invIA = float3(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 16], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 17], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 18], memory_order_relaxed))
        );
        float4 qA = float4(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 4], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 5], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 6], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 7], memory_order_relaxed))
        );
        float3 posA = float3(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 0], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 1], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 2], memory_order_relaxed))
        );

        float3 rA = pair.contact_point.xyz - posA;

        if (invMA > 0.0) {
            float3 dvA = -impulse * invMA;
            atomic_add_float(&pc.rigid_bodies[base + 8], dvA.x);
            atomic_add_float(&pc.rigid_bodies[base + 9], dvA.y);
            atomic_add_float(&pc.rigid_bodies[base + 10], dvA.z);

            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -impulse)));
            atomic_add_float(&pc.rigid_bodies[base + 12], dwA.x);
            atomic_add_float(&pc.rigid_bodies[base + 13], dwA.y);
            atomic_add_float(&pc.rigid_bodies[base + 14], dwA.z);
        }
    } else {
        uint base = (pA_id / SUBGROUP_SIZE) * (10u * SUBGROUP_SIZE) + (pA_id % SUBGROUP_SIZE);
        float mass = as_type<float>(atomic_load_explicit(&pc.particles[base + 6u * SUBGROUP_SIZE], memory_order_relaxed));
        float invMA = mass > 0.0 ? 1.0 / mass : 0.0;
        if (invMA > 0.0) {
            float3 dvA = -impulse * invMA;
            atomic_add_float(&pc.particles[base + 3u * SUBGROUP_SIZE], dvA.x);
            atomic_add_float(&pc.particles[base + 4u * SUBGROUP_SIZE], dvA.y);
            atomic_add_float(&pc.particles[base + 5u * SUBGROUP_SIZE], dvA.z);
        }
    }

    if (is_rb_b) {
        uint base = pB_id * 28;
        float mass = as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 3], memory_order_relaxed));
        float invMB = mass > 0.0 ? 1.0 / mass : 0.0;

        float3 invIB = float3(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 16], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 17], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 18], memory_order_relaxed))
        );
        float4 qB = float4(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 4], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 5], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 6], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 7], memory_order_relaxed))
        );
        float3 posB = float3(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 0], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 1], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 2], memory_order_relaxed))
        );

        float3 rB = pair.contact_point.xyz - posB;

        if (invMB > 0.0) {
            float3 dvB = impulse * invMB;
            atomic_add_float(&pc.rigid_bodies[base + 8], dvB.x);
            atomic_add_float(&pc.rigid_bodies[base + 9], dvB.y);
            atomic_add_float(&pc.rigid_bodies[base + 10], dvB.z);

            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, impulse)));
            atomic_add_float(&pc.rigid_bodies[base + 12], dwB.x);
            atomic_add_float(&pc.rigid_bodies[base + 13], dwB.y);
            atomic_add_float(&pc.rigid_bodies[base + 14], dwB.z);
        }
    } else {
        uint base = (pB_id / SUBGROUP_SIZE) * (10u * SUBGROUP_SIZE) + (pB_id % SUBGROUP_SIZE);
        float mass = as_type<float>(atomic_load_explicit(&pc.particles[base + 6u * SUBGROUP_SIZE], memory_order_relaxed));
        float invMB = mass > 0.0 ? 1.0 / mass : 0.0;
        if (invMB > 0.0) {
            float3 dvB = impulse * invMB;
            atomic_add_float(&pc.particles[base + 3u * SUBGROUP_SIZE], dvB.x);
            atomic_add_float(&pc.particles[base + 4u * SUBGROUP_SIZE], dvB.y);
            atomic_add_float(&pc.particles[base + 5u * SUBGROUP_SIZE], dvB.z);
        }
    }
}
