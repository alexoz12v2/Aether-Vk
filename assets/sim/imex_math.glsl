#ifndef IMEX_MATH_GLSL
#define IMEX_MATH_GLSL

#extension GL_EXT_shader_atomic_float : require
#include "physics_types.glsl"

// --- 1. 64-Bit Emulation Toolkit (uvec2) ---
uvec2 add64(uvec2 a, uvec2 b) {
    uvec2 res;
    res.x = a.x + b.x;
    uint carry = (res.x < a.x) ? 1u : 0u;
    res.y = a.y + b.y + carry;
    return res;
}

uvec2 sub64(uvec2 a, uvec2 b) {
    uvec2 res;
    res.x = a.x - b.x;
    uint borrow = (a.x < b.x) ? 1u : 0u;
    res.y = a.y - b.y - borrow;
    return res;
}

bool greaterThan64(uvec2 a, uvec2 b) {
    if (a.y != b.y) return a.y > b.y;
    return a.x > b.x;
}

bool lessThan64(uvec2 a, uvec2 b) {
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

uvec2 multiply32(uint a, uint b) {
    uvec2 res;
    umulExtended(a, b, res.y, res.x);
    return res;
}

// Convert microseconds to physical seconds for the integration dt
float dt_to_seconds(uvec2 dt_micros) {
    return float(dt_micros.x) * 1e-6 + float(dt_micros.y) * 4294.967296; // 4294967296 = 2^32
}

// --- 2. Force & Entity Data Layouts ---

// Lightweight force accumulator for point-mass particles (no torque needed).
// Lives in a separate buffer, one entry per particle, indexed by particle_id.
struct ParticleForce {
    float f_x; float f_y; float f_z;
    float pad;           // keep 16-byte alignment
};

#define ADD_PARTICLE_FORCE(pf_ref, f) \
    atomicAdd((pf_ref).f_x, (f).x); \
    atomicAdd((pf_ref).f_y, (f).y); \
    atomicAdd((pf_ref).f_z, (f).z)

void clear_particle_force(inout ParticleForce pf) {
    pf.f_x = 0.0; pf.f_y = 0.0; pf.f_z = 0.0;
}

// Full SE(3) wrench for rigid bodies (force + torque at CoM).
// NOTE: keep as 6 floats — binary-compatible with existing barnes_hut.comp SPV.
struct Wrench {
    float force_x;  float force_y;  float force_z;
    float torque_x; float torque_y; float torque_z;
};

// GLSL atomic functions require direct buffer l-values, so we use macros
// instead of `inout` function parameters to avoid passing by value-return.
#define ADD_LINEAR_FORCE(w_ref, f) \
    atomicAdd((w_ref).force_x, (f).x); \
    atomicAdd((w_ref).force_y, (f).y); \
    atomicAdd((w_ref).force_z, (f).z)

#define ADD_TORQUE(w_ref, t) \
    atomicAdd((w_ref).torque_x, (t).x); \
    atomicAdd((w_ref).torque_y, (t).y); \
    atomicAdd((w_ref).torque_z, (t).z)

#define CLEAR_WRENCH(w) \
    (w).force_x = 0.0; (w).force_y = 0.0; (w).force_z = 0.0; \
    (w).torque_x = 0.0; (w).torque_y = 0.0; (w).torque_z = 0.0

// --- 3. Quaternion & Rotation Utilities ---
vec4 quat_conj(vec4 q) { return vec4(-q.xyz, q.w); }

vec4 quat_mult(vec4 q1, vec4 q2) {
    return vec4(
        q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
        q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
        q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
        q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    );
}

// Rotates a vector from local to world space
vec3 quat_rotate(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Inversely rotates a vector from world to local space
vec3 quat_rotate_inv(vec4 q, vec3 v) {
    return quat_rotate(quat_conj(q), v);
}

mat3 quat_to_mat3(vec4 q) {
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;

    return mat3(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy),
        2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx),
        2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy)
    );
}

#endif // IMEX_MATH_GLSL
