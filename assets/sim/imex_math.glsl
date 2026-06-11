#ifndef IMEX_MATH_GLSL
#define IMEX_MATH_GLSL

#include "../bvh_utils.glsl"

// Number of float fields per particle in the AOSOA buffer.
// Slots: 0-2=pos, 3-5=vel, 6=mass, 7-9=force, 10=beta
#define PARTICLE_FIELDS 11u

// ── Particle coordinate-system contract ─────────────────────────────────────
// ALL particle systems are children of a micro-frame entity (frame_type == 1).
// Within a micro-frame:
//   position  — km  (local to the micro frame, origin = frame center)
//   velocity  — km/s
//   force     — km/s²  (per unit mass acceleration; apply_emitters writes this)
//   dt        — seconds  (converted from µs by the CPU before upload)
//
// Macro-frame (frame_type == 0) particle systems are NOT currently supported.
// apply_emitters_to_particles.comp branches on frame.frame_type to convert the
// macro-frame emitter position (AU) into km before computing r_local, so the
// resulting force is always in km/s² regardless of the emitter's frame.
//
// VV integration consistency check:
//   p1_p2:  pos_next  = pos_n  + v_half * dt_s     [km + km/s·s   = km  ✓]
//           v_half    = v_n    + f_n * inv_m * dt/2 [km/s + km/s²·s = km/s ✓]
//   p4_5:   v_next    = v_half + f_next * inv_m * dt/2 [km/s + km/s²·s = km/s ✓]
//
// If a macro-frame particle system is ever introduced, the position step must be
// divided by AU_TO_KM (= 149597870.7) before writing back.
#define KM_TO_AU (1.0 / 149597870.7)

// Compute the AOSOA base index for a given global particle index.
#define P_BASE(gid) (((gid) / SUBGROUP_SIZE) * (PARTICLE_FIELDS * SUBGROUP_SIZE) + ((gid) % SUBGROUP_SIZE))

uvec2 add64(uvec2 a, uvec2 b) { uvec2 res; res.x = a.x + b.x; uint carry = (res.x < a.x) ? 1u : 0u; res.y = a.y + b.y + carry; return res; }
uvec2 sub64(uvec2 a, uvec2 b) { uvec2 res; res.x = a.x - b.x; uint borrow = (a.x < b.x) ? 1u : 0u; res.y = a.y - b.y - borrow; return res; }
bool greaterThan64(uvec2 a, uvec2 b) { if (a.y != b.y) return a.y > b.y; return a.x > b.x; }
bool lessThan64(uvec2 a, uvec2 b) { if (a.y != b.y) return a.y < b.y; return a.x < b.x; }
uvec2 multiply32(uint a, uint b) { uvec2 res; umulExtended(a, b, res.y, res.x); return res; }
float dt_to_seconds(uvec2 dt_micros) { return float(dt_micros.x) * 1e-6 + float(dt_micros.y) * 4294.967296; }

// --- Compare-And-Swap (CAS) Float Atomics Toolkit ---
#define ATOMIC_ADD_FLOAT(dest, val) \
    do { \
        uint old_val = (dest); \
        uint assumed_val; \
        do { \
            assumed_val = old_val; \
            uint new_val = floatBitsToUint(uintBitsToFloat(assumed_val) + (val)); \
            old_val = atomicCompSwap((dest), assumed_val, new_val); \
        } while (assumed_val != old_val); \
    } while(false)

#define ADD_LINEAR_FORCE(w_ref, f) \
    do { \
        ATOMIC_ADD_FLOAT((w_ref).force_x, (f).x); \
        ATOMIC_ADD_FLOAT((w_ref).force_y, (f).y); \
        ATOMIC_ADD_FLOAT((w_ref).force_z, (f).z); \
    } while(false)

#define ADD_TORQUE(w_ref, t) \
    do { \
        ATOMIC_ADD_FLOAT((w_ref).torque_x, (t).x); \
        ATOMIC_ADD_FLOAT((w_ref).torque_y, (t).y); \
        ATOMIC_ADD_FLOAT((w_ref).torque_z, (t).z); \
    } while(false)

#define CLEAR_WRENCH(w_ref) \
    do { \
        (w_ref).force_x = 0u; (w_ref).force_y = 0u; (w_ref).force_z = 0u; \
        (w_ref).torque_x = 0u; (w_ref).torque_y = 0u; (w_ref).torque_z = 0u; \
    } while(false)

vec4 quat_conj(vec4 q) { return vec4(-q.xyz, q.w); }
vec4 quat_mult(vec4 q1, vec4 q2) {
    return vec4(q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y, q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
                q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w, q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z);
}
vec3 quat_rotate(vec4 q, vec3 v) { vec3 t = 2.0 * cross(q.xyz, v); return v + q.w * t + cross(q.xyz, t); }
vec3 quat_rotate_inv(vec4 q, vec3 v) { return quat_rotate(quat_conj(q), v); }

// +x=right,-y=forward,+z=up
// Helper: Converts a quaternion to a 3x3 rotation matrix
mat3 quat_to_mat3(vec4 q) {
    float qxx = q.x * q.x; float qyy = q.y * q.y; float qzz = q.z * q.z;
    float qxz = q.x * q.z; float qxy = q.x * q.y; float qyz = q.y * q.z;
    float qwx = q.w * q.x; float qwy = q.w * q.y; float qwz = q.w * q.z;

    return mat3(
        1.0 - 2.0 * (qyy + qzz),       2.0 * (qxy + qwz),       2.0 * (qxz - qwy),
              2.0 * (qxy - qwz), 1.0 - 2.0 * (qxx + qzz),       2.0 * (qyz + qwx),
              2.0 * (qxz + qwy),       2.0 * (qyz - qwx), 1.0 - 2.0 * (qxx + qyy)
    );
}

#endif // IMEX_MATH_GLSL