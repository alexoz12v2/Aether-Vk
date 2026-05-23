#ifndef IMEX_MATH_GLSL
#define IMEX_MATH_GLSL

#include "../bvh_utils.glsl"

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
mat3 quat_to_mat3(vec4 q) {
    float xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z, xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z, wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
    return mat3(1.0-2.0*(yy+zz), 2.0*(xy+wz), 2.0*(xz-wy), 2.0*(xy-wz), 1.0-2.0*(xx+zz), 2.0*(yz+wx), 2.0*(xz+wy), 2.0*(yz-wx), 1.0-2.0*(xx+yy));
}

#endif // IMEX_MATH_GLSL
