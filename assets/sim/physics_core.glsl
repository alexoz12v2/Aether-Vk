#ifndef PHYSICS_CORE_GLSL
#define PHYSICS_CORE_GLSL

#include "gjk_cta_utils.glsl"

void generate_tangents(vec3 normal, out vec3 t1, out vec3 t2) {
    if (abs(normal.x) >= 0.57735) {
        t1 = normalize(vec3(normal.y, -normal.x, 0.0));
    } else {
        t1 = normalize(vec3(0.0, normal.z, -normal.y));
    }
    t2 = cross(normal, t1);
}

float compute_eff_mass_micro(vec3 dir, vec3 r, float invM, vec3 invI, vec4 q) {
    vec3 I_cross = quat_rotate(q, invI * quat_rotate_inv(q, cross(r, dir)));
    return dot(I_cross, cross(r, dir)) + invM;
}

float compute_eff_mass_macro(vec3 dir_lca, vec3 r_lca, float invM_kg, vec3 invI_kgkm, vec4 q_world, mat3 rot_lca_to_world) {
    vec3 r_world = rot_lca_to_world * r_lca;
    vec3 dir_world = rot_lca_to_world * dir_lca;
    vec3 I_cross_world = quat_rotate(q_world, invI_kgkm * quat_rotate_inv(q_world, cross(r_world, dir_world)));
    vec3 I_cross_lca = transpose(rot_lca_to_world) * I_cross_world;
    return dot(I_cross_lca, cross(r_lca, dir_lca)) + invM_kg;
}

float compute_effective_mass_unified(
    uint space_type,
    vec3 dir, vec3 rA, vec3 rB,
    float invMA, float invMB,
    vec3 invIA, vec3 invIB,
    vec4 qA, vec4 qB,
    mat3 rot_lca_to_world
) {
    if (space_type == 1) { // Cross
        return 1.0 / max(compute_eff_mass_macro(dir, rA, invMA, invIA, qA, rot_lca_to_world) + compute_eff_mass_micro(dir, rB, invMB, invIB, qB), 1e-6);
    } else { // Standard
        vec3 I_crossA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, dir)));
        vec3 I_crossB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, dir)));
        return 1.0 / max(invMA + invMB + dot(I_crossA, cross(rA, dir)) + dot(I_crossB, cross(rB, dir)), 1e-6);
    }
}

vec3 picard_gyroscopic_stabilization(vec3 w_n_local, vec3 I_inv, vec3 I_fwd, vec3 t_local, float half_dt, uint n_iterations) {
    vec3 w_mid_local = w_n_local;
    for (uint iter = 0u; iter < n_iterations; ++iter) {
        vec3 gyro = cross(w_mid_local, I_fwd * w_mid_local);
        vec3 a_ang = I_inv * (t_local - gyro);
        w_mid_local = w_n_local + half_dt * a_ang;
    }
    return w_mid_local;
}

#endif // PHYSICS_CORE_GLSL
