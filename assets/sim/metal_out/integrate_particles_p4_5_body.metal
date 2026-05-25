// @assets/sim/integrate_particles_p4_5.comp
//
// Particle Velocity-Verlet Corrector — Phase 4 & 5
// ─────────────────────────────────────────────────
// Invariant entering this pass:
//   • AOSOA slots 3/4/5 hold v_{n+½}  (stored by integrate_particles_p1_p2)
//   • AOSOA slots 7/8/9 hold F(x_{n+1}) (written by force generators after p3)
//
//   v_{n+1} = v_{n+½} + (dt/2) · M⁻¹ · F(x_{n+1})    [VV corrector]
//
// The force buffer is intentionally NOT cleared — F(x_{n+1}) persists as
// F(x_n) for the NEXT frame's integrate_particles_p1_p2 pass.
//
// Thread 0 additionally advances the emulated 64-bit engine clock:
//   global_time_us += dt_us    (uvec2 carry-propagating addition from imex_math.glsl)
//
// Target: SPIR-V 1.4 · Vulkan 1.1 · flexible across all hardware subgroup sizes.

struct PushConstants_integrate_particles_p4_5 {
    ParticleData particles;
    ClockBuffer  clock;
    float        dt;
    uint         total_particles;
    uint         dt_us_lo;
    uint         dt_us_hi;
    uint         current_time_lo;
    uint         current_time_hi;
};

[[kernel]]
void integrate_particles_p4_5(constant PushConstants_integrate_particles_p4_5& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint gid = thread_position_in_grid.x;

    // ── Thread 0: advance the 64-bit engine clock exactly once per frame ─────
    // This must happen regardless of particle count so the clock always ticks.
    if (gid == 0u) {
        uint2 t_n  = uint2(pc.current_time_lo, pc.current_time_hi);
        uint2 dt_u = uint2(pc.dt_us_lo,        pc.dt_us_hi);
        uint2 res;
        res.x = t_n.x + dt_u.x;
        uint carry = (res.x < t_n.x) ? 1u : 0u;
        res.y = t_n.y + dt_u.y + carry;
        pc.clock.global_time_us = res;
    }

    if (gid >= pc.total_particles) return;

    uint block = gid / SUBGROUP_SIZE;
    uint lane  = gid % SUBGROUP_SIZE;
    uint base  = block * (10u * SUBGROUP_SIZE) + lane;

    // ── Skip inactive / massless particles ────────────────────────────────
    float mass = P_READ(pc.particles, base + 6u * SUBGROUP_SIZE);
    if (mass <= 0.0) return;

    float inv_m   = 1.0 / mass;
    float half_dt = 0.5 * pc.dt;

    // ── Load v_{n+½} (written by p1_p2) ──────────────────────────────────
    float3 v_half = float3(
        P_READ(pc.particles, base + 3u * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 4u * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 5u * SUBGROUP_SIZE)
    );

    // ── Load F(x_{n+1}) (written by force generators after p3) ───────────
    float3 f_next = float3(
        P_READ(pc.particles, base + 7u * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 8u * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 9u * SUBGROUP_SIZE)
    );

    // ── VV Corrector ─────────────────────────────────────────────────────
    float3 v_next = v_half + f_next * inv_m * half_dt;

    // Write v_{n+1} back — force buffer stays intact for next frame
    P_WRITE(pc.particles, base + 3u * SUBGROUP_SIZE, v_next.x);
    P_WRITE(pc.particles, base + 4u * SUBGROUP_SIZE, v_next.y);
    P_WRITE(pc.particles, base + 5u * SUBGROUP_SIZE, v_next.z);
}
