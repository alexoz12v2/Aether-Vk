// @assets/dust.vert
//
// Dust v3 renderer: instanced camera-facing quads. Instance = cluster * children + child.
// Each cluster (super-particle) is amplified into `children` sub-splats at render time (8..64,
// more while the ring is sparse, chosen by the host within a fixed instance budget) with a
// linearized perturbation relative to the cluster's exact Keplerian position:
//   dx_k = sigma_v * age * N3(0,1)              (ejection velocity dispersion)
//        + 1/2 * dbeta_k * g_sun * age^2 * (-sun) (size spread within the cluster's beta stratum)
// Total light per cluster is independent of the children count, the super-particle weight and the
// sim speed; the per-pixel falloff is a display stretch (see `intensity`).
#version 450 core
#extension GL_GOOGLE_include_directive : require
#include "sim/dust_common.glsl"

layout(push_constant, std430) uniform PushConstants {
    DustRenderBuffer render;  // 0
    uint children;            // 8: render-time children per cluster (DUST_CHILDREN..DUST_MAX_CHILDREN)
    uint liveCount;           // 12
    mat4 mvp;                 // 16: particle-system local metres -> clip
    vec4 color;               // 80: rgb stream color, a = flux scale (gain / mean child flux)
    vec4 antiSunG;            // 96: unit anti-sun direction (ps frame), w = solar gravity at comet (m/s^2)
    vec4 params;              // 112: x units per metre, y P00, z P11, w 2/viewport_height
} pc;                         // 128 bytes

layout(location = 0) out vec3 v_color;   // stream color: the saturation ceiling
layout(location = 1) out vec2 v_uv;
layout(location = 2) out float v_opacity; // peak opacity of this splat (before the gaussian)

const float MIN_PX = 1.5;
const float MAX_PX = 48.0;
// child footprint radius as a fraction of the cluster spread (the 8 children already scatter over
// the spread; a footprint as large as the whole spread smears each cluster into a faint disc)
const float CHILD_RADIUS_FRAC = 0.5;
// stride of the child hash: keeps children decorrelated for any `children` <= this
const uint DUST_MAX_CHILDREN = 64u;

const vec2 CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);

void cull() {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // outside the clip volume
    v_color = vec3(0.0);
    v_uv = vec2(0.0);
    v_opacity = 0.0;
}

void main() {
    uint inst = uint(gl_InstanceIndex);
    uint k = clamp(pc.children, 1u, DUST_MAX_CHILDREN);
    uint cluster = inst / k;
    uint child = inst - cluster * k;
    if (cluster >= pc.liveCount) { cull(); return; }
    // render buffer is compact (index = live-range offset); y carries the stable ring slot
    DustRenderCluster R = pc.render.c[cluster];
    uint slot = floatBitsToUint(R.age_id_dbeta_flux.y);
    float flux = R.age_id_dbeta_flux.w;
    if (!(flux > 0.0)) { cull(); return; }

    float age = R.age_id_dbeta_flux.x;
    float spread = R.pos_size.w;

    // deterministic child perturbation
    uint h0 = dust_pcg(slot * DUST_MAX_CHILDREN + child + 0x9E3779B9u);
    uint h1 = dust_pcg(h0);
    uint h2 = dust_pcg(h1);
    uint h3 = dust_pcg(h2);
    uint h4 = dust_pcg(h3);
    vec3 n3 = vec3(
        dust_gauss(dust_u01(h0), dust_u01(h1)),
        dust_gauss(dust_u01(h1 ^ 0x68E31DA4u), dust_u01(h2)),
        dust_gauss(dust_u01(h3), dust_u01(h4))
    );
    float dbeta = R.age_id_dbeta_flux.z * (2.0 * dust_u01(dust_pcg(h4)) - 1.0);
    vec3 offset = spread * n3 + (0.5 * dbeta * pc.antiSunG.w * age * age) * pc.antiSunG.xyz;
    vec3 pos_m = R.pos_size.xyz + offset;

    vec4 clip = pc.mvp * vec4(pos_m, 1.0);
    if (!(clip.w > 0.0) || !(pc.params.y > 0.0) || !(pc.params.z > 0.0)) { cull(); return; }
    // The tail spans 1e5..1e7 m, far beyond the tight depth range of the comet's layer. Dust
    // writes no depth, so clamp it into range instead of letting it be clipped: it stays
    // occluded by the comet through the depth test (reverse Z: 0 = far, w = near).
    clip.z = clamp(clip.z, 0.0, clip.w);

    // child footprint: physical radius, clamped to [MIN_PX, MAX_PX] pixels
    float units_per_m = pc.params.x;
    float px_to_ndc_y = pc.params.w;
    float px_to_ndc_x = px_to_ndc_y * (pc.params.y / pc.params.z);
    float r_units = max(spread * CHILD_RADIUS_FRAC, 1.0) * units_per_m;
    float r_px = r_units * pc.params.z / clip.w / px_to_ndc_y;
    r_px = clamp(r_px, MIN_PX, MAX_PX);

    // Display stretch: the micro layer is RGBA8, so an energy-conserving 1/r_px^2 falloff drives
    // spread-out splats below 1/510 per pixel, where additive blending accumulates nothing.
    // 1/r_px keeps large (old) splats visible while compact (young) ones stay brightest.
    // color.a = gain / mean child flux, so the mean child peaks at `gain` at MIN_PX.
    float intensity = pc.color.a * (flux / float(k)) * (MIN_PX / r_px);

    vec2 corner = CORNERS[gl_VertexIndex % 6];
    clip.xy += corner * r_px * vec2(px_to_ndc_x, px_to_ndc_y) * clip.w;
    gl_Position = clip;
    v_color = pc.color.rgb;
    v_opacity = intensity;
    v_uv = corner;
}