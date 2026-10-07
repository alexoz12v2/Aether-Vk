// @assets/dust.vert
//
// Dust v3 renderer: instanced camera-facing quads, drawn indirectly from the tier's LOD instance
// list (dust_lod.comp): instance -> (cluster, child). Every child is a DUST_CHILD_PX dot: what grows
// on screen is split into more children by the LOD, the dots never grow.
// Streaklines: a cluster with a stream predecessor (r - S: the same stream's previous time
// sample, dust_streak_pred) draws the dust emitted between them along the segment, on the part
// the LOD found visible (dust_streak_clip, rebuilt here from the same mvp):
//   x_c = lerp(P, P_pred, t_c) + sigma(t_c) (N e1 + N' e2)   (stream dispersion, perpendicular)
//       + 1/2 * dbeta_c * g_sun * age(t_c)^2 * (-sun)         (size spread within the stratum)
// with t_c = t0 + (t1 - t0) fract(u_id + c phi^-1). On a spinning nucleus consecutive samples of a
// stream sweep the jet around the pole, so the streaks draw spirals and arcs. Without a
// predecessor the cluster is a point spread (dust_child_offset).
// Children are keyed by the cluster's child-pattern id (low bits of the dbeta field, hashed from
// the emission record), never by its ring slot.
// Pixels show the dust optical depth: each child spreads its cross-section (the LOD wrote the
// flux per child) over its drawn area, so the brightness of a dust column is independent of zoom,
// distance and the children count (mirror: `dust::splat_footprint` / `dust::splat_opacity`).
#version 450 core
#extension GL_GOOGLE_include_directive : require
#include "sim/dust_common.glsl"

layout(push_constant, std430) uniform PushConstants {
    DustRenderBuffer render;  // 0: flux per child (dust_lod.comp)
    DustUintBuffer header;    // 8: the tier's LOD header: instance list (cluster | child <<
                              //    DUST_LOD_CLUSTER_BITS) at DUST_LOD_HEADER_LIST, flow clock, flags
    mat4 mvp;                 // 16: particle-system local metres -> clip
    vec4 color;               // 80: rgb stream color, a = exposure (gain / (tau_ref * white point))
    vec4 antiSunG;            // 96: unit anti-sun direction (ps frame), w = solar gravity at comet (m/s^2)
    vec4 params;              // 112: x units per metre, y P00, z P11, w +-2/viewport_height
                              //      (negative: 8-bit target, stochastic rounding)
} pc;                         // 128 bytes

layout(location = 0) out vec3 v_color;   // stream color: the saturation ceiling
layout(location = 1) out vec2 v_uv;
layout(location = 2) out float v_opacity; // peak opacity of this splat (before the gaussian)
// 8-bit target: per-splat stochastic rounding offset in [0, 1); < 0 = float target, no rounding
layout(location = 3) flat out float v_dither;

const vec2 CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);

void cull() {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // outside the clip volume
    v_color = vec3(0.0);
    v_uv = vec2(0.0);
    v_opacity = 0.0;
    v_dither = -1.0;
}

void main() {
    DustUintBuffer list = DustUintBuffer(uvec2(pc.header.v[DUST_LOD_HEADER_LIST], pc.header.v[DUST_LOD_HEADER_LIST + 1u]));
    uint flags = pc.header.v[DUST_LOD_HEADER_FLAGS];
    float flowTHi = uintBitsToFloat(pc.header.v[DUST_LOD_HEADER_T_HI]);
    float flowTLo = uintBitsToFloat(pc.header.v[DUST_LOD_HEADER_T_LO]);
    float lodLambda = uintBitsToFloat(pc.header.v[DUST_LOD_HEADER_LAMBDA]);
    uint entry = list.v[gl_InstanceIndex];
    uint cluster = entry & DUST_LOD_CLUSTER_MASK;
    uint child = entry >> DUST_LOD_CLUSTER_BITS;
    bool dither = pc.params.w < 0.0;
    DustRenderCluster R = pc.render.c[cluster];
    // child pattern from the emission record (not the ring slot: same children after a seek)
    uint id = dust_child_id(R.age_id_dbeta_flux.z);
    float flux = R.age_id_dbeta_flux.w;
    if (!(flux > 0.0)) { cull(); return; }

    float age = R.age_id_dbeta_flux.x;
    float spread = R.pos_size.w;
    vec3 pos_m;
    bool tracer = child == DUST_TRACER_CHILD;
    float dotAge = age;
    float want = 1.0; // the cluster's LOD demand (dust_lod.comp): the drawn radius keeps its footprint
    int pred = tracer ? -1 : dust_streak_pred(pc.render, cluster);
    if (tracer) {
        // a real particle, at its exact position (dust::child_sample)
        pos_m = R.pos_size.xyz;
    } else if (pred >= 0) {
        DustRenderCluster RQ = pc.render.c[pred];
        vec4 Q = RQ.pos_size;
        float aq = RQ.age_id_dbeta_flux.x;
        // the LOD saw it: a disagreement can only come from rounding at the view edge (same
        // footprints, dust_extent; the dots themselves keep the stream spread)
        vec4 S;
        if (!dust_streak_clip(pc.mvp, R.pos_size.xyz, Q.xyz, dust_extent(R, pc.antiSunG.w), dust_extent(RQ, pc.antiSunG.w), pc.params, S)) S = vec4(0.0, 1.0, 0.0, 0.0);
        pos_m = dust_streak_child(R.pos_size.xyz, Q.xyz, spread, Q.w, age, aq, S.x, S.y, id, child,
                                  R.age_id_dbeta_flux.z, pc.antiSunG);
        dotAge = age + (aq - age) * dust_streak_dot_t(S.x, S.y, id, child);
        want = dust_streak_want(S.z, S.w);
    } else {
        pos_m = R.pos_size.xyz + dust_child_offset(id, child, spread, R.age_id_dbeta_flux.z, age, pc.antiSunG);
        float cw = (pc.mvp * vec4(R.pos_size.xyz, 1.0)).w;
        if (cw > 0.0) want = dust_lod_want(dust_extent(R, pc.antiSunG.w) * pc.params.x * pc.params.z / cw / abs(pc.params.w));
    }

    vec4 clip = pc.mvp * vec4(pos_m, 1.0);
    if (!(clip.w > 0.0) || !(pc.params.y > 0.0) || !(pc.params.z > 0.0)) { cull(); return; }
    // The tail spans 1e5..1e7 m, far beyond the tight depth range of the comet's layer. Dust
    // writes no depth, so clamp it into range instead of letting it be clipped: it stays
    // occluded by the comet through the depth test (reverse Z: 0 = far, w = near).
    clip.z = clamp(clip.z, 0.0, clip.w);

    // footprint: DUST_CHILD_PX dots, larger when the LOD drew fewer than the cluster asks for
    // (dust_splat_radius: still covering it, energy exact), and the same radius in metres here
    float units_per_m = pc.params.x;
    float px_to_ndc_y = abs(pc.params.w);
    float px_to_ndc_x = px_to_ndc_y * (pc.params.y / pc.params.z);
    float px_per_unit = pc.params.z / clip.w / px_to_ndc_y;
    float r_px = tracer ? DUST_TRACER_PX : dust_splat_radius(want, dust_lod_children(want, lodLambda));
    float r_draw_m = r_px / px_per_unit / units_per_m;

    // Optical depth: the child's cross-section (m^2) over its drawn area (the frag gaussian
    // integrates to 1 over the unit disc), times the exposure. A tracer is a highlight at a fixed
    // level of the view's white point instead (the accumulation is relative to it): visible at any
    // zoom, sparse enough not to matter for the white point. The flow modulates the dust around its
    // mean with marks on synchrones: Lagrangian timelines riding the particles, K x faster on the
    // flow clock T (held on pause, so paused frames are identical; dust::flow_factor).
    float intensity = tracer ? DUST_TRACER_LEVEL : pc.color.a * flux / max(r_draw_m * r_draw_m, 1e-30);
    if (!tracer && (flags & DUST_VIEW_FLOW) != 0u) {
        intensity *= dust_flow_factor(dotAge, flowTHi, flowTLo);
    }

    vec2 corner = CORNERS[gl_VertexIndex % 6];
    clip.xy += corner * r_px * vec2(px_to_ndc_x, px_to_ndc_y) * clip.w;
    gl_Position = clip;
    v_color = pc.color.rgb;
    v_opacity = intensity;
    v_uv = corner;
    // decorrelates the splats covering a pixel: all rounding the same way would bias the sum
    v_dither = dither ? dust_u01(dust_pcg(uint(gl_InstanceIndex) ^ 0xA511E9B3u)) : -1.0;
}
