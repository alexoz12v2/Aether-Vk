// @assets/sim/dust_cadence.glsl
//
// Temporal reuse by cadence classes (plan "temporal reuse", step 3), the GLSL mirror of
// `scene/dust/cadence.rs`: its constants, the owner byte, the GPU buffer layouts (the per-system
// cadence buffer, the per-frame params, the per-view state; host: `device/dust_cadence.rs`), the
// rate rule (`cadence_rate`, with `DUST_CADENCE_NEAR_SIGMAS` = 0) and the speck-shell rule
// (`shell_capable`). Same integers and branch order as the CPU mirror; scalars only (no local or
// const arrays: the hot-GLSL memory rule). Include after dust_common.glsl.
#ifndef DUST_CADENCE_GLSL
#define DUST_CADENCE_GLSL

// ── constants (dust::cadence::DUST_CADENCE_*) ──
const uint  DUST_CAD_MAX_CLASSES  = 3u;
const float DUST_CAD_SHELL_MARGIN = 2.0;
const float DUST_CAD_NEAR_SIGMAS  = 0.0;
const float DUST_CAD_AGE_MIN_S    = 1.0;
const float DUST_CAD_SIGMA_MIN_M  = 1.0e-3;
// DUST_CADENCE_SIGMA_MIN_M² as the CPU's f32 product computes it (0x358637BE)
const uint  DUST_CAD_SIGMA_MIN2_BITS = 0x358637BEu;
// 2π/3 as the CPU's f32 `2.0 * PI / 3.0` (0x40060A92)
const uint  DUST_CAD_TWO_PI_3_BITS = 0x40060A92u;
const uint  DUST_CAD_INF_BITS = 0x7F800000u;

// ── the owner byte (dust::cadence::OWN_*) ──
const uint DUST_OWN_CLASS_MASK = 3u;
const uint DUST_OWN_SHOWN      = 4u;
const uint DUST_OWN_BUILD      = 8u;
const uint DUST_OWN_PENDING    = 16u;
const uint DUST_OWN_IN_IMAGE   = 32u;
// cadence::drop_empty: a byte with no state bit left carries no class
uint dust_own_drop_empty(uint b) { return (b & ~DUST_OWN_CLASS_MASK) == 0u ? 0u : b; }
// cadence::owner_byte
uint dust_own_byte(uint k, uint states) { return (k + 1u) | (states & ~DUST_OWN_CLASS_MASK); }
// the state bit of list state s (0 shown, 1 build, 2 pending)
uint dust_own_state_bit(uint s) { return s == 0u ? DUST_OWN_SHOWN : (s == 1u ? DUST_OWN_BUILD : DUST_OWN_PENDING); }

// ── the per-frame params (words; dust_cadence::CADP_*) ──
const uint DUST_CADP_DT        = 0u;   // f32 simulated s per frame
const uint DUST_CADP_DEYE      = 1u;   // 3 x f32: the eye's heliocentric step (m)
const uint DUST_CADP_AW        = 4u;   // 3 x f32: the window term per class (m)
const uint DUST_CADP_EPS       = 7u;   // f32 ε (px)
const uint DUST_CADP_FRAME_PX  = 8u;   // f32 max(width, height)
const uint DUST_CADP_W         = 9u;   // f32 width
const uint DUST_CADP_H         = 10u;  // f32 height
const uint DUST_CADP_NC        = 11u;  // classes
const uint DUST_CADP_NP        = 12u;  // phases (1 | 2)
const uint DUST_CADP_N         = 13u;  // 3 x the class lengths N_k
const uint DUST_CADP_NSWAP     = 16u;  // the (class, phase) pairs whose cycle starts this frame
const uint DUST_CADP_SWAP0     = 17u;  // 6 x k | p << 8, in the CPU's order (k, then p)
const uint DUST_CADP_FLAGS     = 23u;  // DUST_CADF_*
const uint DUST_CADP_DEBUG     = 24u;  // 2 x the debug records' address (lo, hi; test hook)
const uint DUST_CADP_DMAX      = 26u;  // f32 the speck shell radius of the frame (0: none)
const uint DUST_CADP_SHAPE     = 27u;  // f32 the fallback's shape bound θ (0: off; cadence::RateInputs::shape_frac)
const uint DUST_CADP_STATE     = 28u;  // 2 x the state buffer's address (lo, hi)
const uint DUST_CADP_NT        = 30u;  // the view's tiers this frame (every system)
const uint DUST_CADP_ADMIT     = 31u;  // bit k: class k fits the motion budget (cadence::class_admitted)
const uint DUST_CADP_DD        = 32u;  // 12 x f32 |Δe − Δe_c| (m) per (k * 2 + p) * 2 + list (0 build, 1 pending)
const uint DUST_CADP_BOUND     = 44u;  // 3 x f32 the abort bound per class (px)
const uint DUST_CADP_FORCE     = 48u;  // 12 x the host's abort reason per (k * 2 + p) * 2 + list when the rate does not fire (0 none, ABORT_BUDGET, ABORT_PREDICTION)
const uint DUST_CADP_COVER     = 60u;  // bit k * 2 + p: the shown image no longer covers the viewport (cadence::image_covers)
const uint DUST_CADP_FOOT      = 61u;  // 1: the footprint rule (D3) holds (no margin)
const uint DUST_CADP_ANCH      = 62u;  // 2 x the anchor tables' address (lo, hi): per tier DUST_CAD_ANCH_TIER_WORDS words
const uint DUST_CADP_TIER0     = 64u;  // per tier t < 8 (8 words): ring base, capacity, live, first slot, the system ring R
const uint DUST_CADP_TIER_WORDS = 8u;
const uint DUST_CADP_STEPU     = 128u; // per (t * 3 + k) * 2 + p: the chunk's host bound (draw-list entries)
const uint DUST_CADP_CLEAR0    = 192u; // per tier t (8 words): 3 x (first slot, count) owner ranges to clear
const uint DUST_CADP_CLEAR_WORDS = 8u;
const uint DUST_CADP_WORDS     = 256u;
const uint DUST_CAD_MAX_TIERS  = 8u;
const uint DUST_CADF_ADVANCE   = 1u;   // the previous drawn frame was Run: advance the cursors
const uint DUST_CADF_ABORTS    = 2u;   // the abort test runs (Run, no reset)
const uint DUST_CADF_COVERAGE  = 4u;   // the coverage test runs (Run, no reset, a margin)
// the abort reasons (cadence::ABORT_*)
const uint DUST_CAD_ABORT_RATE = 0u;
const uint DUST_CAD_ABORT_BUDGET = 1u;
const uint DUST_CAD_ABORT_PREDICTION = 2u;
const uint DUST_CAD_ABORT_COVERAGE = 3u;
// the anchor tables (cadence's anchor history): per tier, words 0 the entries n, 1 this frame's
// propagate stamp, then n x (stamp, Δx, Δy, Δz f32: the host's anchor_delta_f32 against the
// current anchor), ascending by stamp
const uint DUST_CAD_ANCH_TIER_WORDS = 2304u;
const uint DUST_CAD_ANCH_ENTRY0 = 4u;

// ── the per-view state (words; dust_cadence::CADS_*) ──
// per pair (k * 2 + p): +0 exists shown, +1 exists build, +2 abort flags (bit 0 build, 1 pending),
// +3 / +4 the excess (f32) that aborted the build / pending list, +5 / +6 the build cycle's D3
// rejected / drawn, +7 / +8 the last finished cycle's, +9 / +10 the abort reason of the build /
// pending list, +11 1 = the shown image was dropped by the coverage test this frame
const uint DUST_CADS_PAIR0 = 16u;
const uint DUST_CADS_PAIR_WORDS = 16u;
// per (t * 3 + k) * 2 + p (t < DUST_CAD_MAX_TIERS): +0 / +1 / +2 the shown / build / pending list's length, +3 the build
// cursor (before this frame's advance), +4 its step, +5 / +6 g_max of the build / pending list
// (f32 bits), +8 / +9 the entries drawn / rejected by D3 this frame, +10 the classify's D3
// rejections, +11 its stale slots without an anchor (unshifted)
const uint DUST_CADS_TP0 = 128u;
const uint DUST_CADS_TP_WORDS = 16u;
const uint DUST_CADS_HIST = 1024u;         // 4: the admissible-class histogram of this frame's classifies
const uint DUST_CADS_SHELL_FORCED = 1028u; // classify candidates refused by the shell rule
// the telemetry (dust_cadence::CadenceTelemetry; zeroed by the host each drawn frame, words
// 1024..1048): the fresh slots, the SHOWN slots per pair (k * 2 + p), the light the reprojection
// wrote and the frame's light (counts · 4^ℓ, (lo, hi) fixed point)
const uint DUST_CADS_FRESH = 1029u;
const uint DUST_CADS_SHOWN_PAIR0 = 1032u;   // 6 words
const uint DUST_CADS_SHOWN_LIGHT = 1040u;   // (lo, hi)
const uint DUST_CADS_FRAME_LIGHT = 1042u;   // (lo, hi)
// adds a non-negative float to the (lo, hi) 64-bit counter at word w, with a carry (no 64-bit
// atomics on the baseline: as dust_profile.glsl's add64)
void dust_cad_add64(DustUintBuffer b, uint w, float v) {
    if (!(v > 0.0)) return;
    uint hi = uint(min(floor(v * (1.0 / 4294967296.0)), 4294967295.0));
    uint lo = uint(min(v - float(hi) * 4294967296.0, 4294967295.0));
    uint old = atomicAdd(b.v[w], lo);
    if (old + lo < old) hi += 1u;
    if (hi != 0u) atomicAdd(b.v[w + 1u], hi);
}
const uint DUST_CADS_PROF = 1536u;         // the profiler's lane-cycle words (DUST_PROFILE)
const uint DUST_CADS_WORDS = 2048u;
uint dust_cads_pair(uint k, uint p) { return DUST_CADS_PAIR0 + DUST_CADS_PAIR_WORDS * (k * 2u + p); }
uint dust_cads_tp(uint t, uint k, uint p) { return DUST_CADS_TP0 + DUST_CADS_TP_WORDS * ((t * 3u + k) * 2u + p); }

// ── the per-system cadence buffer (words; dust_cadence::CadenceSysLayout) ──
// [0, R/4) the owner bytes (byte ring base + slot); then the propagate bitmask (R/32 words); the
// lists (3·nc·R words); the draw lists (nc·R); the block counts and bases
uint dust_cad_pmask0(uint R) { return R / 4u; }
uint dust_cad_lists0(uint R) { return R / 4u + R / 32u; }
uint dust_cad_list(uint R, uint nc, uint np, uint rb, uint cap, uint k, uint p, uint s) {
    return dust_cad_lists0(R) + 3u * nc * rb + ((k * np + p) * 3u + s) * (cap / np);
}
uint dust_cad_draw0(uint R, uint nc) { return dust_cad_lists0(R) + 3u * nc * R; }
uint dust_cad_draw(uint R, uint nc, uint np, uint rb, uint cap, uint k, uint p) {
    return dust_cad_draw0(R, nc) + nc * rb + (k * np + p) * (cap / np);
}
uint dust_cad_blocks0(uint R, uint nc) { return dust_cad_draw0(R, nc) + nc * R; }
// the counts of list l = (k * np + p) * 3 + s of tier t at the result, its bases nb words further
uint dust_cad_blocks(uint R, uint nc, uint np, uint rb, uint t, uint nb, uint l) {
    return dust_cad_blocks0(R, nc) + 6u * nc * np * (rb / 1024u + t) + l * 2u * nb;
}

// the owner byte of byte index i (ring base + slot)
uint dust_own_get(DustUintBuffer sys, uint i) { return (sys.v[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }
// stores byte i (the pass's only writer of it; the word's other bytes are other threads': atomics)
void dust_own_set(DustUintBuffer sys, uint i, uint oldB, uint newB) {
    if (oldB == newB) return;
    uint sh = (i & 3u) * 8u;
    atomicAnd(sys.v[i >> 2u], ~(0xFFu << sh));
    if (newB != 0u) atomicOr(sys.v[i >> 2u], newB << sh);
}
// marks bit i (ring base + slot) of the propagate bitmask
void dust_cad_mark(DustUintBuffer sys, uint R, uint i) {
    atomicOr(sys.v[dust_cad_pmask0(R) + (i >> 5u)], 1u << (i & 31u));
}

// ── scalar helpers in the CPU's evaluation order (no contraction) ──
float dust_cad_dot(vec3 a, vec3 b) { precise float s = a.x * b.x + a.y * b.y + a.z * b.z; return s; }
float dust_cad_len(vec3 a) { return sqrt(dust_cad_dot(a, a)); }
bool dust_cad_finite(float x) { return !isinf(x) && !isnan(x); }

// (λ_min, λ_max) of the symmetric 3×3 (xx, xy, xz, yy, yz, zz), closed form (cadence::sym3_eigen_range)
vec2 dust_cad_sym3_range(float a, float b, float d, float e, float f, float g) {
    precise float p1 = b * b + d * d + f * f;
    if (!(p1 > 0.0)) return vec2(min(min(a, e), g), max(max(a, e), g));
    precise float q = (a + e + g) / 3.0;
    precise float p2 = (a - q) * (a - q) + (e - q) * (e - q) + (g - q) * (g - q) + 2.0 * p1;
    float p = sqrt(p2 / 6.0);
    if (!(p > 0.0)) return vec2(q, q);
    float ip = 1.0 / p;
    precise float ba = (a - q) * ip, be = (e - q) * ip, bg = (g - q) * ip, bb = b * ip, bd = d * ip, bf = f * ip;
    precise float det = ba * (be * bg - bf * bf) - bb * (bb * bg - bf * bd) + bd * (bb * bf - be * bd);
    float r = clamp(0.5 * det, -1.0, 1.0);
    float phi = acos(r) / 3.0;
    precise float hi = q + 2.0 * p * cos(phi);
    precise float lo = q + 2.0 * p * cos(phi + uintBitsToFloat(DUST_CAD_TWO_PI_3_BITS));
    return vec2(min(lo, hi), max(hi, lo));
}
vec2 dust_cad_lambda(uint mi) {
    return dust_cad_sym3_range(dust_mom.m[mi].cov_a.x, dust_mom.m[mi].cov_a.y, dust_mom.m[mi].cov_a.z,
                               dust_mom.m[mi].cov_a.w, dust_mom.m[mi].cov_b_age_id.x, dust_mom.m[mi].cov_b_age_id.y);
}

// The stale moments' shift (cadence::shifted_moments, rounds 11-12: a per-generation f32 delta the
// host computes in f64 and casts once, added in f32 to the mean and the 17 edges): 0 at this
// frame's stamp; else the tier's anchor table by binary search; false (and 0) when the stamp left
// the history (the CPU's `unshifted_stale`)
bool dust_cad_shift(DustUintBuffer an, uint t, uint stamp, out vec3 d) {
    d = vec3(0.0);
    uint base = t * DUST_CAD_ANCH_TIER_WORDS;
    if (stamp == an.v[base + 1u]) return true;
    uint lo = 0u, hi = an.v[base];
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        uint e = base + DUST_CAD_ANCH_ENTRY0 + 4u * mid;
        uint sm = an.v[e];
        if (sm == stamp) {
            d = vec3(uintBitsToFloat(an.v[e + 1u]), uintBitsToFloat(an.v[e + 2u]), uintBitsToFloat(an.v[e + 3u]));
            return true;
        }
        if (sm < stamp) lo = mid + 1u; else hi = mid;
    }
    return false;
}

// the polyline's on-screen half-extent with its points shifted by `sh` (dust_polyline_extent_px)
bool dust_cad_extent_px(uint mi, vec3 sh, mat4 mvp, float W, float H, out float ext) {
    vec2 c, p;
    ext = 0.0;
    if (!dust_project_point(dust_mom.m[mi].mean_flux.xyz + sh, mvp, W, H, c)) return false;
    for (uint j = 0u; j < DUST_SIZE_EDGES; ++j) {
        if (!dust_project_point(dust_mom.m[mi].edges[j].xyz + sh, mvp, W, H, p)) return false;
        vec2 d = p - c;
        precise float r2 = d.x * d.x + d.y * d.y;
        ext = max(ext, sqrt(r2));
    }
    return dust_cad_finite(ext);
}
// the time segment on screen, both means shifted (dust_time_segment_px)
bool dust_cad_time_px(uint mi, vec3 sh, uint pi, vec3 shP, mat4 mvp, float W, float H, out float t) {
    vec2 a, c;
    t = 0.0;
    if (!dust_project_point(dust_mom.m[mi].mean_flux.xyz + sh, mvp, W, H, a)) return false;
    if (!dust_project_point(dust_mom.m[pi].mean_flux.xyz + shP, mvp, W, H, c)) return false;
    vec2 d = c - a;
    precise float r2 = d.x * d.x + d.y * d.y;
    t = sqrt(r2);
    return dust_cad_finite(t);
}

// cadence::shell_capable: the cluster's nearest point (mean or an edge) within
// 3σ_max + DUST_CADENCE_SHELL_MARGIN · d_max of the eye (d_max ≤ 0: never)
bool dust_cad_shell_capable(uint mi, vec3 sh, vec3 eye, float dMax) {
    if (!(dMax > 0.0)) return false;
    float d = dust_cad_len(dust_mom.m[mi].mean_flux.xyz + sh - eye);
    for (uint j = 0u; j < DUST_SIZE_EDGES; ++j) d = min(d, dust_cad_len(dust_mom.m[mi].edges[j].xyz + sh - eye));
    float sm = sqrt(max(dust_cad_lambda(mi).y, 0.0));
    precise float lhs = d - 3.0 * sm;
    return lhs < DUST_CAD_SHELL_MARGIN * dMax;
}

// the Jacobian's Frobenius norm (px per m) of mvp at p, ∞ behind the eye (cadence::jacobian_px_per_m)
float dust_cad_jacobian(mat4 mvp, vec3 p, float W, float H) {
    vec3 r0 = vec3(mvp[0][0], mvp[1][0], mvp[2][0]);
    vec3 r1 = vec3(mvp[0][1], mvp[1][1], mvp[2][1]);
    vec3 r3 = vec3(mvp[0][3], mvp[1][3], mvp[2][3]);
    precise float w = dust_cad_dot(r3, p) + mvp[3][3];
    if (!(w > 0.0)) return uintBitsToFloat(DUST_CAD_INF_BITS);
    precise float nx = (dust_cad_dot(r0, p) + mvp[3][0]) / w;
    precise float ny = (dust_cad_dot(r1, p) + mvp[3][1]) / w;
    float kx = 0.5 * W / w;
    float ky = 0.5 * H / w;
    precise vec3 jx = vec3((r0.x - r3.x * nx) * kx, (r0.y - r3.y * nx) * kx, (r0.z - r3.z * nx) * kx);
    precise vec3 jy = vec3((r1.x - r3.x * ny) * ky, (r1.y - r3.y * ny) * ky, (r1.z - r3.z * ny) * ky);
    precise float j2 = dust_cad_dot(jx, jx) + dust_cad_dot(jy, jy);
    return sqrt(j2);
}
// one point of the footprint rule: in front, its 3s·g margin inside [0, W] × [0, H]
bool dust_cad_fp_point(vec3 p, float s, mat4 mvp, float W, float H) {
    vec2 q;
    if (!dust_project_point(p, mvp, W, H, q)) return false;
    precise float r = 3.0 * s * dust_cad_jacobian(mvp, p, W, H);
    return dust_cad_finite(r) && q.x - r >= 0.0 && q.y - r >= 0.0 && q.x + r <= W && q.y + r <= H;
}
// cadence::footprint_inside (D3, no margin only): the mean, the 17 edges and the predecessor's mean
// (σ the larger of the two) wholly inside the frame of mvp; the points shifted by sh / shP
bool dust_cad_footprint(uint mi, vec3 sh, int pred, vec3 shP, mat4 mvp, float W, float H) {
    float sm = sqrt(max(dust_cad_lambda(mi).y, 0.0));
    if (!dust_cad_fp_point(dust_mom.m[mi].mean_flux.xyz + sh, sm, mvp, W, H)) return false;
    for (uint j = 0u; j < DUST_SIZE_EDGES; ++j) {
        if (!dust_cad_fp_point(dust_mom.m[mi].edges[j].xyz + sh, sm, mvp, W, H)) return false;
    }
    if (pred < 0) return true;
    float sp = sqrt(max(dust_cad_lambda(uint(pred)).y, 0.0));
    return dust_cad_fp_point(dust_mom.m[uint(pred)].mean_flux.xyz + shP, max(sm, sp), mvp, W, H);
}

// cadence::cadence_rate (cadence_rate_near at DUST_CADENCE_NEAR_SIGMAS): the class (0 fresh,
// k + 1 the largest k with δ_k · 2 N_k ≤ ε and, with the fallback's shape bound θ > 0,
// 2 N_k · Δt ≤ θ · age), the δ_k (∞ when not finite or no class k), the
// sensitivity g (px/m) and the enclosing flag. `pred` ≥ 0: the predecessor at the same stamp.
uint dust_cad_rate(uint mi, vec3 sh, int pred, vec3 shP, mat4 mvp, vec3 eye, DustUintBuffer prm,
                   out float dl0, out float dl1, out float dl2, out float gOut, out bool enclosingOut) {
    float INF = uintBitsToFloat(DUST_CAD_INF_BITS);
    dl0 = INF; dl1 = INF; dl2 = INF; gOut = INF; enclosingOut = false;
    if (!(dust_mom.m[mi].mean_flux.w > 0.0)) return 0u;
    float W = uintBitsToFloat(prm.v[DUST_CADP_W]), H = uintBitsToFloat(prm.v[DUST_CADP_H]);
    bool persp;
    vec3 fwd = dust_view_forward(mvp, eye, persp);
    // the near point: the edge of the smallest depth along the view (the mean orthographic)
    vec3 nearP = dust_mom.m[mi].mean_flux.xyz + sh;
    if (persp) {
        float best = INF;
        for (uint j = 0u; j < DUST_SIZE_EDGES; ++j) {
            vec3 e = dust_mom.m[mi].edges[j].xyz + sh;
            float dd = dust_cad_dot(e - eye, fwd);
            if (dd < best) { best = dd; nearP = e; }
        }
    }
    vec2 lam = dust_cad_lambda(mi);
    if (persp && DUST_CAD_NEAR_SIGMAS > 0.0) {
        vec3 toEye = eye - nearP;
        float r = dust_cad_len(toEye);
        if (r > 0.0) {
            float l = min(DUST_CAD_NEAR_SIGMAS * sqrt(max(lam.y, 0.0)), 0.9 * r);
            nearP = vec3(nearP.x + toEye.x / r * l, nearP.y + toEye.y / r * l, nearP.z + toEye.z / r * l);
        }
    }
    vec3 v = vec3(dust_mom.m[mi].edges[9].w, dust_mom.m[mi].edges[10].w, dust_mom.m[mi].edges[11].w);
    float dt = uintBitsToFloat(prm.v[DUST_CADP_DT]);
    vec3 dEye = vec3(uintBitsToFloat(prm.v[DUST_CADP_DEYE]), uintBitsToFloat(prm.v[DUST_CADP_DEYE + 1u]),
                     uintBitsToFloat(prm.v[DUST_CADP_DEYE + 2u]));
    precise vec3 s;
    if (persp) s = vec3(v.x * dt - dEye.x, v.y * dt - dEye.y, v.z * dt - dEye.z);
    else s = vec3(v.x * dt, v.y * dt, v.z * dt);
    float sn = dust_cad_len(s);
    vec2 pa, pb;
    float d1 = INF;
    bool oa = dust_project_point(nearP, mvp, W, H, pa);
    bool ob = dust_project_point(vec3(nearP.x + s.x, nearP.y + s.y, nearP.z + s.z), mvp, W, H, pb);
    if (oa && ob) {
        precise float r2 = (pb.x - pa.x) * (pb.x - pa.x) + (pb.y - pa.y) * (pb.y - pa.y);
        d1 = sqrt(r2);
    }
    // the Jacobian's Frobenius norm at the near point (px per m)
    vec3 r0 = vec3(mvp[0][0], mvp[1][0], mvp[2][0]);
    vec3 r1 = vec3(mvp[0][1], mvp[1][1], mvp[2][1]);
    vec3 r3 = vec3(mvp[0][3], mvp[1][3], mvp[2][3]);
    precise float w = dust_cad_dot(r3, nearP) + mvp[3][3];
    float g = INF;
    if (w > 0.0) {
        precise float nx = (dust_cad_dot(r0, nearP) + mvp[3][0]) / w;
        precise float ny = (dust_cad_dot(r1, nearP) + mvp[3][1]) / w;
        float kx = 0.5 * W / w;
        float ky = 0.5 * H / w;
        precise vec3 jx = vec3((r0.x - r3.x * nx) * kx, (r0.y - r3.y * nx) * kx, (r0.z - r3.z * nx) * kx);
        precise vec3 jy = vec3((r1.x - r3.x * ny) * ky, (r1.y - r3.y * ny) * ky, (r1.z - r3.z * ny) * ky);
        precise float j2 = dust_cad_dot(jx, jx) + dust_cad_dot(jy, jy);
        g = sqrt(j2);
    }
    float extP = 0.0, tP = 0.0;
    bool hasE = dust_cad_extent_px(mi, sh, mvp, W, H, extP);
    bool hasT = pred >= 0 && dust_cad_time_px(mi, sh, uint(pred), shP, mvp, W, H, tP);
    float ext = hasE ? (hasT ? max(extP, tP) : extP) : 0.0;
    float age = max(dust_mom.m[mi].cov_b_age_id.z, DUST_CAD_AGE_MIN_S);
    float sigmaMin = max(sqrt(max(lam.x, 0.0)), DUST_CAD_SIGMA_MIN_M);
    vec3 dm = dust_mom.m[mi].mean_flux.xyz + sh - eye;
    bool enclosing = persp && (!hasE || dust_cad_dot(dm, dm) < 9.0 * max(lam.y, 0.0));
    float framePx = uintBitsToFloat(prm.v[DUST_CADP_FRAME_PX]);
    precise float d4 = enclosing ? (sn / sigmaMin + dt / age) * framePx : 0.0;
    if (enclosing) {
        // a cloud around the eye: its sensitivity is frame_px / σ_min; no measurable smallest axis
        // cannot be bounded and is fresh
        if (!(lam.x > uintBitsToFloat(DUST_CAD_SIGMA_MIN2_BITS))) return 0u;
        g = framePx / sigmaMin;
    }
    precise float base;
    if (hasE) {
        precise float d2 = ext * dt / age;
        precise float d3 = abs(dust_mom.m[mi].edges[1].w) * dt * ext;
        base = d1 + d2 + d3 + d4;
    } else {
        base = d4;
    }
    gOut = g;
    enclosingOut = enclosing;
    uint cls = 0u;
    uint nc = prm.v[DUST_CADP_NC];
    float eps = uintBitsToFloat(prm.v[DUST_CADP_EPS]);
    // the by-count fallback's shape bound (θ = 0: off, every class passes it): 2 N_k · Δt ≤ θ · age,
    // the CPU's two f32 products and comparison
    float theta = uintBitsToFloat(prm.v[DUST_CADP_SHAPE]);
    bool shapeOn = theta > 0.0;
    precise float thAge = theta * age;
    if (nc > 0u) {
        precise float dk = base + uintBitsToFloat(prm.v[DUST_CADP_AW]) * g;
        bool fin = dust_cad_finite(dk);
        dl0 = fin ? dk : INF;
        precise float lhs = dk * 2.0 * float(prm.v[DUST_CADP_N]);
        precise float shp = 2.0 * float(prm.v[DUST_CADP_N]) * dt;
        if (fin && lhs <= eps && (!shapeOn || shp <= thAge)) cls = 1u;
    }
    if (nc > 1u) {
        precise float dk = base + uintBitsToFloat(prm.v[DUST_CADP_AW + 1u]) * g;
        bool fin = dust_cad_finite(dk);
        dl1 = fin ? dk : INF;
        precise float lhs = dk * 2.0 * float(prm.v[DUST_CADP_N + 1u]);
        precise float shp = 2.0 * float(prm.v[DUST_CADP_N + 1u]) * dt;
        if (fin && lhs <= eps && (!shapeOn || shp <= thAge)) cls = 2u;
    }
    if (nc > 2u) {
        precise float dk = base + uintBitsToFloat(prm.v[DUST_CADP_AW + 2u]) * g;
        bool fin = dust_cad_finite(dk);
        dl2 = fin ? dk : INF;
        precise float lhs = dk * 2.0 * float(prm.v[DUST_CADP_N + 2u]);
        precise float shp = 2.0 * float(prm.v[DUST_CADP_N + 2u]) * dt;
        if (fin && lhs <= eps && (!shapeOn || shp <= thAge)) cls = 3u;
    }
    return cls;
}

#endif
