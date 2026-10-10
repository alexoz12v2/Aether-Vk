// @assets/sim/dust_common.glsl
//
// Dust system v3: stateless Keplerian super-particles. GLSL mirror of
// `aethervk.core/rlib/src/scene/dust.rs` and `dust/df.rs` (the reference implementation, unit
// tested). Keep every function, constant and layout in sync with the Rust side.
//
// Precision: no shaderFloat64 on the baseline (Mali-G52), so positions, velocities, times, mu and
// the Kepler solve use double-float "df64" (vec2 = hi + lo, ~48-bit mantissa). Every temporary in
// the df_* primitives is `precise` (SPIR-V NoContraction): the error-free transformations are
// wrong if the compiler fuses a*b+c into an FMA.
//
// Requires: bufferDeviceAddress. No float64 / int64.
#ifndef DUST_COMMON_GLSL
#define DUST_COMMON_GLSL

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

const uint  DUST_CHILDREN          = 8u;
// LOD (mirror of dust::LOD_* / DUST_CHILD_PX / DUST_TILES_*): children of every on-screen
// cluster are DUST_CHILD_PX dots, as many as cover its projected cloud, within the tier budget
// stable child-pattern id in the low mantissa bits of the beta half-spread (dust::CHILD_ID_BITS)
const uint  DUST_CHILD_ID_MASK     = (1u << 16u) - 1u;
const float DUST_CHILD_PX          = 1.5;
const uint  DUST_TILES_X           = 64u;
const uint  DUST_TILES_Y           = 36u;
const uint  DUST_TILE_SAMPLES      = 4u;
const uint  DUST_KEPLER_ITERS_F32  = 32u;
const uint  DUST_KEPLER_ITERS_DF   = 4u;
const float DUST_KEPLER_TOL_F32    = 2.4e-7;
const float DUST_KEPLER_TOL_DF     = 1.42e-14;
const float DUST_CHILD_SIGMA_V_REL = 0.05;
// stream lateral dispersion in cone-cell radii (dust::STREAM_SIGMA_CELLS)
const float DUST_STREAM_SIGMA_CELLS = 1.0;
const float DUST_PI_F32            = 3.14159265;
// Streaklines (mirror of dust::DUST_STREAMS / STREAM_* / STREAK_* / RENDER_*): cluster j = i*S + s
// is time sample i of stream s (fixed cone cell, size stratum and speed per jet); in a tier's
// render buffer its stream predecessor is r - S, and the dust between them is drawn as a streak.
const uint  DUST_STREAMS             = 64u;
const float DUST_STREAM_DIR_JITTER   = 0.03;
const float DUST_STREAM_TIME_JITTER  = 0.1;
const uint  DUST_STREAM_BREAK_BIT    = 0x80000000u;  // sign bit of the beta half-spread field
const float DUST_STREAM_BREAK_TURNS  = 0.01;
const float DUST_STREAM_BREAK_FRACTION = 0.75;      // dust::STREAM_BREAK_FRACTION
const uint  DUST_BATCH_BREAK_FLAG    = 16u;          // mass_params.w = shift + 16 * break
const uint  DUST_BATCH_PROVISIONAL_FLAG = 32u;       //   + 32 * provisional (last sample: now)
const float DUST_STREAK_MARGIN       = 0.05;
const float DUST_STREAK_MARGIN_MAX   = 8.0;
const float DUST_STREAK_PHI_INV      = 0.618034;
// View aids (mirror of dust::DUST_VIEW_* / TRACER_* / FLOW_* / LOD_HEADER_*): tracers are bright
// dots on a sparse stable set of real particles; flow pulses are synchrone bands travelling outward
const uint  DUST_VIEW_TRACERS        = 1u;
const uint  DUST_VIEW_FLOW           = 2u;
const uint  DUST_VIEW_DEBUG_STREAM = 1u << 16u;  // diagnostics: one stream (dust::DUST_VIEW_DEBUG_STREAM)
const uint  DUST_VIEW_DEBUG_SAMPLE = 1u << 17u;  // diagnostics: one time sample in every m
const uint  DUST_TRACER_EVERY        = 256u;
const uint  DUST_TRACER_CHILD        = 1023u;
const float DUST_TRACER_PX           = 2.0;
const float DUST_TRACER_LEVEL        = 0.5;
const float DUST_FLOW_SHARE          = 0.85;
const float DUST_FLOW_KAPPA          = 4.0;
const float DUST_FLOW_I0_KAPPA       = 11.301922;
const float DUST_CHILD_PX_MAX        = 12.0;
// render cluster y word: ring slot | stream shift << 27 | live << 31 (dust::render_word)
const uint  DUST_RENDER_SLOT_MASK    = (1u << 22u) - 1u;
const uint  DUST_RENDER_SHIFT_BIT0   = 27u;
const uint  DUST_RENDER_LIVE_BIT     = 0x80000000u;
// ─── v4 render (mirror of dust::splat::*): Gaussian packets, capsule splats, pyramid ───
const uint  DUST_PYRAMID_HEADER_WORDS = 256u;
const uint  DUST_PYRAMID_TEXEL_WORDS  = 5u;   // tau, tau*r, tau*g, tau*b (fixed point), nearest depth (AU bits)
const float DUST_COLOR_SCALE          = 16.0; // the colour words over the tau counts (dust::DUST_COLOR_SCALE)
const uint  DUST_PYR_LEVELS           = 0u;
const uint  DUST_PYR_WIDTH            = 1u;
const uint  DUST_PYR_HEIGHT           = 2u;
const uint  DUST_PYR_TAU_MAX          = 3u;   // largest packet peak tau per px^2 (f32 bits, atomicMax)
const uint  DUST_PYR_FLAGS            = 4u;   // DUST_VIEW_* of the frame
const uint  DUST_PYR_T_HI             = 5u;   // flow clock
const uint  DUST_PYR_T_LO             = 6u;
const uint  DUST_PYR_LEVEL_MASK       = 7u;   // bit l: level l was touched (atomicOr)
const uint  DUST_PYR_TRACER_COUNTS    = 8u;   // tracer dot height in counts (f32 bits)
const uint  DUST_PYR_WHITE            = 9u;   // f32 bits: this frame's white point (radiance factor, host-written)
const uint  DUST_PYR_SOFTENING        = 10u;  // f32 bits: this frame's auto softening (host-written)
const uint  DUST_PYR_PHASE_LUT        = 64u;  // 181 f32 bits: the dust phase function at 0..180 degrees (dust::PYR_PHASE_LUT)
const uint  DUST_PYR_MEASURE          = 12u;  // the white-point measurement grid: offset, width, height
const uint  DUST_PYR_UNIT             = 15u;  // f32 bits: the fixed-point unit of this frame's counts (host-written, for dump readers)
const uint  DUST_WHITE_LEVEL          = 4u;
const uint  DUST_PYR_TABLE            = 16u;  // per level: offset (words), width, height
const float DUST_SPLAT_SIGMA_TEXELS   = 2.0;
const float DUST_PIXEL_FILTER_VAR     = 0.25;
const float DUST_SPLAT_SIGMAS         = 3.0;
// the size polyline (dust::SIZE_BINS / SIZE_EDGES / CHORD_VAR_FACTOR / TIME_PIECES_MAX)
const uint  DUST_SIZE_BINS            = 16u;
const uint  DUST_SIZE_EDGES           = 17u;
const float DUST_SIZE_RANGE_FACTOR    = 10.0;   // dust::SIZE_RANGE_FACTOR (sizes s_ref/F .. s_ref*F)
const float DUST_CHORD_VAR_FACTOR     = 1.21;
const uint  DUST_TIME_PIECES_MAX      = 4u;
const uint  DUST_TIME_SWEEP_MAX       = 4u;    // dust::TIME_SWEEP_MAX
const float DUST_LIT_ARC_END_EPS      = 1e-5;  // dust::LIT_ARC_END_EPS
const float DUST_SPIRAL_MIN_TURN      = 0.01;  // dust::SPIRAL_MIN_TURN (rad)
const float DUST_SPIRAL_MAX_AGE_SAMPLES = 8.0; // dust::SPIRAL_MAX_AGE_SAMPLES
const float DUST_TIME_SWEEP_STEP_PX   = 1.0;   // dust::TIME_SWEEP_STEP_PX
const float DUST_MERGE_PX             = 1.0;    // dust::DUST_MERGE_PX
const float DUST_DET_ANISO_FLOOR      = 1e-6;   // dust::DUST_DET_ANISO_FLOOR
const float DUST_COUNT_MAX_PER_ADD    = 16777216.0;
const float DUST_SPLAT_MAX_TEXELS     = 16384.0;
const float DUST_AU_M                 = 1.495978707e11;

// ─── df64 primitives (mirror of dust::df) ──────────────────────────────────
struct Df3 {
    vec3 hi;
    vec3 lo;
};

const vec2 DF_SUN_MU = vec2(1.32712443e+20, -3.23726082e+12);
const vec2 DF_TWO_PI = vec2(6.28318548e+00, -1.74845553e-07);
// 1/k!, k = 0..15
const vec2 DF_INV_FACT[16] = vec2[16](
    vec2(1.0, 0.0),
    vec2(1.0, 0.0),
    vec2(5.000000000e-01, 0.0),
    vec2(1.666666716e-01, -4.967053879e-09),
    vec2(4.166666791e-02, -1.241763470e-09),
    vec2(8.333333768e-03, -4.346172033e-10),
    vec2(1.388888923e-03, -3.363109444e-11),
    vec2(1.984127011e-04, -2.725596875e-12),
    vec2(2.480158764e-05, -3.406996094e-13),
    vec2(2.755731884e-06, 3.793571224e-14),
    vec2(2.755731998e-07, -7.575112209e-15),
    vec2(2.505210794e-08, 4.417623045e-16),
    vec2(2.087675588e-09, 1.108283981e-16),
    vec2(1.605904437e-10, -5.352526512e-18),
    vec2(1.147074536e-11, 2.372207689e-19),
    vec2(7.647163610e-13, 1.220071047e-20)
);

vec2 df_two_sum(float a, float b) {
    precise float s = a + b;
    precise float bb = s - a;
    precise float e = (a - (s - bb)) + (b - bb);
    return vec2(s, e);
}

// requires |a| >= |b| (or a == 0)
vec2 df_quick_two_sum(float a, float b) {
    precise float s = a + b;
    precise float e = b - (s - a);
    return vec2(s, e);
}

// exact product a*b = hi + lo (Dekker, no FMA)
vec2 df_two_prod(float a, float b) {
    precise float p = a * b;
    precise float ta = 4097.0 * a;
    precise float ah = ta - (ta - a);
    precise float al = a - ah;
    precise float tb = 4097.0 * b;
    precise float bh = tb - (tb - b);
    precise float bl = b - bh;
    precise float e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    return vec2(p, e);
}

vec2 df_add(vec2 a, vec2 b) {
    vec2 s = df_two_sum(a.x, b.x);
    vec2 t = df_two_sum(a.y, b.y);
    precise float e1 = s.y + t.x;
    s = df_quick_two_sum(s.x, e1);
    precise float e2 = s.y + t.y;
    return df_quick_two_sum(s.x, e2);
}
vec2 df_sub(vec2 a, vec2 b) { return df_add(a, -b); }

vec2 df_add_f(vec2 a, float b) {
    vec2 s = df_two_sum(a.x, b);
    precise float e = s.y + a.y;
    return df_quick_two_sum(s.x, e);
}

vec2 df_mul(vec2 a, vec2 b) {
    vec2 p = df_two_prod(a.x, b.x);
    precise float e = p.y + (a.x * b.y + a.y * b.x);
    return df_quick_two_sum(p.x, e);
}

vec2 df_mul_f(vec2 a, float b) {
    vec2 p = df_two_prod(a.x, b);
    precise float e = p.y + a.y * b;
    return df_quick_two_sum(p.x, e);
}

// exact scaling by a power of two
vec2 df_scale_pow2(vec2 a, float p) {
    precise vec2 r = a * p;
    return r;
}

vec2 df_div(vec2 a, vec2 b) {
    float q1 = a.x / b.x;
    vec2 r = df_sub(a, df_mul_f(b, q1));
    float q2 = r.x / b.x;
    r = df_sub(r, df_mul_f(b, q2));
    float q3 = r.x / b.x;
    return df_add_f(df_quick_two_sum(q1, q2), q3);
}

vec2 df_sqrt(vec2 a) {
    if (!(a.x > 0.0)) return vec2(0.0);
    float x = sqrt(a.x);
    vec2 r = df_sub(a, df_two_prod(x, x));
    precise float c = r.x / (2.0 * x);
    return df_quick_two_sum(x, c);
}

vec2 df_floor(vec2 a) {
    float fh = floor(a.x);
    if (fh == a.x) return df_quick_two_sum(fh, floor(a.y));
    return vec2(fh, 0.0);
}

Df3 df3_make(vec2 x, vec2 y, vec2 z) {
    Df3 r;
    r.hi = vec3(x.x, y.x, z.x);
    r.lo = vec3(x.y, y.y, z.y);
    return r;
}
Df3 df3_from_f32(vec3 v) {
    Df3 r;
    r.hi = v;
    r.lo = vec3(0.0);
    return r;
}
vec2 df3_get(Df3 a, int i) { return vec2(a.hi[i], a.lo[i]); }
Df3 df3_add(Df3 a, Df3 b) {
    return df3_make(df_add(df3_get(a, 0), df3_get(b, 0)), df_add(df3_get(a, 1), df3_get(b, 1)), df_add(df3_get(a, 2), df3_get(b, 2)));
}
Df3 df3_sub(Df3 a, Df3 b) {
    return df3_make(df_sub(df3_get(a, 0), df3_get(b, 0)), df_sub(df3_get(a, 1), df3_get(b, 1)), df_sub(df3_get(a, 2), df3_get(b, 2)));
}
Df3 df3_scale(Df3 a, vec2 s) {
    return df3_make(df_mul(df3_get(a, 0), s), df_mul(df3_get(a, 1), s), df_mul(df3_get(a, 2), s));
}
vec2 df3_dot(Df3 a, Df3 b) {
    return df_add(df_add(df_mul(df3_get(a, 0), df3_get(b, 0)), df_mul(df3_get(a, 1), df3_get(b, 1))), df_mul(df3_get(a, 2), df3_get(b, 2)));
}
// nearest f32 vector (hi + lo rounded once)
vec3 df3_to_f32(Df3 a) {
    precise vec3 r = a.hi + a.lo;
    return r;
}

// ─── GPU layouts (mirror of DustCluster / DustRenderCluster / DustBatch) ────

// 80 B: immutable emission record
struct DustCluster {
    vec4 r0_t0_hi;    // heliocentric position (m), w = t0 (scaled s); df64 high part
    vec4 r0_t0_lo;    // ... low part
    vec4 v0_hi_beta;  // heliocentric velocity (m/s) high part, w = beta
    vec4 v0_lo_mass;  // velocity low part, w = super-particle mass (g)
    vec4 misc;        // x sigma_lat (m/s), y sigma_rad (m/s), z mean cross-section per gram (m^2/g),
                      // w half log-size range (low 16 bits: child-pattern id, dust::child_id; sign: stream break)
    vec4 eject;       // xyz ejection velocity of the reference grain dir * v_ref (m/s, root), w s_ref (um)
};

// 32 B: per-frame evaluation, written compactly (index = live-range offset)
struct DustRenderCluster {
    vec4 pos_size;           // particle-system local position (m), w spread radius (m)
    vec4 age_id_dbeta_flux;  // x age (s), y ring slot (uint bits), z child beta half-spread (low 16 bits: child-pattern id),
                             // w flux (m^2, 0 = culled)
};

// 208 B: emission batch
struct DustBatch {
    vec4  comet_r_t_hi;     // jet heliocentric position (m), w = t_start (s); df64 high part
    vec4  comet_r_t_lo;     // ... low part
    vec4  comet_v_dur_hi;   // jet heliocentric velocity (m/s), w = duration (s); df64 high part
    vec4  comet_v_dur_lo;   // ... low part
    vec4  rot_start;        // ps -> root quaternion (xyzw) at t_start
    vec4  spin;             // nucleus spin: unit axis (root frame), w omega >= 0 (rad/s)
    vec4  jet_dir_aperture; // jet direction (ps frame), w half aperture (rad)
    vec4  size_params;      // s_min, s_max (um), 4 - q, mass normalization
    vec4  vel_params;       // v_ref (m/s), relative speed std, s_ref (um), beta * s (um)
    vec4  mass_params;      // batch mass (g), density (g/cm^3), size-stratum key (u01, 24 bits), 0
    uint  first_index;
    uint  count;
    uint  ring_mask;
    uint  seed;
    vec4  lit;              // jet site illumination: psi_start, psi0, total lit phase (rad), mode
    vec4  site_offset;      // jet site offset from the nucleus centre (m, ps frame), w unused
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer DustClusterBuffer {
    DustCluster c[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) buffer DustRenderBuffer {
    DustRenderCluster c[];
};
// 80 B: per-frame second moments of a cluster (dust_propagate.comp -> dust_splat.comp), mirror of
// dust::DustMoments
struct DustMoments {
    vec4 mean_flux;      // ps local mean of the reference grain (m), w flux (m^2, 0 = culled)
    vec4 cov_a;          // Sigma_v xx xy xz yy (m^2)
    vec4 cov_b_age_id;   // yz zz, z age (s), w child-pattern id (uint bits)
    vec4 edges[DUST_SIZE_EDGES]; // size polyline: xyz local position of bin edge j (m); w: edges 0 and 16 their
                                 // speed factor sqrt(beta_j/beta) (linear in j, dust_edge_factor), edge 1 the
                                 // nucleus spin rate (rad/s), edges 2..4 its unit axis (dust_spiral_point)
};
// speed factor sqrt(beta_j/beta) of edge j (dust::DustMoments::edge_factor)
float dust_edge_factor(DustMoments M, uint j) {
    float lo = M.edges[0].w, hi = M.edges[DUST_SIZE_BINS].w;
    return lo + (hi - lo) * (float(j) / float(DUST_SIZE_BINS));
}
float dust_spin_rate(DustMoments M) { return M.edges[1].w; }
vec3 dust_spin_axis(DustMoments M) { return vec3(M.edges[2].w, M.edges[3].w, M.edges[4].w); }
float dust_helio_au(DustMoments M) { return M.edges[5].w; }
vec3 dust_sun_dir(DustMoments M) { return vec3(M.edges[6].w, M.edges[7].w, M.edges[8].w); }
layout(buffer_reference, std430, buffer_reference_align = 16) buffer DustMomentsBuffer {
    DustMoments m[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer DustUintBuffer {
    uint v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer DustBatchRef {
    DustBatch b;
};

// ─── RNG and sampling (bit-identical u32 ops) ──────────────────────────────
uint dust_pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float dust_u01(uint h) { return float(h >> 8u) * (1.0 / 16777216.0); }

// Hashed permutation of [0, n) keyed by p (Kensler 2013), cycle-walked from the next power of two
// (mirror of `dust::permute_index`, u32 ops only: bit-identical)
uint dust_permute(uint i, uint n, uint p) {
    n = max(n, 1u);
    uint w = n - 1u;
    w |= w >> 1u; w |= w >> 2u; w |= w >> 4u; w |= w >> 8u; w |= w >> 16u;
    do {
        i ^= p; i *= 0xE170893Du;
        i ^= p >> 16u;
        i ^= (i & w) >> 4u;
        i ^= p >> 8u; i *= 0x0929EB3Fu;
        i ^= p >> 23u;
        i ^= (i & w) >> 1u; i *= 1u | (p >> 27u);
        i *= 0x6935FA69u;
        i ^= (i & w) >> 11u; i *= 0x74DCB303u;
        i ^= (i & w) >> 2u; i *= 0x9E501CC3u;
        i ^= (i & w) >> 2u; i *= 0xC860A3DFu;
        i &= w;
        i ^= i >> 5u;
    } while (i >= n);
    return (i + p) % n;
}

vec3 dust_qrot(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Hamilton product a * b (xyzw)
vec4 dust_qmul(vec4 a, vec4 b) {
    return vec4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// rotation by `angle` about unit `axis`; half angle reduced to [-pi, pi) (sin/cos accuracy range)
vec4 dust_qaxis_angle(vec3 axis, float angle) {
    const float TWO_PI = 2.0 * DUST_PI_F32;
    precise float h = 0.5 * angle;
    precise float k = floor((h + DUST_PI_F32) * (1.0 / TWO_PI));
    precise float hr = h - k * TWO_PI;
    return vec4(axis * sin(hr), cos(hr));
}

// Emission offset dt in [0, dur] of the stratified sample u, uniform over the lit time of the
// batch window. Mirror of `dust::lit_time_map`.
const float DUST_LIT_MODE_PERIODIC = 1.0;
float dust_lit_time_map(float u, vec4 lit, float omega, float dur) {
    const float TWO_PI = 2.0 * DUST_PI_F32;
    if (!(lit.w == DUST_LIT_MODE_PERIODIC) || !(omega > 0.0) || !(lit.y > 0.0)) return u * dur;
    float ps = lit.x, p0 = lit.y;
    precise float arc = 2.0 * p0;
    float seg_start, arc_start;
    if (ps < -p0)     { seg_start = -p0;          arc_start = -p0; }
    else if (ps < p0) { seg_start = ps;           arc_start = -p0; }
    else              { seg_start = TWO_PI - p0;  arc_start = TWO_PI - p0; }
    precise float seg_len = arc_start + arc - seg_start;
    precise float m = u * lit.z;
    precise float psi;
    // the ends of the lit arcs are inclusive (dust::lit_time_map): u = 1 is the end of the
    // last lit arc, never the start of the next one
    precise float eps = DUST_LIT_ARC_END_EPS * arc;
    if (m <= seg_len + eps) {
        psi = seg_start + min(m, seg_len);
    } else {
        precise float mr = m - seg_len;
        precise float k = floor(mr / arc);
        precise float r = mr - k * arc;
        if (r <= eps) psi = arc_start + k * TWO_PI + arc;
        else          psi = arc_start + TWO_PI + k * TWO_PI + r;
    }
    precise float dt = (psi - ps) / omega;
    return clamp(dt, 0.0, dur);
}

vec3 dust_sample_cone(float u1, float u2, vec3 dir, float aperture) {
    float cosA = cos(aperture);
    float z = 1.0 + (cosA - 1.0) * u2;
    float sinT = sqrt(max(1.0 - z * z, 0.0));
    float phi = 2.0 * DUST_PI_F32 * u1;
    vec3 local = vec3(sinT * cos(phi), sinT * sin(phi), z);
    vec3 up = abs(dir.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(up, dir));
    vec3 b = cross(dir, t);
    return t * local.x + b * local.y + dir * local.z;
}

float dust_gauss(float u1, float u2) {
    return sqrt(-2.0 * log(max(u1, 1e-7))) * cos(2.0 * DUST_PI_F32 * u2);
}


// ─── Streams (mirror of dust::batch_streams / stream_* / size_stratum_mass) ───
// (stream shift, break before the batch) of a batch
uvec2 dust_batch_streams(DustBatch B) {
    uint w = uint(B.mass_params.w);
    return uvec2(w % DUST_BATCH_BREAK_FLAG, (w / DUST_BATCH_BREAK_FLAG) & 1u);
}

// Emission-time quantile of cluster j = i*S + s: time stratum i, a jitter shared by the streams of
// the sample plus DUST_STREAM_TIME_JITTER of the stratum per cluster
float dust_stream_time_u01(DustBatch B, uint j) {
    uint shift = dust_batch_streams(B).x;
    uint samples = max(B.count >> shift, 1u);
    uint i = j >> shift;
    // the last sample of every window sits at the window end (the provisional one: now), so the
    // windows' samples tile the time axis (dust::stream_time_u01)
    if (i + 1u == samples) return 1.0;
    uint hs = dust_pcg(B.seed ^ dust_pcg(i ^ 0x3C6EF372u));
    uint hc = dust_pcg(B.seed ^ dust_pcg(j));
    precise float x = float(i) + (1.0 - DUST_STREAM_TIME_JITTER) * dust_u01(hs);
    precise float y = x + DUST_STREAM_TIME_JITTER * dust_u01(hc);
    return y / float(samples);
}

// Whether the jet site was dark since the previous sample of cluster j's stream
// (dust::stream_dark_before)
bool dust_stream_dark_before(DustBatch B, uint j) {
    uvec2 st = dust_batch_streams(B);
    uint i = j >> st.x;
    float omega = B.spin.w;
    if (!(B.lit.w == DUST_LIT_MODE_PERIODIC) || !(omega > 0.0)) return false;
    float dur = B.comet_v_dur_hi.w + B.comet_v_dur_lo.w;
    float u = dust_stream_time_u01(B, j);
    float uPrev = 0.0, dtPrev = 0.0;
    if (i != 0u) {
        uPrev = dust_stream_time_u01(B, j - (1u << st.x));
        dtPrev = dust_lit_time_map(uPrev, B.lit, omega, dur);
    }
    float dt = dust_lit_time_map(u, B.lit, omega, dur);
    precise float dark = omega * (dt - dtPrev) - (u - uPrev) * B.lit.z;
    // a break needs a real night (more than DUST_STREAM_BREAK_TURNS of a turn) that also
    // dominates the interval (DUST_STREAM_BREAK_FRACTION of it): an old tier's samples are days
    // apart and span several day/night cycles, which the capsule averages (dust::stream_dark_before)
    float elapsed = omega * (dt - dtPrev);
    return dark > 2.0 * DUST_PI_F32 * DUST_STREAM_BREAK_TURNS && dark > DUST_STREAM_BREAK_FRACTION * elapsed;
}

// Whether the stream of cluster j is interrupted before it: missing previous window or a dark
// site (dust::stream_breaks_before)
bool dust_stream_breaks_before(DustBatch B, uint j) {
    uvec2 st = dust_batch_streams(B);
    if ((j >> st.x) == 0u && st.y != 0u) return true;
    return dust_stream_dark_before(B, j);
}

// Harmonic mean size of the distribution 1/<1/s> over the mass (um): the grain whose
// cross-section per gram is the distribution's mean (dust::size_mean_inv_s_um).
// sizeParams = (s_min, s_max, e = 4 - q, .): <1/s> = int s^(e-2) ds / int s^(e-1) ds
float dust_size_int(float a, float b, float k) {
    if (abs(k + 1.0) < 1e-9) return log(b / a);
    return (pow(b, k + 1.0) - pow(a, k + 1.0)) / (k + 1.0);
}
float dust_size_mean_inv_s_um(vec4 sizeParams) {
    float a = sizeParams.x, b = sizeParams.y, e = sizeParams.z;
    float meanInv = dust_size_int(a, b, e - 2.0) / dust_size_int(a, b, e - 1.0);
    if (meanInv > 0.0 && !isinf(meanInv) && !isnan(meanInv)) return 1.0 / meanInv;
    return sqrt(a * b);
}

// sqrt(beta_j) of size-bin edge j for a reference beta: equal cross-section bins of n(s) ~ s^-3.5
// are equally spaced in sqrt(beta) between beta/F and beta*F (dust::size_edge_sqrt_beta)
float dust_size_edge_sqrt_beta(float beta, float f, uint j) {
    f = max(f, 1.0);
    float lo = sqrt(max(beta, 0.0) / f);
    float hi = sqrt(max(beta, 0.0) * f);
    return lo + (hi - lo) * (float(j) / float(DUST_SIZE_BINS));
}
// The cluster's size range factor F from misc.w (half log-size range; child id in the low
// mantissa bits, break in the sign bit) (dust::size_range_factor)
float dust_size_range_factor(DustCluster C) {
    uint bits = floatBitsToUint(C.misc.w) & ~DUST_CHILD_ID_MASK & 0x7FFFFFFFu;
    float halfLn = uintBitsToFloat(bits);
    if (isnan(halfLn) || isinf(halfLn) || halfLn < 0.0) return 1.0;
    return exp(halfLn);
}

// Orthonormal frame with e3 along the ejection (dust::dispersion_frame): the root axes when zero
void dust_dispersion_frame(vec3 eject, out vec3 e1, out vec3 e2, out vec3 e3) {
    float n = length(eject);
    if (!(n > 0.0)) { e1 = vec3(1.0, 0.0, 0.0); e2 = vec3(0.0, 1.0, 0.0); e3 = vec3(0.0, 0.0, 1.0); return; }
    e3 = eject / n;
    vec3 a = abs(e3.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    e1 = normalize(cross(a, e3));
    e2 = cross(e3, e1);
}

// ─── Streaklines (mirror of dust::streak_*) ─────────────────────────────────
uint dust_render_word(uint slot, uint shift) {
    return (slot & DUST_RENDER_SLOT_MASK) | ((shift & 0xFu) << DUST_RENDER_SHIFT_BIT0) | DUST_RENDER_LIVE_BIT;
}
bool dust_render_live(float y) { return (floatBitsToUint(y) & DUST_RENDER_LIVE_BIT) != 0u; }
uint dust_render_shift(float y) { return (floatBitsToUint(y) >> DUST_RENDER_SHIFT_BIT0) & 0xFu; }
bool dust_stream_break(float dbetaField) { return (floatBitsToUint(dbetaField) & DUST_STREAM_BREAK_BIT) != 0u; }

// Stream predecessor of render cluster r (r - S, the same stream's previous, older time sample),
// -1 for the first S clusters, after a stream break, or outside the tier's age band. Reads only
// the y / z words, which the LOD never rewrites (it rewrites w concurrently).
int dust_streak_pred(DustRenderBuffer render, uint r) {
    vec2 yz = render.c[r].age_id_dbeta_flux.yz;
    uint n = 1u << dust_render_shift(yz.x);
    if (r < n || dust_stream_break(yz.y)) return -1;
    return dust_render_live(render.c[r - n].age_id_dbeta_flux.y) ? int(r - n) : -1;
}

// The stable child-pattern id in the low bits of the beta half-spread field (mirror of
// dust::child_id)
uint dust_child_id(float dbetaField) { return floatBitsToUint(dbetaField) & DUST_CHILD_ID_MASK; }
// The cluster with child-pattern id `id` carries a tracer dot (mirror of dust::is_tracer)
bool dust_is_tracer(uint id) {
    return dust_pcg((id & DUST_CHILD_ID_MASK) ^ 0x7AC312E5u) % DUST_TRACER_EVERY == 0u;
}

// PHASE-LUT-BEGIN (generated by scripts/gen_phase_lut.py; the 181 values of
// assets/sim/data/dustphaseHM_table.txt, the Lowell composite dust phase function, Schleicher 2010,
// normalised to 1 at 0 degrees, are dust::phase::PHASE_HM_0DEG and travel in the pyramid header at
// DUST_PYR_PHASE_LUT: a const float[181] in the shader with a dynamic index changed the splat's
// results on the GPU even when unused. Do not edit by hand.)
// Phi(alpha) for a phase angle in radians (clamped to 0..pi), linear between the table's degrees
// (dust::phase::phase_function)
float dust_phase_function(DustUintBuffer pyr, float alphaRad) {
    float deg = clamp(alphaRad * (180.0 / DUST_PI_F32), 0.0, 180.0);
    uint i = min(uint(deg), 179u);
    float f = deg - float(i);
    float a = uintBitsToFloat(pyr.v[DUST_PYR_PHASE_LUT + i]);
    float b = uintBitsToFloat(pyr.v[DUST_PYR_PHASE_LUT + i + 1u]);
    return a + (b - a) * f;
}
// PHASE-LUT-END
// The radiance weight of a packet, p * Phi(alpha) / r_h^2 (dust::radiance_weight): sunDir the unit
// direction to the Sun, toEye the vector to the eye (any length, zero = alpha 0), r_h in AU
float dust_radiance_weight(DustUintBuffer pyr, float albedo, vec3 sunDir, vec3 toEye, float rhAu) {
    float n = dot(toEye, toEye);
    float cosA = n > 0.0 ? dot(sunDir, toEye) / sqrt(n) : 1.0;
    float alpha = acos(clamp(cosA, -1.0, 1.0));
    float r = max(rhAu, 1e-6);
    return albedo * dust_phase_function(pyr, alpha) / (r * r);
}

// ─── Flow (mirror of dust::flow_*): brightness marks on synchrones (dust of one emission instant),
// Lagrangian timelines where the real motion is visible on screen, a log-age time-lapse elsewhere

// von Mises pulse of phase x (turns), mean 1 over a turn
float dust_flow_pulse_of(float x) {
    x = x - floor(x);
    float c = cos(2.0 * DUST_PI_F32 * x);
    return (1.0 - DUST_FLOW_SHARE) + DUST_FLOW_SHARE * exp(DUST_FLOW_KAPPA * c) / DUST_FLOW_I0_KAPPA;
}

// fract(t_e / 2^j) of the emission epoch t_e = T - age on the flow clock T, exact
// (dust::epoch_phase)
float dust_epoch_phase(float tHi, float tLo, float age, int j) {
    float inv = uintBitsToFloat(uint(127 - clamp(j, -126, 127)) << 23u);
    precise float a = tHi * inv; a = a - floor(a);
    precise float b = tLo * inv; b = b - floor(b);
    precise float x = a + b - age * inv;
    return x - floor(x);
}

// Flow factor of a dot of age `age`: Lagrangian timelines, marks at emission epochs
// t_e = T - age = 0 mod 2^j, j = floor(log2 age) and j + 1 (dust::flow_factor): they ride the real
// particles, K x faster on the flow clock
float dust_flow_factor(float age, float tHi, float tLo) {
    float a = max(age, 1.0);
    float l = log(a) * 1.4426950408889634;
    float jf = floor(l);
    float w = l - jf;
    int j = int(jf);
    return (1.0 - w) * dust_flow_pulse_of(dust_epoch_phase(tHi, tLo, a, j))
         + w * dust_flow_pulse_of(dust_epoch_phase(tHi, tLo, a, j + 1));
}

// ─── v4 render (mirror of dust::splat) ──────────────────────────────────────

// erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7); mirror of dust::splat::erf
float dust_erf(float x) {
    float s = x < 0.0 ? -1.0 : 1.0;
    x = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * x);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return s * y;
}

// pyramid level of a splat of smallest screen sigma `sigmaPx`: texels of 2^l px with sigma >=
// DUST_SPLAT_SIGMA_TEXELS texels, the last level at most (dust::splat_level)
uint dust_splat_level(float sigmaPx, uint levels) {
    float r = sigmaPx / DUST_SPLAT_SIGMA_TEXELS;
    if (!(r > 1.0)) return 0u;
    float l = floor(log2(r));
    return min(uint(max(l, 0.0)), max(levels, 1u) - 1u);
}

// Hue ramp position of an age: clamp(age / TTL, 0, 1), a linear clock to the complement at one TTL
// of the system, constant beyond; TTL <= 0 = no rotation (dust::age_hue_fraction)
float dust_age_hue_fraction(float age, float ttl) {
    return ttl > 0.0 ? clamp(age / ttl, 0.0, 1.0) : 0.0;
}

vec3 dust_oklab_from_linear(vec3 c) {
    float l = 0.41222147 * c.r + 0.53633255 * c.g + 0.051445995 * c.b;
    float m = 0.2119035 * c.r + 0.6806995 * c.g + 0.10739696 * c.b;
    float s = 0.08830246 * c.r + 0.28171885 * c.g + 0.6299787 * c.b;
    l = pow(max(l, 0.0), 1.0 / 3.0); m = pow(max(m, 0.0), 1.0 / 3.0); s = pow(max(s, 0.0), 1.0 / 3.0);
    return vec3(0.21045426 * l + 0.7936178 * m - 0.004072047 * s,
                1.9779985 * l - 2.4285922 * m + 0.4505937 * s,
                0.025904037 * l + 0.78277177 * m - 0.80867577 * s);
}
vec3 dust_linear_from_oklab(vec3 lab) {
    float l = lab.x + 0.39633778 * lab.y + 0.21580376 * lab.z;
    float m = lab.x - 0.105561346 * lab.y - 0.06385417 * lab.z;
    float s = lab.x - 0.08948418 * lab.y - 1.2914855 * lab.z;
    l = l * l * l; m = m * m * m; s = s * s * s;
    return vec3(4.0767417 * l - 3.3077116 * m + 0.23096994 * s,
                -1.268438 * l + 2.6097574 * m - 0.34131938 * s,
                -0.004196086 * l - 0.7034186 * m + 1.7076147 * s);
}

// Linear sRGB of an OKLab colour brought into the [0, 1] gamut by reducing its chroma (lightness
// and hue kept), 8 bisection steps on the chroma scale without early exit, then clamped; a colour
// in gamut is the plain conversion (dust::oklab_to_gamut)
bool dust_in_gamut(vec3 c) {
    return all(greaterThanEqual(c, vec3(-1e-4))) && all(lessThanEqual(c, vec3(1.0 + 1e-4)));
}
vec3 dust_oklab_to_gamut(vec3 lab) {
    vec3 full = dust_linear_from_oklab(lab);
    vec3 rgb = full;
    if (!dust_in_gamut(full)) {
        float lo = 0.0, hi = 1.0;
        vec3 best = dust_linear_from_oklab(vec3(lab.x, 0.0, 0.0));
        for (int i = 0; i < 8; ++i) {
            float k = 0.5 * (lo + hi);
            vec3 c = dust_linear_from_oklab(vec3(lab.x, k * lab.y, k * lab.z));
            if (dust_in_gamut(c)) { lo = k; best = c; } else { hi = k; }
        }
        rgb = best;
    }
    return clamp(rgb, 0.0, 1.0);
}

// Colour of dust of age `age` in a system of TTL `ttl`: the stream colour's hue rotated in OKLab
// by pi * f(age), the complement from one TTL on, chroma reduced to the gamut (dust::age_color)
vec3 dust_age_color(vec3 stream, float age, float ttl) {
    vec3 lab = dust_oklab_from_linear(clamp(stream, 0.0, 1.0));
    float th = DUST_PI_F32 * dust_age_hue_fraction(age, ttl);
    float s = sin(th), c = cos(th);
    return dust_oklab_to_gamut(vec3(lab.x, c * lab.y - s * lab.z, s * lab.y + c * lab.z));
}

// 32-bit integer hash (lowbias32; dust::dust_hash32)
uint dust_hash32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// a + b saturating at 0xFFFFFFFF (no saturating atomics in core GLSL): compare-and-swap loop
void dust_atomic_add_sat(DustUintBuffer buf, uint i, uint b) {
    if (b == 0u) return;
    uint old = buf.v[i];
    for (;;) {
        uint sum = old + b;
        if (sum < old) sum = 0xFFFFFFFFu;
        uint seen = atomicCompSwap(buf.v[i], old, sum);
        if (seen == old) return;
        old = seen;
    }
}

// A screen packet (dust::ScreenPacket)
struct DustScreenPacket {
    vec2  mu;        // px
    vec3  cov;       // xx, xy, yy (px^2), pixel filter included
    vec2  seg;       // px
    float amp;       // peak tau per px^2
    float sigmaMin;  // px
    float det;       // determinant of cov with the anisotropy floor (dust::ScreenPacket::det)
    float depthAu;
    bool  openStart; // the capsule continues into the neighbouring bin: no cap (dust::Packet::open)
    bool  openEnd;
    vec2  rho;       // density at the segment's ends, mean 1, linear between (dust::Packet::rho)
    // photometry (albedo > 0, dust::ScreenPacket): the mean (local m), local metres per pixel
    // within the view plane (d = dx kx + dy ky), the Sun direction and r_h at the mean, the
    // homogeneous eye (the direction to it from x is eye.xyz - eye.w * x: the eye point when
    // perspective, w = 1; the unit view axis when orthographic, w = 0: that eye is at infinity),
    // the weight at the mean (the peak word); every texel gets its own weight (dust_texel_weight)
    vec3  mean;
    vec3  kx;
    vec3  ky;
    vec3  sun;
    float rhAu;
    vec4  eye;
    float albedo;
    float wMean;
};
// the photometry of the cluster being splatted (set per cluster by dust_splat.comp; albedo <= 0:
// none) and the pyramid that carries the phase table
float dust_phot_albedo = 0.0;
vec3  dust_phot_sun = vec3(0.0, 0.0, 1.0);
float dust_phot_rh = 1.0;
DustUintBuffer dust_phot_lut;
// the radiance weight at the dust a texel holds (dust::texel_weight): the packet's mean displaced
// by the texel's in-plane offset, the Sun direction and r_h moved with it, the phase angle to the eye
float dust_texel_weight(DustScreenPacket P, vec2 x) {
    vec2 dxy = x - P.mu;
    vec3 d = P.kx * dxy.x + P.ky * dxy.y;
    vec3 pos = P.mean + d;
    vec3 rVec = P.sun * (-P.rhAu * DUST_AU_M) + d;
    float rn = max(length(rVec), 1.0);
    return dust_radiance_weight(dust_phot_lut, P.albedo, rVec * (-1.0 / rn), P.eye.xyz - P.eye.w * pos, rn / DUST_AU_M);
}

// the relative density at the two ends of a size bin's chord with edge speed factors f0 <= f1:
// (f0 + f1)/(2 f0) and (f0 + f1)/(2 f1), mean 1 over the chord (dust::bin_density_ends)
vec2 dust_bin_density_ends(float f0, float f1) {
    f0 = max(f0, 1e-6); f1 = max(f1, 1e-6);
    float s = f0 + f1;
    return vec2(s / (2.0 * f0), s / (2.0 * f1));
}

// Projects a packet (mean m, covariance S = xx xy xz yy yz zz m^2, segment d m, flux) with the
// Jacobian of `mvp` at its mean (dust::project_packet). false: behind the camera or off screen.
bool dust_project_packet(vec3 mean, float S[6], vec3 d, float flux, float exposure, mat4 mvp,
                         float W, float H, vec3 eyeLocal, bool openStart, bool openEnd, vec2 rho,
                         out DustScreenPacket P) {
    vec4 clip = mvp * vec4(mean, 1.0);
    if (!(clip.w > 0.0)) return false;
    P.openStart = openStart;
    P.openEnd = openEnd;
    P.rho = rho;
    vec2 ndc = clip.xy / clip.w;
    P.mu = (ndc * 0.5 + 0.5) * vec2(W, H);
    // rows of the mvp (column-major)
    vec3 r0 = vec3(mvp[0][0], mvp[1][0], mvp[2][0]);
    vec3 r1 = vec3(mvp[0][1], mvp[1][1], mvp[2][1]);
    vec3 r3 = vec3(mvp[0][3], mvp[1][3], mvp[2][3]);
    vec3 jx = (r0 - ndc.x * r3) * (0.5 * W / clip.w);
    vec3 jy = (r1 - ndc.y * r3) * (0.5 * H / clip.w);
    mat3 Sm = mat3(S[0], S[1], S[2],
                   S[1], S[3], S[4],
                   S[2], S[4], S[5]);
    vec3 sx = Sm * jx, sy = Sm * jy;
    P.cov = vec3(dot(jx, sx) + DUST_PIXEL_FILTER_VAR, dot(jx, sy), dot(jy, sy) + DUST_PIXEL_FILTER_VAR);
    P.seg = vec2(dot(jx, d), dot(jy, d));
    float g00 = dot(jx, jx), g01 = dot(jx, jy), g11 = dot(jy, jy);
    float detG = max(g00 * g11 - g01 * g01, 0.0);
    // local metres per pixel within the view plane (dust::project_packet_lit)
    if (detG > 1e-30) {
        P.kx = (jx * g11 - jy * g01) / detG;
        P.ky = (jy * g00 - jx * g01) / detG;
    } else {
        P.kx = vec3(0.0);
        P.ky = vec3(0.0);
    }
    // the view direction: the depth gradient of the projection (the w row when perspective, the
    // clip-z row when orthographic), oriented so that the scene's origin lies in front of the eye
    bool perspective = dot(r3, r3) > 0.0;
    vec3 fwd = vec3(0.0);
    {
        vec3 r2 = vec3(mvp[0][2], mvp[1][2], mvp[2][2]);
        vec3 fwdRaw = perspective ? r3 : r2;
        float fn2 = dot(fwdRaw, fwdRaw);
        if (fn2 > 0.0) {
            fwd = fwdRaw / sqrt(fn2);
            if (dot(-eyeLocal, fwd) < 0.0) fwd = -fwd;
        }
    }
    // the photometric eye: the eye point when perspective, at infinity along the view axis when
    // orthographic (dust::ScreenPacket::eye_w)
    P.eye = (perspective || dot(fwd, fwd) == 0.0) ? vec4(eyeLocal, 1.0) : vec4(-fwd, 0.0);
    P.mean = mean;
    P.sun = dust_phot_sun;
    P.rhAu = dust_phot_rh;
    P.albedo = dust_phot_albedo;
    P.wMean = dust_phot_albedo > 0.0
        ? dust_radiance_weight(dust_phot_lut, dust_phot_albedo, dust_phot_sun, P.eye.xyz - P.eye.w * mean, dust_phot_rh)
        : 1.0;

    // `precise`: no fused multiply-add here, so the cancelling determinant rounds exactly as the
    // CPU reference's (an fma changes it by the whole cancellation error, and with it the
    // packet's amplitude and level: whole packets differed between the two)
    precise float half_ = 0.5 * (P.cov.x + P.cov.z);
    precise float dd = sqrt((0.5 * (P.cov.x - P.cov.z)) * (0.5 * (P.cov.x - P.cov.z)) + P.cov.y * P.cov.y);
    float lamMax = max(half_ + dd, DUST_PIXEL_FILTER_VAR);
    // the determinant with the anisotropy floor (f32 cancellation for very elongated packets)
    precise float detRaw = P.cov.x * P.cov.z - P.cov.y * P.cov.y;
    float detC = max(max(detRaw, DUST_DET_ANISO_FLOOR * lamMax * lamMax), 1e-30);
    P.det = detC;
    P.amp = exposure * flux * sqrt(detG) / (2.0 * DUST_PI_F32 * sqrt(detC));
    // the part of the packet in front of the eye (dust::project_packet_lit): a Gaussian along the
    // view direction cut at the eye's plane
    {
        if (dot(fwd, fwd) > 0.0) {
            float depth = dot(mean - eyeLocal, fwd);
            float varV = S[0] * fwd.x * fwd.x + S[3] * fwd.y * fwd.y + S[5] * fwd.z * fwd.z
                       + 2.0 * (S[1] * fwd.x * fwd.y + S[2] * fwd.x * fwd.z + S[4] * fwd.y * fwd.z);
            float sigmaV = sqrt(max(varV, 0.0));
            if (sigmaV > 0.0) P.amp *= 0.5 * (1.0 + dust_erf(depth / (sigmaV * 1.4142135623730951)));
            else if (depth < 0.0) P.amp = 0.0;
        }
    }
    if (!(P.amp > 0.0) || isinf(P.amp)) return false;
    // the smallest sigma from the floored determinant, never from the cancelling half - dd
    P.sigmaMin = sqrt(max(detC / lamMax, DUST_PIXEL_FILTER_VAR * 0.5));
    float ex = DUST_SPLAT_SIGMAS * sqrt(P.cov.x), ey = DUST_SPLAT_SIGMAS * sqrt(P.cov.z);
    float x0 = min(P.mu.x, P.mu.x + P.seg.x) - ex, x1 = max(P.mu.x, P.mu.x + P.seg.x) + ex;
    float y0 = min(P.mu.y, P.mu.y + P.seg.y) - ey, y1 = max(P.mu.y, P.mu.y + P.seg.y) + ey;
    if (x1 < 0.0 || y1 < 0.0 || x0 > W || y0 > H || !(x0 <= x1) || !(y0 <= y1)) return false;
    P.depthAu = length(mean - eyeLocal) / DUST_AU_M;
    if (any(isnan(P.mu)) || any(isinf(P.mu)) || any(isnan(P.cov)) || any(isinf(P.cov))
        || any(isnan(P.seg)) || any(isinf(P.seg)) || isnan(P.sigmaMin) || isnan(P.depthAu)) return false;
    return true;
}

// Optical depth per px^2 of the capsule at pixel x (dust::capsule_tau)
float dust_capsule_tau(DustScreenPacket P, vec2 x) {
    float det = P.det;
    if (!(det > 0.0)) return 0.0;
    float m00 = P.cov.z / det, m01 = -P.cov.y / det, m11 = P.cov.x / det;
    vec2 e = x - P.mu;
    float c = m00 * e.x * e.x + 2.0 * m01 * e.x * e.y + m11 * e.y * e.y;
    vec2 d = P.seg;
    float a = m00 * d.x * d.x + 2.0 * m01 * d.x * d.y + m11 * d.y * d.y;
    if (!(a > 1e-6)) return P.amp * exp(-0.5 * c);
    float b = m00 * d.x * e.x + m01 * (d.x * e.y + d.y * e.x) + m11 * d.y * e.y;
    float sa = sqrt(a);
    float u0 = b / a;
    // an open end continues into the neighbouring bin: no Gaussian cap there
    float phiEnd = P.openEnd ? 1.0 : 0.5 * (1.0 + dust_erf((1.0 - u0) * sa * 0.70710678));
    float phiStart = P.openStart ? 0.0 : 0.5 * (1.0 + dust_erf(-u0 * sa * 0.70710678));
    float span = phiEnd - phiStart;
    float q = max(c - b * b / a, 0.0);
    // a linear density along the segment (dust::capsule_tau): the integral of u exp(-a(u-u0)^2/2)
    // over it is u0 I0 + (E_start - E_end)/a with the Gaussian's values at the closed ends
    float i0 = sqrt(2.0 * DUST_PI_F32 / a) * span;
    float weight;
    if (P.rho.y == P.rho.x) {
        weight = P.rho.x * i0;
    } else {
        float eStart = P.openStart ? 0.0 : exp(-0.5 * a * u0 * u0);
        float eEnd = P.openEnd ? 0.0 : exp(-0.5 * a * (1.0 - u0) * (1.0 - u0));
        float i1 = u0 * i0 + (eStart - eEnd) / a;
        weight = max(P.rho.x * i0 + (P.rho.y - P.rho.x) * i1, 0.0);
    }
    return P.amp * exp(-0.5 * q) * weight;
}

// tr(Sigma_v)/3 of a cluster's moments
float dust_mean_var(DustMoments M) {
    return (M.cov_a.x + M.cov_a.w + M.cov_b_age_id.y) / 3.0;
}
// sigma_pred / sigma_self of the velocity dispersion (1 without a wider predecessor)
// (dust::width_ratio)
const float DUST_WIDTH_RATIO_MAX = 1024.0;  // dust::WIDTH_RATIO_MAX
float dust_width_ratio(DustMoments M, bool hasPred, DustMoments P) {
    float own = dust_mean_var(M);
    if (!hasPred || !(dust_mean_var(P) > max(own, 0.0))) return 1.0;
    if (!(own > 0.0)) return DUST_WIDTH_RATIO_MAX;
    return min(sqrt(dust_mean_var(P) / own), DUST_WIDTH_RATIO_MAX);
}
// pieces a capsule is cut into: clamp(ceil(ratio) - 1, 1, DUST_TIME_PIECES_MAX) (dust::time_pieces)
uint dust_time_pieces(float ratio) {
    uint c = uint(max(ceil(ratio), 1.0));
    return clamp(c - 1u, 1u, DUST_TIME_PIECES_MAX);
}
// Packet of size bin b of a cluster (dust::subpackets): uniform along the bin's chord (a Gaussian
// of DUST_CHORD_VAR_FACTOR half-chords^2 along it), Sigma_v scaled by the bin's speed factor,
// 1/DUST_SIZE_BINS of the flux
void dust_bin_packet(DustMoments M, uint b, out vec3 mean, out float S[6], out float flux) {
    vec3 a = M.edges[b].xyz, c = M.edges[b + 1u].xyz;
    mean = 0.5 * (a + c);
    vec3 h = 0.5 * (c - a);
    float f = 0.5 * (dust_edge_factor(M, b) + dust_edge_factor(M, b + 1u));
    float f2 = f * f, k = DUST_CHORD_VAR_FACTOR;
    S[0] = f2 * M.cov_a.x + k * h.x * h.x;
    S[1] = f2 * M.cov_a.y + k * h.x * h.y;
    S[2] = f2 * M.cov_a.z + k * h.x * h.z;
    S[3] = f2 * M.cov_a.w + k * h.y * h.y;
    S[4] = f2 * M.cov_b_age_id.x + k * h.y * h.z;
    S[5] = f2 * M.cov_b_age_id.y + k * h.z * h.z;
    flux = M.mean_flux.w / float(DUST_SIZE_BINS);
}

// The packet drawn for size bin b (dust::bin_chord_packet): uniform along its chord (mean = the
// start edge, seg = the chord), Sigma_v scaled by the bin's speed factor plus the bin's time
// segment segT as a Gaussian of DUST_CHORD_VAR_FACTOR half-segments^2, 1/DUST_SIZE_BINS of the flux
void dust_bin_chord(DustMoments M, uint b, vec3 segT, out vec3 mean, out vec3 chord, out float S[6], out float flux) {
    vec3 a = M.edges[b].xyz, c = M.edges[b + 1u].xyz;
    mean = a;
    chord = c - a;
    vec3 h = 0.5 * segT;
    float f = 0.5 * (dust_edge_factor(M, b) + dust_edge_factor(M, b + 1u));
    float f2 = f * f, k = DUST_CHORD_VAR_FACTOR;
    S[0] = f2 * M.cov_a.x + k * h.x * h.x;
    S[1] = f2 * M.cov_a.y + k * h.x * h.y;
    S[2] = f2 * M.cov_a.z + k * h.x * h.z;
    S[3] = f2 * M.cov_a.w + k * h.y * h.y;
    S[4] = f2 * M.cov_b_age_id.x + k * h.y * h.z;
    S[5] = f2 * M.cov_b_age_id.y + k * h.z * h.z;
    flux = M.mean_flux.w / float(DUST_SIZE_BINS);
}

bool dust_project_point(vec3 p, mat4 mvp, float W, float H, out vec2 px);
// sweep count of bin b (dust::time_sweep): the larger screen displacement of its two edges to
// the predecessor's per DUST_TIME_SWEEP_STEP_PX, at most DUST_TIME_SWEEP_MAX
uint dust_time_sweep(DustMoments M, DustMoments P, uint b, mat4 mvp, float W, float H) {
    float gap = 0.0;
    for (uint e = b; e <= b + 1u; ++e) {
        vec2 a, c;
        if (dust_project_point(M.edges[e].xyz, mvp, W, H, a) && dust_project_point(P.edges[e].xyz, mvp, W, H, c))
            gap = max(gap, length(c - a));
    }
    if (isinf(gap) || isnan(gap)) return 1u;
    return clamp(uint(ceil(gap / DUST_TIME_SWEEP_STEP_PX)), 1u, DUST_TIME_SWEEP_MAX);
}
// the k-th of nS sweep sub-capsules of bin b (dust::bin_sweep_packet): the chord interpolated at
// u = (k + 1/2)/nS between the bin's own edges and the predecessor's, Sigma_v scaled by the
// bin's speed factor plus the residual Gaussian of the larger edge displacement / 2 nS along it,
// 1/(DUST_SIZE_BINS nS) of the flux
void dust_bin_sweep(DustMoments M, DustMoments P, uint b, uint k, uint nS, out vec3 mean, out vec3 chord, out float S[6], out float flux) {
    vec3 a = M.edges[b].xyz, c = M.edges[b + 1u].xyz;
    vec3 da = P.edges[b].xyz - a, dc = P.edges[b + 1u].xyz - c;
    vec3 dmax = dot(da, da) > dot(dc, dc) ? da : dc;
    float u = (float(k) + 0.5) / float(nS);
    vec3 ak = a + da * u, ck = c + dc * u;
    float f = 0.5 * (dust_edge_factor(M, b) + dust_edge_factor(M, b + 1u));
    float f2 = f * f;
    float kv = DUST_CHORD_VAR_FACTOR;
    vec3 h = dmax * (0.5 / float(nS));
    mean = ak;
    chord = ck - ak;
    S[0] = f2 * M.cov_a.x + kv * h.x * h.x;
    S[1] = f2 * M.cov_a.y + kv * h.x * h.y;
    S[2] = f2 * M.cov_a.z + kv * h.x * h.z;
    S[3] = f2 * M.cov_a.w + kv * h.y * h.y;
    S[4] = f2 * M.cov_b_age_id.x + kv * h.y * h.z;
    S[5] = f2 * M.cov_b_age_id.y + kv * h.z * h.z;
    flux = M.mean_flux.w / float(DUST_SIZE_BINS * nS);
}
// Rodrigues rotation of v by angle about the unit axis
vec3 dust_rotate3(vec3 v, vec3 axis, float angle) {
    float s = sin(angle), c = cos(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}
// the spiral of a rigidly rotating emitter (dust::spiral_point): the two samples (ages ageO,
// ageP) un-spun by R(+omega age) are linear in age, interpolated, and re-spun by R(-omega age(u));
// the straight segment when omega = 0
vec3 dust_spiral_point(vec3 own, vec3 pred, float ageO, float ageP, vec3 axis, float omega, float u) {
    vec3 lO = dust_rotate3(own, axis, omega * ageO);
    vec3 lP = dust_rotate3(pred, axis, omega * ageP);
    vec3 l = lO + (lP - lO) * u;
    return dust_rotate3(l, axis, -omega * (ageO + u * (ageP - ageO)));
}
// the spiral gate of a sample with a predecessor (dust::spiral_of)
bool dust_spiral_of(DustMoments M, DustMoments P, out vec3 axis, out float omega) {
    omega = dust_spin_rate(M);
    axis = dust_spin_axis(M);
    float dAge = max(P.cov_b_age_id.z - M.cov_b_age_id.z, 0.0);
    return omega * dAge > DUST_SPIRAL_MIN_TURN && M.cov_b_age_id.z <= DUST_SPIRAL_MAX_AGE_SAMPLES * dAge;
}

// The whole polyline as one packet (dust::pooled_packet): the bins' mean, their mean covariance
// plus the scatter of their midpoints, the whole flux
void dust_pooled_packet(DustMoments M, out vec3 mean, out float S[6], out float flux) {
    float n = float(DUST_SIZE_BINS);
    mean = vec3(0.0);
    for (uint b = 0u; b < DUST_SIZE_BINS; ++b) mean += 0.5 * (M.edges[b].xyz + M.edges[b + 1u].xyz) / n;
    for (int k = 0; k < 6; ++k) S[k] = 0.0;
    for (uint b = 0u; b < DUST_SIZE_BINS; ++b) {
        vec3 mb; float Sb[6]; float fb;
        dust_bin_packet(M, b, mb, Sb, fb);
        vec3 d = mb - mean;
        S[0] += (Sb[0] + d.x * d.x) / n;
        S[1] += (Sb[1] + d.x * d.y) / n;
        S[2] += (Sb[2] + d.x * d.z) / n;
        S[3] += (Sb[3] + d.y * d.y) / n;
        S[4] += (Sb[4] + d.y * d.z) / n;
        S[5] += (Sb[5] + d.z * d.z) / n;
    }
    flux = M.mean_flux.w;
}
// the mean covariance of the bins alone, without the scatter of their midpoints
// (dust::pooled_within_cov): the width a predecessor hands to the pieces of a point sample
void dust_pooled_within(DustMoments M, out float S[6]) {
    float n = float(DUST_SIZE_BINS);
    for (int k = 0; k < 6; ++k) S[k] = 0.0;
    for (uint b = 0u; b < DUST_SIZE_BINS; ++b) {
        vec3 mb; float Sb[6]; float fb;
        dust_bin_packet(M, b, mb, Sb, fb);
        for (int k = 0; k < 6; ++k) S[k] += Sb[k] / n;
    }
}
// The texel band of a screen packet (dust::CapsuleBand): the segment's lateral Gaussian band
// (DUST_SPLAT_SIGMAS across) between its ends, a Gaussian cap at a closed end, nothing beyond an
// open one; a point packet is its bounding box
struct DustBand {
    vec2  mu, d, n;
    float dd, w, uLo, uHi;
    bool  point;
    vec2  ext;   // point packet: bounding box half-extents
};
DustBand dust_band_new(DustScreenPacket P) {
    DustBand B;
    B.mu = P.mu; B.d = P.seg;
    B.dd = dot(P.seg, P.seg);
    B.point = !(B.dd > 1e-12);
    B.ext = DUST_SPLAT_SIGMAS * sqrt(max(vec2(P.cov.x, P.cov.z), vec2(0.0)));
    B.n = vec2(0.0); B.w = 0.0; B.uLo = 0.0; B.uHi = 0.0;
    if (B.point) return B;
    float len = sqrt(B.dd);
    vec2 t = P.seg / len;
    B.n = vec2(-t.y, t.x);
    float qt = P.cov.x * t.x * t.x + 2.0 * P.cov.y * t.x * t.y + P.cov.z * t.y * t.y;
    float qn = P.cov.x * B.n.x * B.n.x + 2.0 * P.cov.y * B.n.x * B.n.y + P.cov.z * B.n.y * B.n.y;
    float cap = DUST_SPLAT_SIGMAS * sqrt(max(qt, 0.0)) / len;
    B.w = DUST_SPLAT_SIGMAS * sqrt(max(qn, 0.0));
    B.uLo = P.openStart ? 0.0 : -cap;
    B.uHi = P.openEnd ? 1.0 : 1.0 + cap;
    return B;
}
// texel rows [y0, y1] at texel size s; false when empty
bool dust_band_rows(DustBand B, float s, uint h, out int y0, out int y1) {
    float yLo, yHi;
    if (B.point) { yLo = B.mu.y - B.ext.y; yHi = B.mu.y + B.ext.y; }
    else {
        yLo = 1e30; yHi = -1e30;
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
            float u = i == 0 ? B.uLo : B.uHi, v = j == 0 ? -B.w : B.w;
            float y = B.mu.y + u * B.d.y + v * B.n.y;
            yLo = min(yLo, y); yHi = max(yHi, y);
        }
    }
    y0 = int(max(floor(yLo / s), 0.0));
    y1 = min(int(ceil(yHi / s)), int(h) - 1);
    return y0 <= y1 && yLo <= yHi;
}
// texel columns [x0, x1] on the row at yc px; false when the row is outside the band. Across the
// segment the band is a Gaussian tail (a superset of texels is fine); along it the open chords of
// a polyline tile the line, so exactly the texel centres with uLo <= u < uHi are visited
// (dust::CapsuleBand::columns)
bool dust_band_columns(DustBand B, float yc, float s, uint w, out int x0, out int x1) {
    if (B.point) {
        x0 = int(floor((B.mu.x - B.ext.x) / s));
        x1 = int(ceil((B.mu.x + B.ext.x) / s));
    } else {
        float ry = yc - B.mu.y;
        if (abs(B.n.x) > 1e-6) {
            float a = B.mu.x + (-B.w - ry * B.n.y) / B.n.x;
            float b = B.mu.x + (B.w - ry * B.n.y) / B.n.x;
            x0 = int(floor(min(a, b) / s)); x1 = int(ceil(max(a, b) / s));
        } else if (abs(ry * B.n.y) > B.w) return false;
        else { x0 = 0; x1 = int(w) - 1; }
        if (abs(B.d.x) > 1e-6) {
            float a = B.mu.x + (B.uLo * B.dd - ry * B.d.y) / B.d.x;
            float b = B.mu.x + (B.uHi * B.dd - ry * B.d.y) / B.d.x;
            int c0 = int(ceil(min(a, b) / s - 0.5));
            int c1 = int(ceil(max(a, b) / s - 0.5)) - 1;
            x0 = max(x0, c0); x1 = min(x1, c1);
        } else {
            float u = ry * B.d.y / B.dd;
            if (u < B.uLo || u >= B.uHi) return false;
        }
    }
    x0 = max(x0, 0);
    x1 = min(x1, int(w) - 1);
    return x0 <= x1;
}

// Pixel position of a local point (w <= 0: behind the camera, returns false)
bool dust_project_point(vec3 p, mat4 mvp, float W, float H, out vec2 px) {
    vec4 clip = mvp * vec4(p, 1.0);
    if (!(clip.w > 0.0)) return false;
    px = (clip.xy / clip.w * 0.5 + 0.5) * vec2(W, H);
    return true;
}
// Whether the size polyline is below DUST_MERGE_PX on screen (dust::polyline_merged): the two end
// edges and the middle one all within it of the mean
bool dust_polyline_merged(DustMoments M, mat4 mvp, float W, float H) {
    vec2 c, p;
    if (!dust_project_point(M.mean_flux.xyz, mvp, W, H, c)) return false;
    uint idx[3] = uint[3](0u, DUST_SIZE_BINS / 2u, DUST_SIZE_BINS);
    for (int k = 0; k < 3; ++k) {
        if (!dust_project_point(M.edges[idx[k]].xyz, mvp, W, H, p)) return false;
        vec2 d = p - c;
        if (!(dot(d, d) < DUST_MERGE_PX * DUST_MERGE_PX)) return false;
    }
    return true;
}

// ─── Kepler (universal variables, Laguerre-Conway). Mirror of `dust::kepler` ─

// f32 Stumpff c0..c3 (phase 1 only)
vec4 dust_stumpff_f32(float x) {
    float xr = x;
    uint n = 0u;
    while (abs(xr) > 0.1 && n < 64u) { xr *= 0.25; n++; }
    float c2 = DF_INV_FACT[2].x - xr * (DF_INV_FACT[4].x - xr * (DF_INV_FACT[6].x - xr * (DF_INV_FACT[8].x - xr * DF_INV_FACT[10].x)));
    float c3 = DF_INV_FACT[3].x - xr * (DF_INV_FACT[5].x - xr * (DF_INV_FACT[7].x - xr * (DF_INV_FACT[9].x - xr * DF_INV_FACT[11].x)));
    vec4 c = vec4(1.0 - xr * c2, 1.0 - xr * c3, c2, c3);
    for (uint i = 0u; i < n; i++) {
        c = vec4(2.0 * c.x * c.x - 1.0, c.x * c.y, 0.5 * c.y * c.y, 0.25 * (c.w + c.y * c.z));
    }
    return c;
}

// df64 Stumpff c0..c3, series up to x^6 after reduction to |x| <= 0.1
void dust_stumpff_df(vec2 x, out vec2 c0, out vec2 c1, out vec2 c2, out vec2 c3) {
    vec2 xr = x;
    uint n = 0u;
    while (abs(xr.x) > 0.1 && n < 64u) { xr = df_scale_pow2(xr, 0.25); n++; }
    c2 = df_sub(DF_INV_FACT[12], df_mul(xr, DF_INV_FACT[14]));
    c2 = df_sub(DF_INV_FACT[10], df_mul(xr, c2));
    c2 = df_sub(DF_INV_FACT[8], df_mul(xr, c2));
    c2 = df_sub(DF_INV_FACT[6], df_mul(xr, c2));
    c2 = df_sub(DF_INV_FACT[4], df_mul(xr, c2));
    c2 = df_sub(DF_INV_FACT[2], df_mul(xr, c2));
    c3 = df_sub(DF_INV_FACT[13], df_mul(xr, DF_INV_FACT[15]));
    c3 = df_sub(DF_INV_FACT[11], df_mul(xr, c3));
    c3 = df_sub(DF_INV_FACT[9], df_mul(xr, c3));
    c3 = df_sub(DF_INV_FACT[7], df_mul(xr, c3));
    c3 = df_sub(DF_INV_FACT[5], df_mul(xr, c3));
    c3 = df_sub(DF_INV_FACT[3], df_mul(xr, c3));
    c0 = df_sub(vec2(1.0, 0.0), df_mul(xr, c2));
    c1 = df_sub(vec2(1.0, 0.0), df_mul(xr, c3));
    for (uint i = 0u; i < n; i++) {
        vec2 n0 = df_add_f(df_scale_pow2(df_mul(c0, c0), 2.0), -1.0);
        vec2 n1 = df_mul(c0, c1);
        vec2 n2 = df_scale_pow2(df_mul(c1, c1), 0.5);
        vec2 n3 = df_scale_pow2(df_add(c3, df_mul(c1, c2)), 0.25);
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
}

// Laguerre-Conway step (n = 5); false when the denominator vanishes
bool dust_laguerre(float f, float fp, float fpp, out float ds) {
    float disc = sqrt(abs(16.0 * fp * fp - 20.0 * f * fpp));
    float denom = fp >= 0.0 ? fp + disc : fp - disc;
    ds = 0.0;
    if (denom == 0.0) return false;
    ds = 5.0 * f / denom;
    return true;
}

// propagates (r0, v0) by dt under mu (any sign): f32 Laguerre, then df64 polishing
void dust_kepler(Df3 r0, Df3 v0, vec2 mu, vec2 dt, out Df3 r, out Df3 v) {
    vec2 r0n = df_sqrt(df3_dot(r0, r0));
    if (!(r0n.x > 0.0) || dt.x == 0.0) { r = r0; v = v0; return; }
    vec2 v2 = df3_dot(v0, v0);
    vec2 eta = df3_dot(r0, v0);
    vec2 alpha = df_sub(df_div(df_scale_pow2(mu, 2.0), r0n), v2); // = mu / a
    vec2 zeta = df_sub(mu, df_mul(alpha, r0n));

    // bound elliptic arguments: remove whole periods
    vec2 t = dt;
    if (mu.x > 0.0 && alpha.x > 0.0) {
        vec2 period = df_div(df_mul(DF_TWO_PI, mu), df_mul(alpha, df_sqrt(alpha)));
        vec2 k = df_div(t, period);
        k = k.x >= 0.0 ? df_floor(k) : -df_floor(-k);
        t = df_sub(t, df_mul(k, period));
    }

    // phase 1: f32 Laguerre from s = t / r0
    float r0f = r0n.x, etaf = eta.x, zetaf = zeta.x, alphaf = alpha.x;
    float tf = t.x + t.y;
    float s = tf / r0f;
    for (uint it = 0u; it < DUST_KEPLER_ITERS_F32; it++) {
        vec4 c = dust_stumpff_f32(alphaf * s * s);
        float s2 = s * s;
        float f   = r0f * s + etaf * s2 * c.z + zetaf * s2 * s * c.w - tf;
        float fp  = r0f + etaf * s * c.y + zetaf * s2 * c.z;
        float fpp = etaf * c.x + zetaf * s * c.y;
        float ds;
        if (!dust_laguerre(f, fp, fpp, ds)) break;
        s -= ds;
        if (!(abs(ds) > DUST_KEPLER_TOL_F32 * abs(s))) break;
    }

    // phase 2: df64 residual, f32 correction
    vec2 sd = vec2(s, 0.0);
    vec2 c0, c1, c2, c3;
    for (uint it = 0u; it < DUST_KEPLER_ITERS_DF; it++) {
        dust_stumpff_df(df_mul(df_mul(alpha, sd), sd), c0, c1, c2, c3);
        vec2 s2 = df_mul(sd, sd);
        vec2 f = df_sub(df_add(df_add(df_mul(r0n, sd), df_mul(df_mul(eta, s2), c2)), df_mul(df_mul(df_mul(zeta, s2), sd), c3)), t);
        vec2 fp = df_add(df_add(r0n, df_mul(df_mul(eta, sd), c1)), df_mul(df_mul(zeta, s2), c2));
        float fpp = etaf * c0.x + zetaf * sd.x * c1.x;
        float ds;
        if (!dust_laguerre(f.x, fp.x, fpp, ds)) break;
        sd = df_add_f(sd, -ds);
        if (!(abs(ds) > DUST_KEPLER_TOL_DF * abs(sd.x))) break;
    }

    dust_stumpff_df(df_mul(df_mul(alpha, sd), sd), c0, c1, c2, c3);
    vec2 s2 = df_mul(sd, sd);
    vec2 g1 = df_mul(sd, c1);
    vec2 g2 = df_mul(s2, c2);
    vec2 g3 = df_mul(df_mul(s2, sd), c3);
    vec2 rn = df_add(df_add(r0n, df_mul(eta, g1)), df_mul(zeta, g2));
    vec2 ff = df_sub(vec2(1.0, 0.0), df_div(df_mul(mu, g2), r0n));
    vec2 gg = df_sub(t, df_mul(mu, g3));
    vec2 fd = -df_div(df_mul(mu, g1), df_mul(rn, r0n));
    vec2 gd = df_sub(vec2(1.0, 0.0), df_div(df_mul(mu, g2), rn));
    r = df3_add(df3_scale(r0, ff), df3_scale(v0, gg));
    v = df3_add(df3_scale(r0, fd), df3_scale(v0, gd));
}

#endif // DUST_COMMON_GLSL
