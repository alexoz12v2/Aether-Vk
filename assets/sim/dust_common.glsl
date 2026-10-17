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
// render cluster y word: ring slot | log2 capacity << 22 | stream shift << 27 | live << 31
// (dust::render_word)
const uint  DUST_RENDER_SLOT_MASK    = (1u << 22u) - 1u;
const uint  DUST_RENDER_LOG2CAP_BIT0 = 22u;
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
const uint  DUST_TIME_PIECES_MAX      = 8u;
const uint  DUST_TIME_SWEEP_MAX       = 4u;    // dust::TIME_SWEEP_MAX
const float DUST_LIT_ARC_END_EPS      = 1e-5;  // dust::LIT_ARC_END_EPS
const float DUST_SPIRAL_MIN_TURN      = 0.01;  // dust::SPIRAL_MIN_TURN (rad)
const float DUST_SPIRAL_MAX_AGE_SAMPLES = 8.0; // dust::SPIRAL_MAX_AGE_SAMPLES
const float DUST_TIME_SWEEP_STEP_PX   = 1.0;   // dust::TIME_SWEEP_STEP_PX
// the cluster level of detail (dust::PYR_LOD, dust::group_bins / lod_px_active): the header word
// (f32 bits, pixels per drawn segment, 0 = off, host-written), the size bins one bin of the loop
// spans (1 outside a grouped cluster: every function reading edges b and b + 1 reads b + dust_group)
// and the frame's knob (the sweep step); set per cluster by dust_splat.comp
const uint  DUST_PYR_LOD              = 61u;
uint  dust_group  = 1u;
float dust_lod_px = 0.0;
const float DUST_MERGE_PX             = 1.0;    // dust::DUST_MERGE_PX
const float DUST_DET_ANISO_FLOOR      = 1e-6;   // dust::DUST_DET_ANISO_FLOOR
const float DUST_COUNT_MAX_PER_ADD    = 16777216.0;
const float DUST_SPLAT_MAX_TEXELS     = 16384.0;
const uint  DUST_PYR_SPLIT_CHILDREN   = 11u;   // children made by the perspective split (atomicAdd; dust::PYR_SPLIT_CHILDREN)
const uint  DUST_PYR_VOLUME_EVALS    = 245u;  // atomicAdd: ray-integral evaluations of the volumetric packets (dust::PYR_VOLUME_EVALS)
const uint  DUST_PYR_EWA_TEXELS        = 63u;   // atomicAdd: floor(band / DUST_EWA_TEXEL_UNIT) + 1 per EWA packet (dust::PYR_EWA_TEXELS)
const float DUST_EWA_TEXEL_UNIT        = 64.0;  // dust::DUST_EWA_TEXEL_UNIT
const uint  DUST_EWA_FRAME_BUDGET     = 268435456u; // dust::DUST_EWA_FRAME_BUDGET (1 << 28 texels): the host's target for the EWA bias
const uint  DUST_EWA_BIAS_MAX         = 6u;     // dust::DUST_EWA_BIAS_MAX
const uint  DUST_FRAME_HARD_FACTOR    = 4u;     // dust::DUST_FRAME_HARD_FACTOR: the in-frame gates at this many budgets
const uint  DUST_PYR_SPLIT_DROPPED_FLUX = 246u; // f32 bits atomicMax: largest flux dropped behind the eye (dust::PYR_SPLIT_DROPPED_FLUX)
const uint  DUST_PYR_SPLIT_DEPTH      = 247u;  // host-written: bits 0-7 the split depth limit, bits 8-15 the volume level bias, 16-17 the volume axis, 18-21 the EWA bias (dust::PYR_SPLIT_DEPTH)
const float DUST_SPLIT_EPS            = 0.1;    // dust::DUST_SPLIT_EPS
const float DUST_SPLIT_A              = 0.8;    // dust::DUST_SPLIT_A
const uint  DUST_SPLIT_DEPTH_MAX      = 4u;     // dust::DUST_SPLIT_DEPTH_MAX
const uint  DUST_SPLIT_BUDGET         = 1048576u; // dust::DUST_SPLIT_BUDGET (1 << 20): the host's target; the in-frame gate at DUST_FRAME_HARD_FACTOR of it
const float DUST_SPLIT_CLIP_EPS       = 1e-3;   // dust::DUST_SPLIT_CLIP_EPS
const float DUST_VOLUME_RATIO         = 0.2;    // dust::DUST_VOLUME_RATIO
const float DUST_VOLUME_SIGMAS        = 3.0;    // dust::DUST_VOLUME_SIGMAS
const float DUST_VOLUME_TEXELS_PER_SIGMA = 4.0; // dust::DUST_VOLUME_TEXELS_PER_SIGMA
const uint  DUST_VOLUME_MIN_LEVEL     = 3u;     // dust::DUST_VOLUME_MIN_LEVEL
const uint  DUST_VOLUME_PIECES_MAX    = 16u;    // dust::DUST_VOLUME_PIECES_MAX
const float DUST_VOLUME_PIECE_SIGMAS  = 2.0;    // dust::DUST_VOLUME_PIECE_SIGMAS
const float DUST_VOLUME_PIECE_VAR_CAPPED = 0.3025; // dust::DUST_VOLUME_PIECE_VAR_CAPPED
const uint  DUST_VOLUME_AXIS_CHORD    = 0u;     // dust::DUST_VOLUME_AXIS_CHORD (bits 16-17 of DUST_PYR_SPLIT_DEPTH)
const uint  DUST_VOLUME_AXIS_TIME     = 1u;     // dust::DUST_VOLUME_AXIS_TIME
const uint  DUST_VOLUME_AXIS_GRID     = 2u;     // dust::DUST_VOLUME_AXIS_GRID
const uint  DUST_VOLUME_MAX_EVALS     = 4096u;  // dust::DUST_VOLUME_MAX_EVALS: texels x pieces per volumetric packet
const uint  DUST_VOLUME_FRAME_BUDGET  = 67108864u; // dust::DUST_VOLUME_FRAME_BUDGET (1 << 26): the in-frame gate at DUST_FRAME_HARD_FACTOR of it
// K1c, the texel's integral (dust::volume_class and its constants)
const uint  DUST_VOLUME_COARSE_LEVELS = 2u;     // dust::DUST_VOLUME_COARSE_LEVELS
const float DUST_VOLUME_SMALL_TEXELS  = 2.0;    // dust::DUST_VOLUME_SMALL_TEXELS
const uint  DUST_VOLUME_THIN_LEVELS   = 6u;     // dust::DUST_VOLUME_THIN_LEVELS
const uint  DUST_VOLUME_LINE_LEVELS   = 5u;     // dust::DUST_VOLUME_LINE_LEVELS
const uint  DUST_VOLUME_SUBRAY_ORDER_MAX = 4u;  // dust::DUST_VOLUME_SUBRAY_ORDER_MAX
const uint  DUST_VOLUME_SUBRAY_ORDER_TOP = 3u;  // dust::DUST_VOLUME_SUBRAY_ORDER_TOP
const uint  DUST_VOLUME_SUBRAY_ORDER_LOW = 2u;  // dust::DUST_VOLUME_SUBRAY_ORDER_LOW
const float DUST_SMALL_COVER          = 4.0;    // dust::DUST_SMALL_COVER
const float DUST_GH5_N1 = 1.3556262;  // dust::DUST_GH5_N1 (5-point Gauss-Hermite, probabilists')
const float DUST_GH5_N2 = 2.8569700;  // dust::DUST_GH5_N2
const float DUST_GH5_W0 = 0.5333333;  // dust::DUST_GH5_W0
const float DUST_GH5_W1 = 0.2220759;  // dust::DUST_GH5_W1
const float DUST_GH5_W2 = 0.0112574;  // dust::DUST_GH5_W2
const float DUST_SMALL_DEPTH5_EPS = 0.1; // dust::DUST_SMALL_DEPTH5_EPS
const uint  DUST_VOLUME_CLASS_CENTRE  = 0u;     // dust::DUST_VOLUME_CLASS_CENTRE
const uint  DUST_VOLUME_CLASS_SMALL   = 1u;     // dust::DUST_VOLUME_CLASS_SMALL
const uint  DUST_VOLUME_CLASS_SUBRAY  = 2u;     // dust::DUST_VOLUME_CLASS_SUBRAY
const uint  DUST_VOLUME_CLASS_LINE    = 3u;     // dust::DUST_VOLUME_CLASS_LINE
const float DUST_GH3_NODE             = 1.7320508; // dust::DUST_GH3_NODE (sqrt 3)
const float DUST_UT2_NODE             = 1.41421356; // dust::DUST_UT2_NODE (sqrt 2)
const float DUST_INV_2PI              = 0.15915494; // dust::DUST_INV_2PI
const float DUST_SMALL_RHO_MAX        = 0.9;    // dust::DUST_SMALL_RHO_MAX
const float DUST_SQRT_2PI             = 2.5066283; // dust::DUST_SQRT_2PI
const float DUST_SQRT_2               = 1.41421356; // core::f32::consts::SQRT_2
const float DUST_FRAC_1_SQRT_2        = 0.70710678; // core::f32::consts::FRAC_1_SQRT_2
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
layout(buffer_reference, std430, buffer_reference_align = 16) buffer DustMomentsBuffer {
    DustMoments m[];
};
// the moments of the pass (set by the shader's main): the functions below read a cluster's
// moments by index through it (dust_mom.m[mi]), never a by-value copy of its 80 floats
DustMomentsBuffer dust_mom;
// speed factor sqrt(beta_j/beta) of edge j (dust::DustMoments::edge_factor)
float dust_edge_factor(uint mi, uint j) {
    float lo = dust_mom.m[mi].edges[0].w, hi = dust_mom.m[mi].edges[DUST_SIZE_BINS].w;
    return lo + (hi - lo) * (float(j) / float(DUST_SIZE_BINS));
}
float dust_spin_rate(uint mi) { return dust_mom.m[mi].edges[1].w; }
vec3 dust_spin_axis(uint mi) { return vec3(dust_mom.m[mi].edges[2].w, dust_mom.m[mi].edges[3].w, dust_mom.m[mi].edges[4].w); }
float dust_helio_au(uint mi) { return dust_mom.m[mi].edges[5].w; }
vec3 dust_sun_dir(uint mi) { return vec3(dust_mom.m[mi].edges[6].w, dust_mom.m[mi].edges[7].w, dust_mom.m[mi].edges[8].w); }
// the speck pass's walk mask (dust::DustMoments::specks_mask): bit b = size bin b is walked for
// specks this frame, so the continuum gives up its shell mass; edges[15].w as bits (0 from the
// propagate), written by dust_specks.comp before the splat (atomicOr on the word)
uint dust_specks_mask(uint mi) { return floatBitsToUint(dust_mom.m[mi].edges[15].w); }
const uint  DUST_MOMENTS_WORDS           = 80u;  // uints per DustMoments
const uint  DUST_MOMENTS_SPECKS_MASK_WORD = 12u + 15u * 4u + 3u;  // edges[15].w
const float DUST_SPECKS_WALK_MIN         = 0.25; // dust::specks::DUST_SPECKS_WALK_MIN

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
uint dust_render_word(uint slot, uint shift, uint log2cap) {
    return (slot & DUST_RENDER_SLOT_MASK) | ((log2cap & 0x1Fu) << DUST_RENDER_LOG2CAP_BIT0)
        | ((shift & 0xFu) << DUST_RENDER_SHIFT_BIT0) | DUST_RENDER_LIVE_BIT;
}
bool dust_render_live(float y) { return (floatBitsToUint(y) & DUST_RENDER_LIVE_BIT) != 0u; }
uint dust_render_shift(float y) { return (floatBitsToUint(y) >> DUST_RENDER_SHIFT_BIT0) & 0xFu; }
uint dust_render_slot(float y) { return floatBitsToUint(y) & DUST_RENDER_SLOT_MASK; }
uint dust_render_log2_capacity(float y) { return (floatBitsToUint(y) >> DUST_RENDER_LOG2CAP_BIT0) & 0x1Fu; }
// the tier's ring mask (capacity - 1; dust::render_mask)
uint dust_render_mask(float y) {
    uint l = dust_render_log2_capacity(y);
    return l >= 32u ? 0xFFFFFFFFu : (1u << l) - 1u;
}
bool dust_stream_break(float dbetaField) { return (floatBitsToUint(dbetaField) & DUST_STREAM_BREAK_BIT) != 0u; }

// Stream predecessor of the render cluster in ring slot `slot` (the buffer is slot indexed): the
// slot (slot - S) & mask of the same stream's previous, older time sample, -1 when the cluster's
// ring offset k = (slot - firstSlot) & mask is below S or not below liveCount (so a retired,
// reused slot is never a predecessor: k - S is in the live range), after a stream break, or
// outside the tier's age band (dust::streak_pred_slot). Reads only the y / z words, which the
// LOD never rewrites (it rewrites w concurrently).
int dust_streak_pred(DustRenderBuffer render, uint slot, uint firstSlot, uint liveCount) {
    vec2 yz = render.c[slot].age_id_dbeta_flux.yz;
    uint mask = dust_render_mask(yz.x);
    uint n = 1u << dust_render_shift(yz.x);
    uint k = (slot - firstSlot) & mask;
    if (k < n || k >= liveCount || dust_stream_break(yz.y)) return -1;
    uint p = (slot - n) & mask;
    return dust_render_live(render.c[p].age_id_dbeta_flux.y) ? int(p) : -1;
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

// The dither segments (dust::DUST_SEGMENT_TEXELS, dust::segment_carry): a packet's texels are
// drawn in row segments of 16 columns, each starting its carry at a hashed phase in [-1/2, 1/2),
// so a segment's counts do not depend on the thread drawing it. The seeds: per cluster (its
// compact index, set by the splat's main) and per packet in drawing order (dust::next_packet_seed)
const uint DUST_SEGMENT_TEXELS = 16u;
uint dust_cluster_seed = 0u;
uint dust_packet_ord = 0u;
// the counting pass of dust_splat.comp (a lane per sub-packet): the walk up to every seed draw
// without drawing, touching the pyramid or its counters; dust_seed_count the seeds it would draw
bool dust_counting = false;
uint dust_seed_count = 0u;
uint dust_next_packet_seed() {
    uint s = dust_hash32(dust_cluster_seed ^ (dust_packet_ord * 0x9E3779B9u));
    dust_packet_ord += 1u;
    return s;
}
float dust_segment_carry(uint seed, uint key) {
    return float(dust_hash32(seed ^ dust_hash32(key)) >> 8u) * (1.0 / 16777216.0) - 0.5;
}

// the frame's DUST_PYR_SPLIT_DEPTH word, set by the splat pass (diagnostic bits; 0 elsewhere)
uint dust_pyr_split_word = 0u;
// the frame's deep-chord bit (dust::DUST_FRAME_DEEP_CHORD of DUST_PYR_FLAGS, set by dust_split.comp;
// false elsewhere): a chord deep only by its span at the depth limit drawn by the EWA at
// dust_deep_chord_factors instead of by the ray integral
const uint DUST_FRAME_DEEP_CHORD = 1u << 20u;
bool dust_deep_chord = false;
// the deep chord's Jacobian point u_g = sqrt(w0) / (sqrt(w0) + sqrt(w1)) (the screen segment
// exact) and flux factor k (the perspective's light along the chord); the proof in
// dust::deep_chord_factors (same arithmetic, same order)
vec2 dust_deep_chord_factors(float w0, float w1, vec2 rho) {
    float s0 = sqrt(w0), s1 = sqrt(w1);
    float uG = s0 / (s0 + s1);
    float dw = w1 - w0;
    float ubar = w0 * w1 * (log(w1 / w0) + w0 / w1 - 1.0) / (dw * dw);
    float m = 0.5 * (rho.x + rho.y);
    float k = (m > 0.0 && !isinf(ubar) && !isnan(ubar))
        ? (rho.x + (rho.y - rho.x) * clamp(ubar, 0.0, 1.0)) / m : 1.0;
    return vec2(uG, k);
}
// a + b saturating at 0xFFFFFFFF (no saturating atomics in core GLSL): one atomicAdd, and when
// it wrapped (old + b < old) an atomicMax to the ceiling. Every interleaving ends saturated once
// any add wrapped: an add that lands on a saturated word wraps and re-saturates it, and an add
// that lands between a wrap and its max is overridden by the max; a word no add wrapped is
// exact. Same final state as the compare-and-swap loop this replaces (profile 2026-10-15: the
// loops were ~58 % of the volume texel cost and ~30 % of the queue draw under contention).
// Timing probe (bits 30-31 of DUST_PYR_SPLIT_DEPTH = 1 via dust_pyr_split_word, set by the splat
// only; `AETHERVK_DUST_TIMING_PROBE=plain_add`; diagnostics, never in production): a plain
// atomicAdd, which wraps instead of saturating. What the saturation costs.
void dust_atomic_add_sat(DustUintBuffer buf, uint i, uint b) {
    if (b == 0u) return;
    uint old = atomicAdd(buf.v[i], b);
    if (((dust_pyr_split_word >> 30u) & 3u) == 1u) return;
    if (old + b < old) atomicMax(buf.v[i], 0xFFFFFFFFu);
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
// the packet's physical support (dust::Photometry::support): the half-space n.x >= w (xyz = n,
// zero = none) its cell's grains lie in front of; set per packet by dust_splat.comp
vec4 dust_phot_support = vec4(0.0);
// the packet's cell origin along the support's normal, n.o (dust::Photometry::origin; the apex of
// its cone of grains): valid when dust_phot_has_origin (Rust: not NaN); set with the support
bool dust_phot_has_origin = false;
float dust_phot_origin = 0.0;
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
    // density ∝ 1/f, mean 1 over the chord (dust::bin_density_ends; the reciprocals until
    // 2026-10-16 drew a chord group over a 10x factor range at 3x its flux)
    return vec2(2.0 * f1 / s, 2.0 * f0 / s);
}
// the polyline's slow end bin is drawn as DUST_END_SUBBINS sub-bins (dust::DUST_END_SUBBINS,
// dust::bin_subbins): there the chord's Gaussian and the mid-factor width reach past the slowest grain
const uint DUST_END_SUBBINS = 2u;
// bit 29 of the frame's DUST_PYR_SPLIT_DEPTH word (dust::DUST_FRAME_LEGACY_KERNEL_BIT): the
// kernel before the cone of grains (no dust_conic_piece, no sub-bins), for A/B measurements
bool dust_legacy_kernel() { return ((dust_pyr_split_word >> 29u) & 1u) != 0u; }
uint dust_bin_subbins(uint b) { return (b == 0u && !dust_legacy_kernel()) ? DUST_END_SUBBINS : 1u; }
// sub-bin j of nj of size bin b (dust::sub_edges): the bin's chord and speed-factor range cut into
// nj equal parts, the bin's own edges exactly at its ends
void dust_sub_edges(uint mi, uint b, uint j, uint nj, out vec3 a, out vec3 c, out float fa, out float fc) {
    vec3 e0 = dust_mom.m[mi].edges[b].xyz, e1 = dust_mom.m[mi].edges[b + dust_group].xyz;
    float f0 = dust_edge_factor(mi, b), f1 = dust_edge_factor(mi, b + dust_group);
    if (j == 0u) { a = e0; fa = f0; }
    else { float u = float(j) / float(nj); a = e0 + (e1 - e0) * u; fa = f0 + (f1 - f0) * u; }
    if (j + 1u == nj) { c = e1; fc = f1; }
    else { float u = float(j + 1u) / float(nj); c = e0 + (e1 - e0) * u; fc = f0 + (f1 - f0) * u; }
}
// the share of size bin b's flux in its sub-bin j of nj (dust::sub_share): the bin's 1/f density
// ramp integrated over the sub-bin, over the bin (1 for one sub-bin)
float dust_sub_share(uint mi, uint b, uint j, uint nj) {
    if (nj <= 1u) return 1.0;
    vec2 r = dust_bin_density_ends(dust_edge_factor(mi, b), dust_edge_factor(mi, b + dust_group));
    float u0 = float(j) / float(nj), u1 = float(j + 1u) / float(nj);
    return (u1 - u0) * (r.x + (r.y - r.x) * 0.5 * (u0 + u1)) / (0.5 * (r.x + r.y));
}

// The view direction of `mvp` (dust::view_forward): the depth gradient of the projection, unit;
// zero for a degenerate matrix. Perspective: the w row as it is (w > 0 in front of the camera,
// whatever side of it the scene's origin is on); orthographic: the clip-z row, oriented so that
// the scene's origin lies in front of the eye. The projection and the split use the same direction.
vec3 dust_view_forward(mat4 mvp, vec3 eyeLocal, out bool perspective) {
    vec3 r3 = vec3(mvp[0][3], mvp[1][3], mvp[2][3]);
    vec3 r2 = vec3(mvp[0][2], mvp[1][2], mvp[2][2]);
    perspective = dot(r3, r3) > 0.0;
    vec3 fwdRaw = perspective ? r3 : r2;
    float fn2 = dot(fwdRaw, fwdRaw);
    vec3 fwd = vec3(0.0);
    if (fn2 > 0.0) {
        fwd = fwdRaw / sqrt(fn2);
        if (!perspective && dot(-eyeLocal, fwd) < 0.0) fwd = -fwd;
    }
    return fwd;
}
// f^T S f: the packet's variance along the unit direction f (dust::var_along)
float dust_var_along(float S[6], vec3 f) {
    return S[0] * f.x * f.x + S[3] * f.y * f.y + S[5] * f.z * f.z
         + 2.0 * (S[1] * f.x * f.y + S[2] * f.x * f.z + S[4] * f.y * f.z);
}

// the support constants (dust::DUST_SUPPORT_MASS_MIN / DUST_SUPPORT_CUT_MIN)
const float DUST_SUPPORT_MASS_MIN = 0.05;
const float DUST_SUPPORT_CUT_MIN = 1.0e-3;
bool dust_has_support(vec4 s) { return s.x != 0.0 || s.y != 0.0 || s.z != 0.0; }
// the mass of the 3-sigma ball of a 3-D standard normal (dust::DUST_BALL3_MASS): a volumetric
// piece's density is its Gaussian inside its 3-sigma ellipsoid over this (step F)
const float DUST_BALL3_MASS = 0.9707094;
const float DUST_BOUND_R2 = 9.0;
const float DUST_BOUND_VAR = 0.91781;   // dust::DUST_BOUND_VAR
// the share of a ray's bounded Gaussian (centre -b/a, variance 1/a in t along eye + t u) on the
// ray's chord of the piece's 3-sigma ellipsoid (|t - t*| <= sqrt((9 - rr2) / a)) and of the
// packet's capsule [capLo, capHi] (dust_capsule_interval), in front of the eye and of the support,
// over DUST_BALL3_MASS (dust::ray_kept)
float dust_ray_kept(float a, float b, float rr2, float capLo, float capHi, vec3 eye, vec3 u, vec4 s) {
    if (!(rr2 < DUST_BOUND_R2)) return 0.0;
    float sa = sqrt(a);
    float tStar = -b / a;
    float k = sa * 0.7071067811865476;
    float h = sqrt((DUST_BOUND_R2 - rr2) / a);
    float lo = max(max(capLo, 0.0), tStar - h);
    float hi = min(capHi, tStar + h);
    if (dust_has_support(s)) {
        float nu = dot(s.xyz, u), ne = dot(s.xyz, eye) - s.w;
        if (abs(nu) < 1e-12) {
            if (!(ne >= 0.0)) return 0.0;
        } else {
            float tPlane = -ne / nu;
            if (nu > 0.0) lo = max(lo, tPlane);
            else hi = min(hi, tPlane);
        }
    }
    if (!(hi > lo)) return 0.0;
    return max(0.5 * (dust_erf((hi - tStar) * k) - dust_erf((lo - tStar) * k)), 0.0) / DUST_BALL3_MASS;
}

// A packet to splat (dust::Packet): start of the segment (local m), covariance, segment, flux,
// open ends, density at the ends (mean 1, linear between)
struct DustPacket {
    vec3  mean;
    float S[6];
    vec3  seg;
    float flux;
    bool  openStart;
    bool  openEnd;
    vec2  rho;
    vec3  segT;   // the leaf's Gaussianised bounded axis (dust::Packet::seg_t; the EWA ignores it)
};
// the mass of p in front of its support s: the normal marginal averaged over the capsule with its
// linear density, 16 midpoints (dust::support_mass); 1 without a support
float dust_support_mass(DustPacket p, vec4 s) {
    if (!dust_has_support(s)) return 1.0;
    float sig = sqrt(max(dust_var_along(p.S, s.xyz), 0.0));
    float m0 = dot(s.xyz, p.mean) - s.w, ds = dot(s.xyz, p.seg);
    float acc = 0.0, wsum = 0.0;
    for (int k = 0; k < 16; ++k) {
        float u = (float(k) + 0.5) / 16.0;
        float w = p.rho.x + (p.rho.y - p.rho.x) * u;
        float x = m0 + ds * u;
        float f = sig > 0.0 ? 0.5 * (1.0 + dust_erf(x / (sig * 1.4142135623730951))) : (x >= 0.0 ? 1.0 : 0.0);
        acc += w * f;
        wsum += w;
    }
    return wsum > 0.0 ? acc / wsum : 1.0;
}
// the sub-chord u0 <= u <= u1 of a chord packet (dust::sub_chord): start moved along the segment,
// segment and flux scaled by the kept part (linear density: kept length times the mean density
// there), rho re-interpolated and renormalised to mean 1, the ends as given
DustPacket dust_sub_chord(DustPacket p, float u0, float u1, bool openStart, bool openEnd) {
    float ra = p.rho.x + (p.rho.y - p.rho.x) * u0;
    float rb = p.rho.x + (p.rho.y - p.rho.x) * u1;
    float rm = max(0.5 * (ra + rb), 1e-6);
    DustPacket q;
    q.mean = p.mean + p.seg * u0;
    q.S = p.S;
    q.seg = p.seg * (u1 - u0);
    q.flux = p.flux * (u1 - u0) * rm;
    q.openStart = openStart;
    q.openEnd = openEnd;
    q.rho = vec2(ra / rm, rb / rm);
    q.segT = p.segT;
    return q;
}
// A cut packet the eye is not near, as the moments of its truncation (dust::support_truncated):
// DUST_SUPPORT_TRUNC_PIECES sub-chords (shared out over the chord's run through the plane's +-3
// sigma zone, one each before and after it; interior ends closed), each cut along the plane's
// normal into DUST_SUPPORT_TRUNC_SLABS slabs replaced by their doubly truncated moments
const uint DUST_SUPPORT_TRUNC_PIECES = 8u;
const uint DUST_SUPPORT_TRUNC_SLABS = 4u;
// the slabs' inner edges past the plane in deviations of the truncated marginal
// (dust::DUST_SUPPORT_SLAB_EDGES)
float dust_support_slab_edge(uint e) { return e == 0u ? 0.5 : (e == 1u ? 1.2 : 2.2); }
float dust_npdf(float z) { return exp(-0.5 * z * z) * 0.3989423; }
float dust_nupper(float z) { return 0.5 * (1.0 - dust_erf(z * 0.7071067811865476)); }
// the sub-chords' bounds (dust::support_truncated)
void dust_support_trunc_bounds(DustPacket p, vec4 s, float v0, out float bounds[DUST_SUPPORT_TRUNC_PIECES + 1u]) {
    float k = float(DUST_SUPPORT_TRUNC_PIECES);
    for (uint j = 0u; j <= DUST_SUPPORT_TRUNC_PIECES; ++j) bounds[j] = float(j) / k;
    float dn = dot(s.xyz, p.seg);
    if (abs(dn) > 0.0) {
        float half_ = 3.0 * sqrt(v0) / abs(dn);
        float ux = (s.w - dot(s.xyz, p.mean)) / dn;
        float za = max(ux - half_, 0.0), zb = min(ux + half_, 1.0);
        if (zb > za && (za > 0.0 || zb < 1.0)) {
            uint before = za > 0.0 ? 1u : 0u, after = zb < 1.0 ? 1u : 0u;
            uint inner = DUST_SUPPORT_TRUNC_PIECES - before - after;
            bounds[0] = 0.0;
            for (uint i = 0u; i <= inner; ++i) bounds[before + i] = za + (zb - za) * float(i) / float(inner);
            bounds[DUST_SUPPORT_TRUNC_PIECES] = 1.0;
        }
    }
}
// slab e of sub-chord j (dust::support_truncated): false when it holds nothing; q the slab's
// packet, w its kept flux (before the renormalisation over the packet)
bool dust_support_trunc_piece(DustPacket p, vec4 s, vec3 sn, float v0, float u0, float u1, uint j, uint e,
                              out DustPacket q, out float w) {
    w = 0.0;
    q = p;
    if (!(u1 > u0)) return false;
    q = dust_sub_chord(p, u0, u1, j == 0u && p.openStart, j + 1u == DUST_SUPPORT_TRUNC_PIECES && p.openEnd);
    vec3 n = s.xyz;
    float ra = q.rho.x, rb = q.rho.y;
    float uc = ra + rb > 0.0 ? (ra + 2.0 * rb) / (3.0 * (ra + rb)) : 0.5;
    float ds = dot(n, q.seg);
    float m = dot(n, q.mean + q.seg * uc) - s.w;
    float v = v0 + ds * ds / 12.0;
    if (!(v > 0.0)) {
        if (e != 0u || !(m >= 0.0)) return false;
        w = q.flux;
        return true;
    }
    float sig = sqrt(v);
    float alpha = -m / sig;
    float pk = dust_nupper(alpha);
    if (!(pk > 1e-6)) return false;
    float lam = dust_npdf(alpha) / pk;
    float t = sqrt(max(1.0 + alpha * lam - lam * lam, 1e-4));
    vec3 se = (sn + q.seg * (ds / 12.0)) * (1.0 / sig);
    float za = e == 0u ? alpha : alpha + dust_support_slab_edge(e - 1u) * t;
    bool last = e + 1u == DUST_SUPPORT_TRUNC_SLABS;
    float pa = dust_npdf(za), qa = dust_nupper(za);
    float pb = 0.0, qb = 0.0, bpb = 0.0;
    if (!last) {
        float zb = alpha + dust_support_slab_edge(e) * t;
        pb = dust_npdf(zb); qb = dust_nupper(zb); bpb = zb * pb;
    }
    float ps = qa - qb;
    if (!(ps > 1e-9)) return false;
    float mu = (pa - pb) / ps;
    float var_ = clamp(1.0 + (za * pa - bpb) / ps - mu * mu, 0.0, 1.0);
    float cv = (1.0 - var_) / v;
    q.S[0] -= cv * sn.x * sn.x;
    q.S[1] -= cv * sn.x * sn.y;
    q.S[2] -= cv * sn.x * sn.z;
    q.S[3] -= cv * sn.y * sn.y;
    q.S[4] -= cv * sn.y * sn.z;
    q.S[5] -= cv * sn.z * sn.z;
    q.mean += se * mu;
    w = q.flux * ps;
    return true;
}
// A packet of a young cell as its cone of grains (dust::conic_slabs): DUST_SUPPORT_TRUNC_PIECES
// equal sub-chords, each cut along the support's normal into slabs (edges -1, 0, 1 sigma of its
// normal marginal, the first from the plane when it cuts the packet), each slab's covariance mapped
// by the apex nn^T + s_k (I - nn^T) about the origin plane wO and its lateral shift scaled by s_k
const float DUST_CONIC_FAR = 1.0e4;
float dust_conic_slab_edge(uint i) { return i == 0u ? -1.0 : (i == 1u ? 0.0 : 1.0); }
vec3 dust_apex_map(vec3 x, vec3 n, float sk) {
    float xn = dot(n, x);
    return n * xn + (x - n * xn) * sk;
}
// slab e of sub-chord j (dust::conic_slabs): false when it does not exist or holds nothing; q the
// slab's packet, w its kept flux (before the renormalisation over the packet). sn = S n, v0 = n.S n,
// dC the packet centre's distance from the origin plane, cut whether the plane cuts the packet
bool dust_conic_piece(DustPacket p, vec4 s, float wO, vec3 sn, float v0, float dC, bool cut, uint j, uint e,
                      out DustPacket q, out float w) {
    w = 0.0;
    vec3 n = s.xyz;
    float k = float(DUST_SUPPORT_TRUNC_PIECES);
    float u0 = float(j) / k, u1 = float(j + 1u) / k;
    q = dust_sub_chord(p, u0, u1, j == 0u && p.openStart, j + 1u == DUST_SUPPORT_TRUNC_PIECES && p.openEnd);
    float qa = q.rho.x, qb = q.rho.y;
    float uc = qa + qb > 0.0 ? (qa + 2.0 * qb) / (3.0 * (qa + qb)) : 0.5;
    vec3 qc = q.mean + q.seg * uc;
    float ds = dot(n, q.seg);
    float v = v0 + ds * ds / 12.0;
    float sig = sqrt(max(v, 1e-30));
    float alpha = cut ? max((s.w - dot(n, qc)) / sig, -DUST_CONIC_FAR) : -DUST_CONIC_FAR;
    vec3 se = (sn + q.seg * (ds / 12.0)) * (1.0 / sig);
    // the slab edges above alpha: a suffix of (-1, 0, 1)
    uint c = 0u;
    for (uint i = 0u; i < DUST_SUPPORT_TRUNC_SLABS - 1u; ++i) if (dust_conic_slab_edge(i) > alpha) c++;
    uint ne = c + 1u;
    if (e >= ne) return false;
    uint first = DUST_SUPPORT_TRUNC_SLABS - 1u - c;
    float za = e == 0u ? alpha : dust_conic_slab_edge(first + e - 1u);
    float zb = e + 1u == ne ? DUST_CONIC_FAR : dust_conic_slab_edge(first + e);
    float pa = dust_npdf(za), qa_ = dust_nupper(za), apa = za * dust_npdf(za);
    float pb = dust_npdf(zb), qb_ = dust_nupper(zb), bpb = zb * dust_npdf(zb);
    float ps = qa_ - qb_;
    if (!(ps > 1e-9)) return false;
    float mu = (pa - pb) / ps;
    float var_ = clamp(1.0 + (apa - bpb) / ps - mu * mu, 0.0, 1.0);
    float cv = (1.0 - var_) / max(v, 1e-30);
    float C0 = q.S[0] - cv * sn.x * sn.x;
    float C1 = q.S[1] - cv * sn.x * sn.y;
    float C2 = q.S[2] - cv * sn.x * sn.z;
    float C3 = q.S[3] - cv * sn.y * sn.y;
    float C4 = q.S[4] - cv * sn.y * sn.z;
    float C5 = q.S[5] - cv * sn.z * sn.z;
    vec3 mean = q.mean + se * mu;
    float sk = max((dot(n, mean + q.seg * uc) - wO) / dC, 0.0);
    vec3 c0 = dust_apex_map(vec3(C0, C1, C2), n, sk);
    vec3 c1 = dust_apex_map(vec3(C1, C3, C4), n, sk);
    vec3 c2 = dust_apex_map(vec3(C2, C4, C5), n, sk);
    vec3 r0 = dust_apex_map(vec3(c0.x, c1.x, c2.x), n, sk);
    vec3 r1 = dust_apex_map(vec3(c0.y, c1.y, c2.y), n, sk);
    vec3 r2 = dust_apex_map(vec3(c0.z, c1.z, c2.z), n, sk);
    q.S[0] = r0.x; q.S[1] = r0.y; q.S[2] = r0.z; q.S[3] = r1.y; q.S[4] = r1.z; q.S[5] = r2.z;
    vec3 off = se * mu;
    vec3 lat = off - n * dot(n, off);
    q.mean = mean + lat * (sk - 1.0);
    w = q.flux * ps;
    return true;
}
const int DUST_SPLIT_DROP  = 0;
const int DUST_SPLIT_KEEP  = 1;
const int DUST_SPLIT_CLIP  = 2;   // a replaces p at the same level
const int DUST_SPLIT_SPLIT = 3;   // a and b, half the flux each
const int DUST_SPLIT_VOLUME = 4;  // the eye inside or near it: the exact ray integral (dust_volume_scatter)

// the parameter in [0, 1] of the point of the segment d (from the origin) nearest to q
// (dust::segment_param)
float dust_segment_param(vec3 q, vec3 d) {
    float dd = dot(d, d);
    return dd > 0.0 ? clamp(dot(q, d) / dd, 0.0, 1.0) : 0.0;
}
// whether a packet is drawn by the exact ray integral (dust::is_volumetric; perspective views
// only): tr S > DUST_VOLUME_RATIO^2 dist^2, dist from the eye to the packet's segment
bool dust_is_volumetric(DustPacket p, vec3 eye) {
    vec3 q = eye - p.mean;
    vec3 r = q - p.seg * dust_segment_param(q, p.seg);
    float tr = p.S[0] + p.S[3] + p.S[5];
    return tr > DUST_VOLUME_RATIO * DUST_VOLUME_RATIO * dot(r, r);
}
// lower Cholesky factor l00 l10 l20 l11 l21 l22, pivots^2 floored at DUST_DET_ANISO_FLOOR tr S
// (dust::cholesky3)
void dust_cholesky3(float S[6], out float L[6]) {
    precise float fl = DUST_DET_ANISO_FLOOR * max(S[0] + S[3] + S[5], 1e-30);
    precise float l00 = sqrt(max(S[0], fl));
    precise float l10 = S[1] / l00;
    precise float l20 = S[2] / l00;
    precise float l11 = sqrt(max(S[3] - l10 * l10, fl));
    precise float l21 = (S[4] - l20 * l10) / l11;
    precise float l22 = sqrt(max(S[5] - l20 * l20 - l21 * l21, fl));
    L[0] = l00; L[1] = l10; L[2] = l20; L[3] = l11; L[4] = l21; L[5] = l22;
}
// L^-1 x, forward substitution (dust::cholesky_solve3)
vec3 dust_cholesky_solve3(float L[6], vec3 x) {
    precise float y0 = x.x / L[0];
    precise float y1 = (x.y - L[1] * y0) / L[3];
    precise float y2 = (x.z - L[2] * y0 - L[4] * y1) / L[5];
    return vec3(y0, y1, y2);
}
// pyramid level of a volumetric packet of angular sigma thetaPx (dust::volume_level)
uint dust_volume_level(float thetaPx, uint levels) {
    float r = thetaPx / DUST_VOLUME_TEXELS_PER_SIGMA;
    if (!(r > 1.0)) return 0u;
    float l = floor(log2(r));
    return min(uint(max(l, 0.0)), max(levels, 1u) - 1u);
}
// the radiance weight of the dust at x seen along the ray u from the eye (dust::volume_weight)
float dust_volume_weight(vec3 meanRef, vec3 x, vec3 u) {
    vec3 d = x - meanRef;
    vec3 rVec = dust_phot_sun * (-dust_phot_rh * DUST_AU_M) + d;
    float rn = max(sqrt(dot(rVec, rVec)), 1.0);
    return dust_radiance_weight(dust_phot_lut, dust_phot_albedo, rVec * (-1.0 / rn), -u, rn / DUST_AU_M);
}
// One step of the perspective split (dust::split_step; the rules, the proof and the numbers are
// documented there). depth = (mean - eye).f of the chord's start, sigma_v^2 = f^T S f, span = seg.f:
// (1) depth + 3 sigma_v + max(0, span) < 0: drop (wholly behind the eye plane);
// (2b) perspective, the eye inside or near the packet (dust_is_volumetric): drawn whole by its
//     exact ray integral (dust_volume_scatter), which cuts every ray at the eye — before the clip;
// (2) the chord's ends on both sides of the plane: clip to the in-front part less
//     DUST_SPLIT_CLIP_EPS of it (the kept start strictly in front), the cut end closed;
// (3) 3 sigma_v > depth (the ellipsoid crosses the plane) or, in perspective only (the EWA is
//     exact for an orthographic projection), sigma_v > DUST_SPLIT_EPS depth or
//     |span| > DUST_SPLIT_EPS depth, below the frame's depth limit depthMax (DUST_PYR_SPLIT_DEPTH):
//     bisect the chord (inner ends open) when |span| > 3 sigma_v, else slice the Gaussian by
//     depth: h = DUST_SPLIT_A S f / sigma_v, children N(mean -+ h, S - h h^T), flux / 2 each
//     (mean and covariance of the mixture are the packet's; S - h h^T is PSD: Schur complement);
// (4) else keep.
int dust_split_step(DustPacket p, bool clipped, uint level, uint depthMax, vec3 eye, vec3 fwd, bool perspective, bool force, out DustPacket a, out DustPacket b) {
    a = p; b = p;
    if (!(dot(fwd, fwd) > 0.0)) return DUST_SPLIT_KEEP;
    float depth = dot(p.mean - eye, fwd);
    float sigmaV = sqrt(max(dust_var_along(p.S, fwd), 0.0));
    float spanS = dot(p.seg, fwd);
    float span = abs(spanS);
    if (depth + 3.0 * sigmaV + max(spanS, 0.0) < 0.0) return DUST_SPLIT_DROP;
    // (2b) before the clip (step F3): the eye inside or near the packet — the exact ray integral
    // cuts each ray at the eye itself, so a chord across the eye plane is drawn whole (the clip's
    // closed end ramped the density in front of the eye down over the pieces' sigma: a chord along
    // the view with the eye inside drew 0.79 of its light)
    if (!force && perspective && dust_is_volumetric(p, eye)) return DUST_SPLIT_VOLUME;
    float d1 = depth + spanS;
    if (!clipped && span > 0.0 && (depth < 0.0) != (d1 < 0.0)) {
        float uStar = -depth / spanS;
        if (depth < 0.0) {
            float uLo = uStar + DUST_SPLIT_CLIP_EPS * (1.0 - uStar);
            a = dust_sub_chord(p, uLo, 1.0, false, p.openEnd);
            return DUST_SPLIT_CLIP;
        }
        float uHi = uStar * (1.0 - DUST_SPLIT_CLIP_EPS);
        a = dust_sub_chord(p, 0.0, uHi, p.openStart, false);
        return DUST_SPLIT_CLIP;
    }
    bool want = force || 3.0 * sigmaV > depth
             || (perspective && (sigmaV > DUST_SPLIT_EPS * depth || span > DUST_SPLIT_EPS * depth));
    if (want && level < min(depthMax, DUST_SPLIT_DEPTH_MAX)) {
        if (span > 3.0 * sigmaV) {
            a = dust_sub_chord(p, 0.0, 0.5, p.openStart, true);
            b = dust_sub_chord(p, 0.5, 1.0, true, p.openEnd);
            return DUST_SPLIT_SPLIT;
        }
        if (sigmaV > 0.0) {
            float k = DUST_SPLIT_A / sigmaV;
            vec3 h = vec3((p.S[0] * fwd.x + p.S[1] * fwd.y + p.S[2] * fwd.z) * k,
                          (p.S[1] * fwd.x + p.S[3] * fwd.y + p.S[4] * fwd.z) * k,
                          (p.S[2] * fwd.x + p.S[4] * fwd.y + p.S[5] * fwd.z) * k);
            a.S[0] = p.S[0] - h.x * h.x;
            a.S[1] = p.S[1] - h.x * h.y;
            a.S[2] = p.S[2] - h.x * h.z;
            a.S[3] = p.S[3] - h.y * h.y;
            a.S[4] = p.S[4] - h.y * h.z;
            a.S[5] = p.S[5] - h.z * h.z;
            a.flux = 0.5 * p.flux;
            // the sliced covariance no longer holds seg_t's 1.21 term (dust::Packet::seg_t)
            a.segT = vec3(0.0);
            b.S = a.S;
            b.flux = a.flux;
            b.segT = vec3(0.0);
            a.mean = p.mean - h;
            b.mean = p.mean + h;
            return DUST_SPLIT_SPLIT;
        }
    }
    // (4) still too deep for one Jacobian at the depth limit: the exact ray integral (dust::split_step)
    // (a chord deep only by its span with the deep-chord bit: kept, its EWA at dust_deep_chord_factors)
    if (want && !force && perspective) {
        bool deepOnly = depth > 0.0 && span > 3.0 * sigmaV
                     && !(3.0 * sigmaV > depth) && !(sigmaV > DUST_SPLIT_EPS * depth);
        if (!(deepOnly && dust_deep_chord)) return DUST_SPLIT_VOLUME;
    }
    return DUST_SPLIT_KEEP;
}

// The speck shell's hand-over (dust::specks::ShellCut, same arithmetic): the ball of radius dMax
// (local m) about the eye whose grains the speck pass draws; dMax <= 0 or not finite: none
const int DUST_SHELL_OUTSIDE   = 0;
const int DUST_SHELL_INSIDE    = 1;
const int DUST_SHELL_STRADDLES = 2;
bool dust_shell_active(float dMax) { return dMax > 0.0 && !isinf(dMax); }
// wholly inside / outside / straddling, conservatively: the packet's centre (its segment's
// midpoint), its 3-sigma reach sqrt(tr S) plus half its segment (ShellCut::classify)
int dust_shell_classify(DustPacket p, vec3 eye, float dMax) {
    if (!dust_shell_active(dMax)) return DUST_SHELL_OUTSIDE;
    vec3 centre = p.mean + p.seg * 0.5;
    vec3 dd = centre - eye;
    float dist = sqrt(dot(dd, dd));
    float tr = max(p.S[0] + p.S[3] + p.S[5], 0.0);
    float reach = 3.0 * sqrt(tr) + 0.5 * sqrt(dot(p.seg, p.seg));
    if (dist + reach < dMax) return DUST_SHELL_INSIDE;
    if (dist - reach > dMax) return DUST_SHELL_OUTSIDE;
    return DUST_SHELL_STRADDLES;
}
// the fraction of a straddling packet's mass outside the ball (ShellCut::outside_fraction: the
// radial erf about the eye times the transverse disc term times kappa, documented there)
float dust_shell_outside_fraction(DustPacket p, vec3 eye, float dMax) {
    if (!dust_shell_active(dMax)) return 1.0;
    vec3 centre = p.mean + p.seg * 0.5;
    vec3 d = centre - eye;
    float dist = sqrt(dot(d, d));
    float c0 = p.S[0] + p.seg.x * p.seg.x / 12.0;
    float c1 = p.S[1] + p.seg.x * p.seg.y / 12.0;
    float c2 = p.S[2] + p.seg.x * p.seg.z / 12.0;
    float c3 = p.S[3] + p.seg.y * p.seg.y / 12.0;
    float c4 = p.S[4] + p.seg.y * p.seg.z / 12.0;
    float c5 = p.S[5] + p.seg.z * p.seg.z / 12.0;
    float tr = max(c0 + c3 + c5, 1e-30);
    vec3 rh = dist > 0.0 ? d * (1.0 / dist) : vec3(0.0, 0.0, 1.0);
    float varR = max(c0 * rh.x * rh.x + c3 * rh.y * rh.y + c5 * rh.z * rh.z
                     + 2.0 * (c1 * rh.x * rh.y + c2 * rh.x * rh.z + c4 * rh.y * rh.z), 0.0);
    float sigmaR = max(sqrt(varR), 1e-20);
    float varT = max((tr - varR) * 0.5, 1e-30);
    float radial = 0.5 * (dust_erf((dMax - dist) / (sigmaR * 1.4142135623730951))
                        - dust_erf((-dMax - dist) / (sigmaR * 1.4142135623730951)));
    float eT = exp(-dMax * dMax / (2.0 * varT));
    float transverse = 1.0 - eT;
    float kappa = 1.0 - eT / 3.0;
    return clamp(1.0 - clamp(radial * transverse * kappa, 0.0, 1.0), 0.0, 1.0);
}

// Projects a packet (mean m, covariance S = xx xy xz yy yz zz m^2, segment d m, flux) with the
// Jacobian of `mvp` at its chord's midpoint (dust::project_packet). false: the start behind the camera (a
// guard: dust_split_step drops, clips or slices a packet before it gets here), nothing in front
// of the eye plane, or off screen.
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
    // the Jacobian at the chord's midpoint (the start when the midpoint is not in front of the
    // eye; dust::project_packet_lit): exact to second order in the chord's depth span
    // only for a chord whose depth span is within DUST_SPLIT_EPS of its depth (|w_end - w| <= eps w;
    // a deeper chord keeps the start, dust::project_packet_lit); bit 23 of DUST_PYR_SPLIT_DEPTH
    // (diagnostics, AETHERVK_DUST_JACOBIAN_AT_START=1): always at the start
    float wEnd = dot(r3, mean + d) + mvp[3][3];
    bool shallow = abs(wEnd - clip.w) <= DUST_SPLIT_EPS * clip.w;
    float jAt = (shallow && ((dust_pyr_split_word >> 23u) & 1u) == 0u) ? 0.5 : 0.0;
    // the deep chord's EWA (dust_deep_chord, off by default): the Jacobian at the geometric-mean
    // depth and the flux by the perspective's light factor (dust::project_packet_lit)
    float deepK = 1.0;
    if (!shallow && dust_deep_chord && clip.w > 0.0 && wEnd > 0.0) {
        vec2 f = dust_deep_chord_factors(clip.w, wEnd, rho);
        jAt = f.x;
        deepK = f.y;
    }
    vec4 clipJ = mvp * vec4(mean + jAt * d, 1.0);
    if (!(clipJ.w > 0.0)) clipJ = clip;
    vec2 ndcJ = clipJ.xy / clipJ.w;
    vec3 jx = (r0 - ndcJ.x * r3) * (0.5 * W / clipJ.w);
    vec3 jy = (r1 - ndcJ.y * r3) * (0.5 * H / clipJ.w);
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
    // the view direction (dust_view_forward: the same one the split uses)
    bool perspective;
    vec3 fwd = dust_view_forward(mvp, eyeLocal, perspective);
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
    P.amp = exposure * flux * deepK * sqrt(detG) / (2.0 * DUST_PI_F32 * sqrt(detC));
    // the part of the packet in front of the eye (dust::project_packet_lit): its Gaussian along
    // the view direction, centred on the capsule's middle, cut at the eye's plane (1 for the
    // split's leaves; it matters for the leaves left straddling the plane)
    {
        if (dot(fwd, fwd) > 0.0) {
            float depth = dot(mean + 0.5 * d - eyeLocal, fwd);
            float sigmaV = sqrt(max(dust_var_along(S, fwd), 0.0));
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
// exp(x) - 1 without the cancellation at small x (mirror of dust::expm1_f32: the same polynomial
// and threshold on both sides)
float dust_expm1(float x) {
    return abs(x) < 1e-3 ? x * (1.0 + 0.5 * x * (1.0 + x / 3.0)) : exp(x) - 1.0;
}

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
    // the squared distance to the line from the residual e - u0 d, not c - b^2/a (catastrophic
    // cancellation for a chord whose start is far off screen; dust::capsule_tau)
    precise vec2 r = e - u0 * d;
    precise float q = max(m00 * r.x * r.x + 2.0 * m01 * r.x * r.y + m11 * r.y * r.y, 0.0);
    // a linear density along the segment (dust::capsule_tau): the integral of u exp(-a(u-u0)^2/2)
    // over it is u0 I0 + (E_start - E_end)/a with the Gaussian's values at the closed ends
    float i0 = sqrt(2.0 * DUST_PI_F32 / a) * span;
    float weight;
    if (P.rho.y == P.rho.x) {
        weight = P.rho.x * i0;
    } else {
        float i1;
        if (!P.openStart && !P.openEnd) {
            // both ends closed: the Gaussians' difference from expm1 where they are close (the
            // cancellation at a << 1), the plain difference otherwise (dust::capsule_tau)
            float z1 = -0.5 * a * u0 * u0;
            float z2 = -0.5 * a * (1.0 - u0) * (1.0 - u0);
            float dz = z2 - z1;
            float diff = abs(dz) < 1.0 ? -exp(z1) * dust_expm1(dz) : exp(z1) - exp(z2);
            i1 = u0 * i0 + diff / a;
        } else {
            float eStart = P.openStart ? 0.0 : exp(-0.5 * a * u0 * u0);
            float eEnd = P.openEnd ? 0.0 : exp(-0.5 * a * (1.0 - u0) * (1.0 - u0));
            i1 = u0 * i0 + (eStart - eEnd) / a;
        }
        weight = max(P.rho.x * i0 + (P.rho.y - P.rho.x) * i1, 0.0);
    }
    return P.amp * exp(-0.5 * q) * weight;
}

// tr(Sigma_v)/3 of a cluster's moments
float dust_mean_var(uint mi) {
    return (dust_mom.m[mi].cov_a.x + dust_mom.m[mi].cov_a.w + dust_mom.m[mi].cov_b_age_id.y) / 3.0;
}
// sigma_pred / sigma_self of the velocity dispersion (1 without a wider predecessor)
// (dust::width_ratio)
const float DUST_WIDTH_RATIO_MAX = 1024.0;  // dust::WIDTH_RATIO_MAX
float dust_width_ratio(uint mi, bool hasPred, uint pi) {
    float own = dust_mean_var(mi);
    if (!hasPred || !(dust_mean_var(pi) > max(own, 0.0))) return 1.0;
    if (!(own > 0.0)) return DUST_WIDTH_RATIO_MAX;
    return min(sqrt(dust_mean_var(pi) / own), DUST_WIDTH_RATIO_MAX);
}
// bound k of n time pieces of a segment whose width grows ratio-fold (dust::piece_bound): every
// piece the same width ratio, (ratio^(k/n) - 1)/(ratio - 1); equal steps when ratio ~ 1
float dust_piece_bound(uint k, uint n, float ratio) {
    if (k >= n) return 1.0;
    if (!(ratio > 1.0001)) return float(k) / float(n);
    float e = exp(log(ratio) * (float(k) / float(n)));
    return clamp((e - 1.0) / (ratio - 1.0), 0.0, 1.0);
}
// pieces a capsule is cut into: clamp(ceil(ratio) - 1, 1, DUST_TIME_PIECES_MAX) (dust::time_pieces)
uint dust_time_pieces(float ratio) {
    uint c = uint(max(ceil(ratio), 1.0));
    return clamp(c - 1u, 1u, DUST_TIME_PIECES_MAX);
}
// Packet of size bin b of a cluster (dust::subpackets): uniform along the bin's chord (a Gaussian
// of DUST_CHORD_VAR_FACTOR half-chords^2 along it), Sigma_v scaled by the bin's speed factor,
// 1/DUST_SIZE_BINS of the flux
void dust_bin_packet_sub(uint mi, uint b, uint j, uint nj, out vec3 mean, out float S[6], out float flux) {
    vec3 a, c; float fa, fc;
    dust_sub_edges(mi, b, j, nj, a, c, fa, fc);
    mean = 0.5 * (a + c);
    vec3 h = 0.5 * (c - a);
    float f = 0.5 * (fa + fc);
    float f2 = f * f, k = DUST_CHORD_VAR_FACTOR;
    S[0] = f2 * dust_mom.m[mi].cov_a.x + k * h.x * h.x;
    S[1] = f2 * dust_mom.m[mi].cov_a.y + k * h.x * h.y;
    S[2] = f2 * dust_mom.m[mi].cov_a.z + k * h.x * h.z;
    S[3] = f2 * dust_mom.m[mi].cov_a.w + k * h.y * h.y;
    S[4] = f2 * dust_mom.m[mi].cov_b_age_id.x + k * h.y * h.z;
    S[5] = f2 * dust_mom.m[mi].cov_b_age_id.y + k * h.z * h.z;
    flux = dust_mom.m[mi].mean_flux.w * float(dust_group) / float(DUST_SIZE_BINS) * dust_sub_share(mi, b, j, nj);
}
void dust_bin_packet(uint mi, uint b, out vec3 mean, out float S[6], out float flux) {
    dust_bin_packet_sub(mi, b, 0u, 1u, mean, S, flux);
}

// The packet drawn for size bin b (dust::bin_chord_packet): uniform along its chord (mean = the
// start edge, seg = the chord), Sigma_v scaled by the bin's speed factor plus the bin's time
// segment segT as a Gaussian of DUST_CHORD_VAR_FACTOR half-segments^2, 1/DUST_SIZE_BINS of the flux
void dust_bin_chord_sub(uint mi, uint b, uint j, uint nj, vec3 segT, out vec3 mean, out vec3 chord, out float S[6], out float flux) {
    vec3 a, c; float fa, fc;
    dust_sub_edges(mi, b, j, nj, a, c, fa, fc);
    mean = a;
    chord = c - a;
    vec3 h = 0.5 * segT;
    float f = 0.5 * (fa + fc);
    float f2 = f * f, k = DUST_CHORD_VAR_FACTOR;
    S[0] = f2 * dust_mom.m[mi].cov_a.x + k * h.x * h.x;
    S[1] = f2 * dust_mom.m[mi].cov_a.y + k * h.x * h.y;
    S[2] = f2 * dust_mom.m[mi].cov_a.z + k * h.x * h.z;
    S[3] = f2 * dust_mom.m[mi].cov_a.w + k * h.y * h.y;
    S[4] = f2 * dust_mom.m[mi].cov_b_age_id.x + k * h.y * h.z;
    S[5] = f2 * dust_mom.m[mi].cov_b_age_id.y + k * h.z * h.z;
    flux = dust_mom.m[mi].mean_flux.w * float(dust_group) / float(DUST_SIZE_BINS) * dust_sub_share(mi, b, j, nj);
}
void dust_bin_chord(uint mi, uint b, vec3 segT, out vec3 mean, out vec3 chord, out float S[6], out float flux) {
    dust_bin_chord_sub(mi, b, 0u, 1u, segT, mean, chord, S, flux);
}

bool dust_project_point(vec3 p, mat4 mvp, float W, float H, out vec2 px);
// sweep count of bin b (dust::time_sweep): the larger screen displacement of its two edges to
// the predecessor's per DUST_TIME_SWEEP_STEP_PX, at most DUST_TIME_SWEEP_MAX
uint dust_time_sweep(uint mi, uint pi, uint b, mat4 mvp, float W, float H) {
    float gap = 0.0;
    for (uint e = b; e <= b + dust_group; e += dust_group) {
        vec2 a, c;
        if (dust_project_point(dust_mom.m[mi].edges[e].xyz, mvp, W, H, a) && dust_project_point(dust_mom.m[pi].edges[e].xyz, mvp, W, H, c))
            gap = max(gap, length(c - a));
    }
    if (isinf(gap) || isnan(gap)) return 1u;
    // the level of detail's step (dust_lod_px), at least a pixel
    float step = max(dust_lod_px, DUST_TIME_SWEEP_STEP_PX);
    return clamp(uint(min(ceil(gap / step), 1.0e6)), 1u, DUST_TIME_SWEEP_MAX);
}
// the k-th of nS sweep sub-capsules of bin b (dust::bin_sweep_packet): the chord interpolated at
// u = (k + 1/2)/nS between the bin's own edges and the predecessor's, Sigma_v scaled by the
// bin's speed factor plus the residual Gaussian of the larger edge displacement / 2 nS along it,
// 1/(DUST_SIZE_BINS nS) of the flux
void dust_bin_sweep_sub(uint mi, uint pi, uint b, uint j, uint nj, uint k, uint nS, out vec3 mean, out vec3 chord, out float S[6], out float flux) {
    vec3 a, c, pa, pc; float fa, fc, pfa, pfc;
    dust_sub_edges(mi, b, j, nj, a, c, fa, fc);
    dust_sub_edges(pi, b, j, nj, pa, pc, pfa, pfc);
    vec3 da = pa - a, dc = pc - c;
    vec3 dmax = dot(da, da) > dot(dc, dc) ? da : dc;
    float u = (float(k) + 0.5) / float(nS);
    vec3 ak = a + da * u, ck = c + dc * u;
    float f = 0.5 * (fa + fc);
    float f2 = f * f;
    float kv = DUST_CHORD_VAR_FACTOR;
    vec3 h = dmax * (0.5 / float(nS));
    mean = ak;
    chord = ck - ak;
    S[0] = f2 * dust_mom.m[mi].cov_a.x + kv * h.x * h.x;
    S[1] = f2 * dust_mom.m[mi].cov_a.y + kv * h.x * h.y;
    S[2] = f2 * dust_mom.m[mi].cov_a.z + kv * h.x * h.z;
    S[3] = f2 * dust_mom.m[mi].cov_a.w + kv * h.y * h.y;
    S[4] = f2 * dust_mom.m[mi].cov_b_age_id.x + kv * h.y * h.z;
    S[5] = f2 * dust_mom.m[mi].cov_b_age_id.y + kv * h.z * h.z;
    flux = dust_mom.m[mi].mean_flux.w * float(dust_group) / float(DUST_SIZE_BINS * nS) * dust_sub_share(mi, b, j, nj);
}
void dust_bin_sweep(uint mi, uint pi, uint b, uint k, uint nS, out vec3 mean, out vec3 chord, out float S[6], out float flux) {
    dust_bin_sweep_sub(mi, pi, b, 0u, 1u, k, nS, mean, chord, S, flux);
}
// Rodrigues rotation of v by angle about the unit axis
vec3 dust_rotate3(vec3 v, vec3 axis, float angle) {
    float s = sin(angle), c = cos(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}
// R S R^T for the rotation by angle about the unit axis (dust::rotate_cov; S as xx, xy, xz, yy,
// yz, zz): the cone pieces' predecessor covariance turned to the piece's emission direction
void dust_rotate_cov(inout float S[6], vec3 axis, float angle) {
    vec3 r[3];
    r[0] = dust_rotate3(vec3(1.0, 0.0, 0.0), axis, angle);
    r[1] = dust_rotate3(vec3(0.0, 1.0, 0.0), axis, angle);
    r[2] = dust_rotate3(vec3(0.0, 0.0, 1.0), axis, angle);
    mat3 m = mat3(S[0], S[1], S[2], S[1], S[3], S[4], S[2], S[4], S[5]);
    float o[6];
    uint idx = 0u;
    for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
            float acc = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) acc += r[i][a] * m[i][j] * r[j][b];
            }
            o[idx++] = acc;
        }
    }
    for (int k = 0; k < 6; ++k) S[k] = o[k];
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
// the cell line of a sample (dust::cell_line): its size polyline o + f c in the speed factor f;
// false once radiation pressure bent it (the middle edge > 5 % of the chord off the line)
bool dust_cell_line(uint mi, out vec3 o, out vec3 c) {
    uint nb = DUST_SIZE_BINS;
    vec3 e0 = dust_mom.m[mi].edges[0].xyz, e1 = dust_mom.m[mi].edges[nb].xyz;
    float f0 = dust_edge_factor(mi, 0u), f1 = dust_edge_factor(mi, nb);
    o = vec3(0.0); c = vec3(0.0);
    if (!(f1 > f0)) return false;
    c = (e1 - e0) * (1.0 / (f1 - f0));
    o = e0 - c * f0;
    vec3 d = dust_mom.m[mi].edges[nb / 2u].xyz - (o + c * dust_edge_factor(mi, nb / 2u));
    vec3 ch = e1 - e0;
    return dot(ch, ch) > 0.0 && !(dot(d, d) > 0.0025 * dot(ch, ch));
}
// the support plane of a cell line (dust::support_plane): normal along c, through the slowest
// grain (f_lo three relative speed spreads slow, at the origin when <= 0)
vec4 dust_support_plane(vec3 o, vec3 c, float vStd, float fLo) {
    float cn = sqrt(dot(c, c));
    if (!(cn > 0.0)) return vec4(0.0);
    vec3 n = c * (1.0 / cn);
    float s0 = max(fLo * (1.0 - 3.0 * vStd), 0.0);
    vec3 x = o + n * (cn * s0);
    return vec4(n, dot(n, x));
}
// a cone piece's support (dust::subpackets_supported): from the cell line at the piece's centre
// age uc — the two samples' lines interpolated along the spiral, the predecessor's direction
// turned to the piece's emission and scaled with the age; the sample's own without both lines
// or with a single piece
vec4 dust_piece_support(bool pieces, bool lineM, bool lineP, vec3 oM, vec3 cM, vec3 oP, vec3 cP,
                        bool spiral, vec3 axis, float omega, float ageO, float ageP, float uc,
                        float vStd, float fLo, vec4 ownSup) {
    if (!pieces || !lineM || !lineP) return ownSup;
    float ageU = ageO + uc * (ageP - ageO);
    vec3 o, c;
    if (spiral) {
        o = dust_spiral_point(oM, oP, ageO, ageP, axis, omega, uc);
        c = dust_rotate3(cP, axis, omega * (ageP - ageU)) * (ageU / max(ageP, 1e-30));
    } else {
        o = oM + (oP - oM) * uc;
        c = cM + (cP - cM) * uc;
    }
    return dust_support_plane(o, c, vStd, fLo);
}
// dust_piece_support with the cell origin along the plane's normal (dust::subpackets_cells:
// has = the plane is set, wO = n.o): the own cell's (ownSup, ownHas, ownO) when not a piece
vec4 dust_piece_support_o(bool pieces, bool lineM, bool lineP, vec3 oM, vec3 cM, vec3 oP, vec3 cP,
                          bool spiral, vec3 axis, float omega, float ageO, float ageP, float uc,
                          float vStd, float fLo, vec4 ownSup, bool ownHas, float ownO, out bool has, out float wO) {
    if (!pieces || !lineM || !lineP) { has = ownHas; wO = ownO; return ownSup; }
    float ageU = ageO + uc * (ageP - ageO);
    vec3 o, c;
    if (spiral) {
        o = dust_spiral_point(oM, oP, ageO, ageP, axis, omega, uc);
        c = dust_rotate3(cP, axis, omega * (ageP - ageU)) * (ageU / max(ageP, 1e-30));
    } else {
        o = oM + (oP - oM) * uc;
        c = cM + (cP - cM) * uc;
    }
    vec4 sp = dust_support_plane(o, c, vStd, fLo);
    has = dust_has_support(sp);
    wO = dot(sp.xyz, o);
    return sp;
}
// the spiral gate of a sample with a predecessor (dust::spiral_of)
bool dust_spiral_of(uint mi, uint pi, out vec3 axis, out float omega) {
    omega = dust_spin_rate(mi);
    axis = dust_spin_axis(mi);
    float dAge = max(dust_mom.m[pi].cov_b_age_id.z - dust_mom.m[mi].cov_b_age_id.z, 0.0);
    return omega * dAge > DUST_SPIRAL_MIN_TURN && dust_mom.m[mi].cov_b_age_id.z <= DUST_SPIRAL_MAX_AGE_SAMPLES * dAge;
}

// The whole polyline as one packet (dust::pooled_packet): the bins' mean, their mean covariance
// plus the scatter of their midpoints, the whole flux
void dust_pooled_packet(uint mi, out vec3 mean, out float S[6], out float flux) {
    float n = float(DUST_SIZE_BINS);
    mean = vec3(0.0);
    for (uint b = 0u; b < DUST_SIZE_BINS; ++b) mean += 0.5 * (dust_mom.m[mi].edges[b].xyz + dust_mom.m[mi].edges[b + 1u].xyz) / n;
    for (int k = 0; k < 6; ++k) S[k] = 0.0;
    for (uint b = 0u; b < DUST_SIZE_BINS; ++b) {
        vec3 mb; float Sb[6]; float fb;
        dust_bin_packet(mi, b, mb, Sb, fb);
        vec3 d = mb - mean;
        S[0] += (Sb[0] + d.x * d.x) / n;
        S[1] += (Sb[1] + d.x * d.y) / n;
        S[2] += (Sb[2] + d.x * d.z) / n;
        S[3] += (Sb[3] + d.y * d.y) / n;
        S[4] += (Sb[4] + d.y * d.z) / n;
        S[5] += (Sb[5] + d.z * d.z) / n;
    }
    flux = dust_mom.m[mi].mean_flux.w;
}
// the mean covariance of the bins alone, without the scatter of their midpoints
// (dust::pooled_within_cov): the width a predecessor hands to the pieces of a point sample
void dust_pooled_within(uint mi, out float S[6]) {
    float n = float(DUST_SIZE_BINS);
    for (int k = 0; k < 6; ++k) S[k] = 0.0;
    for (uint b = 0u; b < DUST_SIZE_BINS; ++b) {
        vec3 mb; float Sb[6]; float fb;
        dust_bin_packet(mi, b, mb, Sb, fb);
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

// The fraction u1 - u0 of the screen segment mu + u seg, u in [0, 1], inside the frame
// [-ex, w + ex] x [-ey, h + ey] (Liang-Barsky; 0 when it misses it): the drawable length of a
// chord for the texel estimate of the scatter valve (dust::visible_fraction)
float dust_visible_fraction(vec2 mu, vec2 seg, float ex, float ey, float w, float h) {
    float u0 = 0.0, u1 = 1.0;
    float pk[4] = float[4](-seg.x, seg.x, -seg.y, seg.y);
    float qk[4] = float[4](mu.x + ex, w + ex - mu.x, mu.y + ey, h + ey - mu.y);
    for (int k = 0; k < 4; ++k) {
        if (pk[k] == 0.0) {
            if (qk[k] < 0.0) return 0.0;
        } else {
            float t = qk[k] / pk[k];
            if (pk[k] < 0.0) u0 = max(u0, t);
            else u1 = min(u1, t);
        }
    }
    return u1 > u0 ? u1 - u0 : 0.0;
}

// Pixel position of a local point (w <= 0: behind the camera, returns false)
bool dust_project_point(vec3 p, mat4 mvp, float W, float H, out vec2 px) {
    vec4 clip = mvp * vec4(p, 1.0);
    if (!(clip.w > 0.0)) return false;
    px = (clip.xy / clip.w * 0.5 + 0.5) * vec2(W, H);
    return true;
}
// clip w of a local point (dust::clip_w): the depth along the view in perspective, 1 orthographic
float dust_clip_w(vec3 p, mat4 mvp) {
    return mvp[0][3] * p.x + mvp[1][3] * p.y + mvp[2][3] * p.z + mvp[3][3];
}
// Whether the size polyline is one pooled packet (dust::polyline_merged): the two end edges and
// the middle one within DUST_MERGE_PX of the mean on screen, and (perspective) the 17 edges' depth
// span below DUST_SPLIT_EPS of the depth at the mean (a syndyne along the line of sight is a
// point on screen but deep: not a pooled Gaussian)
bool dust_polyline_merged(uint mi, mat4 mvp, float W, float H) {
    vec2 c, p;
    if (!dust_project_point(dust_mom.m[mi].mean_flux.xyz, mvp, W, H, c)) return false;
    uint idx[3] = uint[3](0u, DUST_SIZE_BINS / 2u, DUST_SIZE_BINS);
    for (int k = 0; k < 3; ++k) {
        if (!dust_project_point(dust_mom.m[mi].edges[idx[k]].xyz, mvp, W, H, p)) return false;
        vec2 d = p - c;
        if (!(dot(d, d) < DUST_MERGE_PX * DUST_MERGE_PX)) return false;
    }
    float wm = dust_clip_w(dust_mom.m[mi].mean_flux.xyz, mvp);
    float lo = wm, hi = wm;
    for (uint j = 0u; j < DUST_SIZE_EDGES; ++j) {
        float we = dust_clip_w(dust_mom.m[mi].edges[j].xyz, mvp);
        lo = min(lo, we);
        hi = max(hi, we);
    }
    return hi - lo < DUST_SPLIT_EPS * wm;
}
// whether the time segment to the predecessor is shallow enough to pool (dust::time_shallow)
bool dust_time_shallow(uint mi, uint pi, mat4 mvp) {
    float wm = dust_clip_w(dust_mom.m[mi].mean_flux.xyz, mvp), wp = dust_clip_w(dust_mom.m[pi].mean_flux.xyz, mvp);
    return abs(wp - wm) < DUST_SPLIT_EPS * wm || wp == wm;
}

// ─── The cluster level of detail (mirror of dust::lod_*: DUST_PYR_LOD) ───
// dust_lod_px pixels per drawn segment. A polyline whose bins are shorter than that on screen is
// drawn as chord groups of 2, 4, 8 or 16 bins (the smallest power of two whose group chord is at
// least the knob; 16 = one chord along the whole polyline); the time pieces are capped at
// ceil(T / knob) with T the time segment on screen; the sweeps step by the knob. Points behind the
// camera keep full detail. The depth rule: a group chord or a time piece may not span more depth
// than the split tolerates for one Jacobian (DUST_SPLIT_EPS x the depth of the polyline's nearest
// edge), so the level of detail never makes a packet the split would have to bisect.
// the frame's pixels per drawn segment (0 = off) of the header word (dust::lod_px)
float dust_lod_px_of(DustUintBuffer pyr) {
    float v = uintBitsToFloat(pyr.v[DUST_PYR_LOD]);
    return (!isnan(v) && !isinf(v) && v > 0.0) ? v : 0.0;
}
// the polyline's depth (dust::polyline_depth): the depth span per bin and the depth of its nearest
// edge along fwd from eye over its 17 edges; false when an edge is at or behind the eye (full
// detail, as behind the camera)
bool dust_polyline_depth(uint mi, vec3 eye, vec3 fwd, out float binDepth, out float near) {
    float lo = 0.0, hi = 0.0;
    for (uint j = 0u; j < DUST_SIZE_EDGES; ++j) {
        float d = dot(dust_mom.m[mi].edges[j].xyz - eye, fwd);
        lo = j == 0u ? d : min(lo, d);
        hi = j == 0u ? d : max(hi, d);
    }
    binDepth = (hi - lo) / float(DUST_SIZE_BINS);
    near = lo;
    return lo > 0.0 && !isinf(hi) && !isnan(hi);
}
// the largest chord group (a power of two <= 16) whose chord spans no more depth than the split
// tolerates, g * binDepth <= DUST_SPLIT_EPS * near (dust::lod_group_depth_cap); 16 without a depth
// (orthographic), 1 when even a bin is too deep
uint dust_lod_group_depth_cap(bool hasDepth, float binDepth, float near) {
    if (!hasDepth) return DUST_SIZE_BINS;
    float lim = DUST_SPLIT_EPS * near;
    uint g = 1u;
    while (g < DUST_SIZE_BINS && float(2u * g) * binDepth <= lim) g *= 2u;
    return g;
}
// the fewest time pieces whose depth span the split tolerates, ceil(span / (DUST_SPLIT_EPS * near))
// for a time segment of depth span `span` (dust::time_pieces_depth); 1 without a depth
uint dust_time_pieces_depth(bool hasSpan, float span, bool hasDepth, float near) {
    if (!(hasSpan && hasDepth && near > 0.0)) return 1u;
    return clamp(uint(min(ceil(abs(span) / (DUST_SPLIT_EPS * near)), 1.0e6)), 1u, DUST_TIME_PIECES_MAX);
}
// the polyline's on-screen half-extent (dust::polyline_extent_px): the largest distance of its 17
// edges from its mean, in pixels; false when a point is behind the camera
bool dust_polyline_extent_px(uint mi, mat4 mvp, float W, float H, out float ext) {
    vec2 c, p;
    if (!dust_project_point(dust_mom.m[mi].mean_flux.xyz, mvp, W, H, c)) return false;
    ext = 0.0;
    for (uint j = 0u; j < DUST_SIZE_EDGES; ++j) {
        if (!dust_project_point(dust_mom.m[mi].edges[j].xyz, mvp, W, H, p)) return false;
        vec2 d = p - c;
        ext = max(ext, sqrt(d.x * d.x + d.y * d.y));
    }
    return !isinf(ext) && !isnan(ext);
}
// the time segment on screen (dust::time_segment_px): the distance between the means of the
// cluster and its predecessor, in pixels; false behind the camera
bool dust_time_segment_px(uint mi, uint pi, mat4 mvp, float W, float H, out float t) {
    vec2 a, c;
    if (!dust_project_point(dust_mom.m[mi].mean_flux.xyz, mvp, W, H, a)) return false;
    if (!dust_project_point(dust_mom.m[pi].mean_flux.xyz, mvp, W, H, c)) return false;
    vec2 d = c - a;
    t = sqrt(d.x * d.x + d.y * d.y);
    return !isinf(t) && !isnan(t);
}
// the chord group of a polyline of half-extent ext on screen (dust::lod_group): 1 when off,
// unknown (behind the camera) or when a bin is already lodPx long; else the smallest of 2, 4, 8,
// 16 whose group chord g * 2 ext / 16 reaches lodPx (16 when none), never more than depthCap
uint dust_lod_group(bool hasExt, float ext, float lodPx, uint depthCap) {
    if (!hasExt || !(lodPx > 0.0)) return 1u;
    float binPx = 2.0 * ext / float(DUST_SIZE_BINS);
    if (binPx >= lodPx) return 1u;
    uint g = 2u;
    while (g < DUST_SIZE_BINS && float(g) * binPx < lodPx) g *= 2u;
    return min(g, max(depthCap, 1u));
}
// the time pieces after the level of detail (dust::time_pieces_lod): nT when off or unknown, else
// clamp(ceil(tPx / lodPx), 1, nT), and at least depthMin of them
uint dust_time_pieces_lod(uint nT, bool hasT, float tPx, float lodPx, uint depthMin) {
    if (!(hasT && lodPx > 0.0)) return nT;
    uint c = uint(min(ceil(tPx / lodPx), 1.0e6));
    return clamp(max(c, depthMin), 1u, max(nT, 1u));
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
