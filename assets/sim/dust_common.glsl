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
const uint  DUST_MAX_CHILDREN      = 1024u;
const uint  DUST_LOD_CLUSTER_BITS  = 22u;
const uint  DUST_LOD_CLUSTER_MASK  = (1u << 22u) - 1u;
// stable child-pattern id in the low mantissa bits of the beta half-spread (dust::CHILD_ID_BITS)
const uint  DUST_CHILD_ID_MASK     = (1u << 16u) - 1u;
const float DUST_CHILD_RESHAPE_TAU = 3600.0;  // dust::CHILD_RESHAPE_TAU_S
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
const float DUST_STREAM_SIZE_JITTER  = 0.05;
const float DUST_STREAM_TIME_JITTER  = 0.1;
const float DUST_STREAM_SPEED_JITTER = 0.1;
const uint  DUST_STREAM_BREAK_BIT    = 0x80000000u;  // sign bit of the beta half-spread field
const float DUST_STREAM_BREAK_TURNS  = 0.01;
const uint  DUST_BATCH_BREAK_FLAG    = 16u;          // mass_params.w = shift + 16 * break
const uint  DUST_BATCH_PROVISIONAL_FLAG = 32u;       //   + 32 * provisional (last sample: now)
const uint  DUST_BATCH_SIZE_ROTATION_UNIT = 64u;    //   + 64 * size rotation (dust::stream_stratum)
const float DUST_STREAK_MARGIN       = 0.05;
const float DUST_STREAK_MARGIN_MAX   = 8.0;
const float DUST_STREAK_PHI_INV      = 0.618034;
// View aids (mirror of dust::DUST_VIEW_* / TRACER_* / FLOW_* / LOD_HEADER_*): tracers are bright
// dots on a sparse stable set of real particles; flow pulses are synchrone bands travelling outward
const uint  DUST_VIEW_TRACERS        = 1u;
const uint  DUST_VIEW_FLOW           = 2u;
const uint  DUST_TRACER_EVERY        = 256u;
const uint  DUST_TRACER_CHILD        = 1023u;
const float DUST_TRACER_PX           = 2.0;
const float DUST_TRACER_LEVEL        = 0.5;
const float DUST_FLOW_SHARE          = 0.85;
const float DUST_FLOW_KAPPA          = 4.0;
const float DUST_FLOW_I0_KAPPA       = 11.301922;
const uint  DUST_LOD_HEADER_LIST     = 6u;  // u64 (2 words), written by dust_lod.comp
const uint  DUST_LOD_HEADER_FLOW_SPEED = 8u; // the host's flow uniform (dust::DustFlowUniform):
const uint  DUST_LOD_HEADER_FLAGS    = 9u;  //   time-lapse factor K (diagnostics), DUST_VIEW_* flags,
const uint  DUST_LOD_HEADER_T_HI     = 10u; //   flow clock T hi / lo (exact emission epochs)
const uint  DUST_LOD_HEADER_T_LO     = 11u;
const uint  DUST_LOD_HEADER_LAMBDA   = 13u; // this frame's budget share (pass B, for dust.vert)
const uint  DUST_LOD_HEADER_ON_SCREEN = 14u; // clusters on screen (pass A)
const uint  DUST_LOD_HEADER_SUN_G    = 15u; // solar gravity at the jet (host): footprint beta extent
const uint  DUST_LOD_HEADER_HIST     = 16u; // demand histogram: count, sum per bin (pass A)
const uint  DUST_LOD_HIST_BINS       = 21u;
const float DUST_LOD_DEMAND_PASS     = -1.0; // DustLodPushConstants::lambda of pass A
const float DUST_LOD_TARGET_FILL     = 0.9;
const float DUST_CHILD_PX_MAX        = 12.0;
// render cluster y word: ring slot | stream shift << 27 | live << 31 (dust::render_word)
const uint  DUST_RENDER_SLOT_MASK    = (1u << 22u) - 1u;
const uint  DUST_RENDER_SHIFT_BIT0   = 27u;
const uint  DUST_RENDER_LIVE_BIT     = 0x80000000u;

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
    vec4 misc;        // x sigma_v (m/s), y grain radius (um), z cross-section per gram (m^2/g), w child beta half-spread
                      // (low 16 bits: child-pattern id, dust::child_id)
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
    if (m < seg_len) {
        psi = seg_start + m;
    } else {
        precise float mr = m - seg_len;
        precise float k = floor(mr / arc);
        precise float r = mr - k * arc;
        psi = arc_start + TWO_PI + k * TWO_PI + r;
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

// 3 independent N(0,1) from the hash chain started at h0 (mirror of `dust::gauss3`); h4 = last hash
vec3 dust_gauss3(uint h0, out uint h4) {
    uint h1 = dust_pcg(h0);
    uint h2 = dust_pcg(h1);
    uint h3 = dust_pcg(h2);
    h4 = dust_pcg(h3);
    return vec3(
        dust_gauss(dust_u01(h0), dust_u01(h1)),
        dust_gauss(dust_u01(h1 ^ 0x68E31DA4u), dust_u01(h2)),
        dust_gauss(dust_u01(h3), dust_u01(h4))
    );
}

// Child-pattern id of a cluster from its beta half-spread field (mirror of `dust::child_id`)
uint dust_child_id(float dbetaField) { return floatBitsToUint(dbetaField) & DUST_CHILD_ID_MASK; }

// Offset of child `child` of the cluster with child-pattern id `id` from the cluster centre (mirror
// of `dust::child_offset`):
//   spread * (cos(th) N3 + sin(th) N3')   ejection velocity dispersion; th = rate * pi/2 *
//                                          log2(1 + age / tau) reshapes the cloud as it ages
//                                          (per-axis variance stays spread^2)
// + 1/2 dbeta g age^2 anti-sun             size spread within the cluster's beta stratum
// Depends only on (id, child): children 0..k keep their place when the LOD changes k, and the id
// comes from the emission record, so a seek draws the same children as playback.
vec3 dust_child_offset(uint id, uint child, float spread, float dbetaHalf, float age, vec4 antiSunG) {
    uint h4, m4;
    vec3 n3 = dust_gauss3(dust_pcg((id & DUST_CHILD_ID_MASK) * DUST_MAX_CHILDREN + child + 0x9E3779B9u), h4);
    uint h5 = dust_pcg(h4);
    float dbeta = abs(dbetaHalf) * (2.0 * dust_u01(h5) - 1.0);
    vec3 m3 = dust_gauss3(dust_pcg(h5 ^ 0x5BD1E995u), m4);
    float rate = 0.5 + dust_u01(dust_pcg(m4));
    float th = rate * (0.5 * DUST_PI_F32 * 1.4426950408889634) * log(1.0 + max(age, 0.0) * (1.0 / DUST_CHILD_RESHAPE_TAU));
    return spread * (cos(th) * n3 + sin(th) * m3) + (0.5 * dbeta * antiSunG.w * age * age) * antiSunG.xyz;
}

// ─── Streams (mirror of dust::batch_streams / stream_* / size_stratum_mass) ───
// (stream shift, break before the batch) of a batch
uvec2 dust_batch_streams(DustBatch B) {
    uint w = uint(B.mass_params.w);
    return uvec2(w % DUST_BATCH_BREAK_FLAG, (w / DUST_BATCH_BREAK_FLAG) & 1u);
}

// Size stratum of cluster j = i*S + s: the stream's keyed rank advanced by one per time sample of
// the tier's grid, (rank + rotation + i) mod S (dust::stream_stratum)
uint dust_stream_stratum(DustBatch B, uint j) {
    uint shift = dust_batch_streams(B).x;
    uint n = 1u << shift;
    uint key = uint(B.mass_params.z * 16777216.0);
    uint rank = dust_permute(j & (n - 1u), n, dust_pcg(key ^ 0x2545F491u));
    return (rank + uint(B.mass_params.w) / DUST_BATCH_SIZE_ROTATION_UNIT + (j >> shift)) & (n - 1u);
}

// Emission-time quantile of cluster j = i*S + s: time stratum i, a jitter shared by the streams of
// the sample plus DUST_STREAM_TIME_JITTER of the stratum per cluster
float dust_stream_time_u01(DustBatch B, uint j) {
    uint shift = dust_batch_streams(B).x;
    uint samples = max(B.count >> shift, 1u);
    uint i = j >> shift;
    // the provisional batch's last sample: the window end, now (dust::BATCH_PROVISIONAL_FLAG)
    if (i + 1u == samples && ((uint(B.mass_params.w) / DUST_BATCH_PROVISIONAL_FLAG) & 1u) != 0u) return 1.0;
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
    return dark > 2.0 * DUST_PI_F32 * DUST_STREAM_BREAK_TURNS;
}

// Whether the stream of cluster j is interrupted before it: missing previous window, size wrap
// to the smallest stratum, or a dark site (dust::stream_breaks_before)
bool dust_stream_breaks_before(DustBatch B, uint j) {
    uvec2 st = dust_batch_streams(B);
    if ((j >> st.x) == 0u && st.y != 0u) return true;
    if (st.x > 0u && dust_stream_stratum(B, j) == 0u) return true;
    return dust_stream_dark_before(B, j);
}

// Mass fraction of the size stratum [p/n, (p+1)/n] of the log-uniform size quantile
float dust_size_stratum_mass(vec4 sizeParams, uint p, uint n) {
    n = max(n, 1u);
    float r = sizeParams.y / sizeParams.x;
    float e = sizeParams.z;
    float full = pow(r, e) - 1.0;
    if (!(abs(full) > 1e-6)) return 1.0 / float(n);
    float u0 = float(p) / float(n), u1 = float(p + 1u) / float(n);
    return (pow(r, e * u1) - pow(r, e * u0)) / full;
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

// Footprint extent (m) of a render cluster for the LOD: its stream spread or, larger, the beta
// spread of its size stratum 1/2 dbeta g age^2 that its children are scattered over (mirror of
// dust::dust_extent)
float dust_extent(DustRenderCluster C, float sunG) {
    return max(C.pos_size.w, 0.5 * abs(C.age_id_dbeta_flux.z) * sunG * C.age_id_dbeta_flux.x * C.age_id_dbeta_flux.x);
}

// One Liang-Barsky plane f0 + t fd >= 0 on [a, b]; false when nothing is left
bool dust_lb(float f0, float fd, inout float a, inout float b) {
    if (fd == 0.0) return f0 >= 0.0;
    float t = -f0 / fd;
    if (fd > 0.0) { if (t > a) a = t; }
    else if (t < b) b = t;
    return a <= b;
}

// [a, b] snapped outwards to a power-of-two grid of step in (len/16, len/8] (exact, bit ops)
vec2 dust_snap_range(float a, float b) {
    float len = max(b - a, 1e-30);
    uint bits = floatBitsToUint(len);
    int cl = int((bits >> 23u) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
    int e = clamp(cl - 3, -126, 0);
    float step = uintBitsToFloat(uint(e + 127) << 23u);
    return vec2(max(floor(a / step) * step, 0.0), min(-floor(-b / step) * step, 1.0));
}

// Visible part of the streak p -> q (lateral sigma sp at p, sq at q): Liang-Barsky in homogeneous
// clip space against w > 0 and |x|, |y| <= (1 + m) w, m = margin + 3 sigma (ndc). Out:
// x, y the snapped parameter range, z the on-screen length (px), w the lateral width (px).
bool dust_streak_clip(mat4 mvp, vec3 p, vec3 q, float sp, float sq, vec4 params, out vec4 S) {
    S = vec4(0.0);
    float units = params.x, p00 = params.y, p11 = params.z, pxn = abs(params.w);
    if (!(p00 > 0.0) || !(p11 > 0.0) || !(pxn > 0.0)) return false;
    vec4 c0 = mvp * vec4(p, 1.0);
    vec4 c1 = mvp * vec4(q, 1.0);
    vec4 d = c1 - c0;
    float a = 0.0, b = 1.0;
    float eps = 1e-6 * (abs(c0.w) + abs(c1.w));
    if (!dust_lb(c0.w - eps, d.w, a, b)) return false;
    float wNear = min(c0.w + a * d.w, c0.w + b * d.w);
    if (!(wNear > 0.0)) return false;
    float sigma = max(sp, sq) * units;
    float m = 1.0 + min(DUST_STREAK_MARGIN + 3.0 * sigma * max(p00, p11) / wNear, DUST_STREAK_MARGIN_MAX);
    for (int k = 0; k < 2; ++k) {
        if (!dust_lb(m * c0.w - c0[k], m * d.w - d[k], a, b)) return false;
        if (!dust_lb(m * c0.w + c0[k], m * d.w + d[k], a, b)) return false;
    }
    float wa = c0.w + a * d.w, wb = c0.w + b * d.w;
    float pxX = pxn * (p00 / p11);
    float dx = (c0.x + b * d.x) / wb - (c0.x + a * d.x) / wa;
    float dy = (c0.y + b * d.y) / wb - (c0.y + a * d.y) / wa;
    float lenPx = sqrt((dx / pxX) * (dx / pxX) + (dy / pxn) * (dy / pxn));
    float widthPx = min(sigma * p11 / min(wa, wb) / pxn, 2.0 / pxn);
    S = vec4(dust_snap_range(a, b), lenPx, widthPx);
    return true;
}

// Children a streak asks for: DUST_CHILD_PX dots along its visible length, times its lateral width
// in dots (mirror of dust::streak_want)
float dust_streak_want(float lenPx, float widthPx) {
    return max(max(lenPx, widthPx) / DUST_CHILD_PX, 1.0) * max(widthPx / DUST_CHILD_PX, 1.0);
}

// Parameter of streak dot `child` on [t0, t1] (mirror of dust::streak_dot_t)
float dust_streak_dot_t(float t0, float t1, uint id, uint child) {
    precise float x = dust_u01(dust_pcg((id & DUST_CHILD_ID_MASK) ^ 0x1B873593u)) + float(child) * DUST_STREAK_PHI_INV;
    return t0 + (t1 - t0) * (x - floor(x));
}

// Child `child` of the streak p -> q on [t0, t1] (mirror of dust::streak_child): at
// t = t0 + (t1 - t0) fract(u_id + child phi^-1) (any prefix 0..k evenly spread, stable when k
// changes), lateral gaussian of sigma lerp(sp, sq, t) perpendicular to the streak, plus the
// beta-stratum term at age lerp(ap, aq, t)
vec3 dust_streak_child(vec3 p, vec3 q, float sp, float sq, float ap, float aq, float t0, float t1,
                       uint id, uint child, float dbetaField, vec4 antiSunG) {
    id &= DUST_CHILD_ID_MASK;
    float t = dust_streak_dot_t(t0, t1, id, child);
    vec3 d = q - p;
    float dl = length(d);
    vec3 dn = dl > 0.0 ? d / dl : d;
    vec3 up = abs(dn.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 c = cross(up, dn);
    float cl = length(c);
    vec3 e1 = cl > 0.0 ? c / cl : c;
    vec3 e2 = cross(dn, e1);
    uint h0 = dust_pcg((id * DUST_MAX_CHILDREN + child) ^ 0x7F4A7C15u);
    uint h1 = dust_pcg(h0);
    uint h2 = dust_pcg(h1);
    uint h3 = dust_pcg(h2);
    uint h4 = dust_pcg(h3);
    float n1 = dust_gauss(dust_u01(h0), dust_u01(h1));
    float n2 = dust_gauss(dust_u01(h2), dust_u01(h3));
    float sigma = sp + (sq - sp) * t;
    float age = ap + (aq - ap) * t;
    float dbeta = abs(dbetaField) * (2.0 * dust_u01(h4) - 1.0);
    return p + d * t + sigma * (n1 * e1 + n2 * e2) + (0.5 * dbeta * antiSunG.w * age * age) * antiSunG.xyz;
}

// Children wanted by a cluster whose spread covers `spreadPx` pixels (mirror of dust::lod_want)
float dust_lod_want(float spreadPx) {
    float r = spreadPx / DUST_CHILD_PX;
    return max(r * r, 1.0);
}

// Children drawn under the budget share `lambda`, below the tracer's child index (mirror of
// dust::lod_children)
uint dust_lod_children(float want, float lambda) {
    return uint(clamp(floor(lambda * want + 0.5), 1.0, float(DUST_TRACER_CHILD)));
}

// A cluster's demand: ceil(want), at most 2^20 (mirror of dust::lod_demand)
uint dust_lod_demand(float want) { return clamp(uint(ceil(want)), 1u, 1u << 20u); }

// Demand histogram bin: floor(log2(d)), d = dust_lod_demand (mirror of dust::lod_hist_bin)
uint dust_lod_hist_bin(uint d) { return min(uint(findMSB(max(d, 1u))), DUST_LOD_HIST_BINS - 1u); }
// d / 2^bin in 1/256 (mirror of dust::lod_hist_add)
uint dust_lod_hist_add(uint d) { return (max(d, 1u) << 8u) >> dust_lod_hist_bin(d); }

// Cost of share l: sum over bins of count * max(1, l * mean demand) (dust::lod_lambda_from)
float dust_lod_cost(DustUintBuffer header, float l) {
    float c = 0.0;
    for (uint b = 0u; b < DUST_LOD_HIST_BINS; ++b) {
        float n = float(header.v[DUST_LOD_HEADER_HIST + 2u * b]);
        float s = float(header.v[DUST_LOD_HEADER_HIST + 2u * b + 1u]);
        if (n > 0.0) {
            float mean = s / n * (uintBitsToFloat((b + 127u) << 23u) * (1.0 / 256.0));
            c += n * clamp(l * mean, 1.0, float(DUST_TRACER_CHILD));
        }
    }
    return c;
}

// This frame's budget share from this frame's demand histogram: bisection in log space of
// sum count * max(1, l * mean) = fill * budget (mirror of dust::lod_lambda_from)
float dust_lod_lambda(DustUintBuffer header, uint budget, uint onScreen, float lambdaMax) {
    float lmax = max(lambdaMax, 1e-6);
    float target = DUST_LOD_TARGET_FILL * float(budget) - float(onScreen / DUST_TRACER_EVERY);
    if (dust_lod_cost(header, lmax) <= target) return lmax;
    float lo = -19.93, hi = log(lmax) * 1.4426950408889634;
    for (int it = 0; it < 24; ++it) {
        float mid = 0.5 * (lo + hi);
        if (dust_lod_cost(header, exp(mid * 0.6931471805599453)) <= target) lo = mid; else hi = mid;
    }
    return clamp(exp(lo * 0.6931471805599453), 1e-6, lmax);
}

// Drawn radius (px) of k children asking for want: they keep covering the footprint, softer when
// fewer (mirror of dust::splat_radius_px)
float dust_splat_radius(float want, uint k) {
    return min(DUST_CHILD_PX * sqrt(max(want / float(max(k, 1u)), 1.0)), DUST_CHILD_PX_MAX);
}

// The cluster with child-pattern id `id` carries a tracer dot (mirror of dust::is_tracer)
bool dust_is_tracer(uint id) {
    return dust_pcg((id & DUST_CHILD_ID_MASK) ^ 0x7AC312E5u) % DUST_TRACER_EVERY == 0u;
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
