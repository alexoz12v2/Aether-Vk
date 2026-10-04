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

const uint  DUST_CHILDREN          = 8u;
const uint  DUST_KEPLER_ITERS_F32  = 32u;
const uint  DUST_KEPLER_ITERS_DF   = 4u;
const float DUST_KEPLER_TOL_F32    = 2.4e-7;
const float DUST_KEPLER_TOL_DF     = 1.42e-14;
const float DUST_CHILD_SIGMA_V_REL = 0.05;
const float DUST_PI_F32            = 3.14159265;

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
};

// 32 B: per-frame evaluation, written compactly (index = live-range offset)
struct DustRenderCluster {
    vec4 pos_size;           // particle-system local position (m), w spread radius (m)
    vec4 age_id_dbeta_flux;  // x age (s), y ring slot (uint bits), z child beta half-spread, w flux (m^2, 0 = culled)
};

// 192 B: emission batch
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
    vec4  mass_params;      // batch mass (g), density (g/cm^3), low-discrepancy shift, 0
    uint  first_index;
    uint  count;
    uint  ring_mask;
    uint  seed;
    vec4  lit;              // jet site illumination: psi_start, psi0, total lit phase (rad), mode
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer DustClusterBuffer {
    DustCluster c[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) buffer DustRenderBuffer {
    DustRenderCluster c[];
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

// fract(j / phi + shift) in 32-bit fixed point (mirror of `dust::lattice_u01`)
float dust_lattice_u01(uint j, float shift) {
    uint shift_u = uint(shift * 16777216.0) << 8u;
    return dust_u01(j * 0x9E3779B9u + shift_u);
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
