// @assets/sim/dust_kepler_lr.glsl
//
// Round 9 step O: dust_kepler_coef (dust_common.glsl) at a small register budget: the same solve,
// expression for expression and in the same order of evaluation per value, with its df64 state in
// the lane's stash (memory words, the including shader's kp_put/kp_get) instead of registers. A
// lone solve kept r0n, eta, zeta, alpha, t, sd and the four Stumpff values (each multiplied value
// also with its Dekker split) live across the iteration: 60+ registers on Mali-G52 and GCN. Here
// every value is written to the stash once, when it is made, and re-read where it is used (the
// same bits); registers hold only the operands of the df64 operation at hand and a running
// partial sum: ~24 registers. The cost is memory traffic (the stash words, see KP_WORDS).
// Stumpff: dust_stumpff_df_lr, the same series and doubling steps in a lower-pressure order.
// Keep in step with dust_kepler: a solver change touches dust_kepler, dust_kepler_coef and this file.
//
// The including shader defines first (each re-evaluated at each use; a reload fence inside keeps
// the compilers from merging the reloads into one long-lived value):
//   vec2 kp_r0c(uint k)  vec2 kp_v0c(uint k)   component k of r0 / v0 as (hi, lo), k = 0..2 (v0 with its
//                                              velocity offset if any, as dust_kepler receives it)
//   vec2 kp_mu()  vec2 kp_dt()
//   void kp_put(uint w, float x)  float kp_get(uint w)     w = 0..KP_WORDS-1 (the lane's stash)
// and gets dust_kepler_coef_lr() (false: still, r = r0, v = v0) with the coefficients in the stash,
// and kp_apply(k, KP_FF, KP_GG) / kp_apply(k, KP_FD, KP_GD) = component k of r / v.
#ifndef DUST_KEPLER_LR_GLSL
#define DUST_KEPLER_LR_GLSL

// the dot-product loops are not unrolled, so that no compiler hoists all the reloads into one cluster
#ifndef KP_LOOP
#define KP_LOOP [[dont_unroll]]
#endif

// dust_stumpff_df with c0 made right after the c2 series and the four doubling products in the
// order n3, n2, n1, n0 (independent expressions; each one exactly dust_stumpff_df's)
void dust_stumpff_df_lr(vec2 x, out vec2 c0, out vec2 c1, out vec2 c2, out vec2 c3) {
    vec2 xr = x;
    uint n = 0u;
    while (abs(xr.x) > 0.1 && n < 64u) { xr = df_scale_pow2(xr, 0.25); n++; }
    c2 = df_sub(DF_INV_FACT[12], df_mul(xr, DF_INV_FACT[14]));
    c2 = df_sub(DF_INV_FACT[10], df_mul(xr, c2));
    c2 = df_sub(DF_INV_FACT[8], df_mul(xr, c2));
    c2 = df_sub(DF_INV_FACT[6], df_mul(xr, c2));
    c2 = df_sub(DF_INV_FACT[4], df_mul(xr, c2));
    c2 = df_sub(DF_INV_FACT[2], df_mul(xr, c2));
    c0 = df_sub(vec2(1.0, 0.0), df_mul(xr, c2));
    c3 = df_sub(DF_INV_FACT[13], df_mul(xr, DF_INV_FACT[15]));
    c3 = df_sub(DF_INV_FACT[11], df_mul(xr, c3));
    c3 = df_sub(DF_INV_FACT[9], df_mul(xr, c3));
    c3 = df_sub(DF_INV_FACT[7], df_mul(xr, c3));
    c3 = df_sub(DF_INV_FACT[5], df_mul(xr, c3));
    c3 = df_sub(DF_INV_FACT[3], df_mul(xr, c3));
    c1 = df_sub(vec2(1.0, 0.0), df_mul(xr, c3));
    for (uint i = 0u; i < n; i++) {
        vec2 n3 = df_scale_pow2(df_add(c3, df_mul(c1, c2)), 0.25);
        vec2 n2 = df_scale_pow2(df_mul(c1, c1), 0.5);
        vec2 n1 = df_mul(c0, c1);
        vec2 n0 = df_add_f(df_scale_pow2(df_mul(c0, c0), 2.0), -1.0);
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
}

// the stash words: t, alpha, the iterate sd, the round's Stumpff values (of c0 only the hi word is
// read), r0n, eta, zeta, and the round's f, f', f''
const uint KP_T = 0u, KP_ALPHA = 2u, KP_SD = 4u, KP_C0 = 6u, KP_C1 = 7u, KP_C2 = 9u, KP_C3 = 11u;
const uint KP_R0N = 13u, KP_ETA = 15u, KP_ZETA = 17u, KP_F = 19u, KP_FP = 20u, KP_FPP = 21u;
const uint KP_WORDS = 22u;
vec2 kp_get2(uint w) { return vec2(kp_get(w), kp_get(w + 1u)); }
void kp_put2(uint w, vec2 x) { kp_put(w, x.x); kp_put(w + 1u, x.y); }
// dust_kepler_coef's df3_dot(r0, r0), df3_dot(r0, v0), df3_dot(v0, v0): (a0 b0 + a1 b1) + a2 b2,
// one component pair per loop iteration (the same operations in the same order)
vec2 kp_r0n() {
    vec2 a = kp_r0c(0u);
    vec2 d = df_mul(a, a);
    KP_LOOP for (uint k = 1u; k < 3u; k++) { a = kp_r0c(k); d = df_add(d, df_mul(a, a)); }
    return df_sqrt(d);
}
vec2 kp_eta() {
    vec2 d = df_mul(kp_r0c(0u), kp_v0c(0u));
    KP_LOOP for (uint k = 1u; k < 3u; k++) d = df_add(d, df_mul(kp_r0c(k), kp_v0c(k)));
    return d;
}
vec2 kp_v2() {
    vec2 a = kp_v0c(0u);
    vec2 d = df_mul(a, a);
    KP_LOOP for (uint k = 1u; k < 3u; k++) { a = kp_v0c(k); d = df_add(d, df_mul(a, a)); }
    return d;
}
vec2 kp_sd() { return kp_get2(KP_SD); }
vec2 kp_s2() { return df_mul(kp_sd(), kp_sd()); }
// psi = (alpha sd) sd -> the stash's c0.x, c1, c2, c3
void kp_stumpff_put() {
    vec2 c0, c1, c2, c3;
    dust_stumpff_df_lr(df_mul(df_mul(kp_get2(KP_ALPHA), kp_sd()), kp_sd()), c0, c1, c2, c3);
    kp_put(KP_C0, c0.x); kp_put2(KP_C1, c1); kp_put2(KP_C2, c2); kp_put2(KP_C3, c3);
}

// dust_kepler_coef in three parts, the state between them in the stash
// prep: r0n, eta, alpha, zeta, t (periods removed) and the f32 phase's sd; false = still (r0n = 0
// or dt = 0: dust_kepler_coef returns there)
bool kp_solve_prep() {
    kp_put2(KP_R0N, kp_r0n());
    if (!(kp_get(KP_R0N) > 0.0) || kp_dt().x == 0.0) return false;
    kp_put2(KP_ETA, kp_eta());
    kp_put2(KP_ALPHA, df_sub(df_div(df_scale_pow2(kp_mu(), 2.0), kp_get2(KP_R0N)), kp_v2())); // = mu / a
    kp_put2(KP_ZETA, df_sub(kp_mu(), df_mul(kp_get2(KP_ALPHA), kp_get2(KP_R0N))));
    // bound elliptic arguments: remove whole periods
    {
        vec2 t = kp_dt();
        vec2 mu = kp_mu(), alpha = kp_get2(KP_ALPHA);
        if (mu.x > 0.0 && alpha.x > 0.0) {
            vec2 period = df_div(df_mul(DF_TWO_PI, mu), df_mul(alpha, df_sqrt(alpha)));
            vec2 k = df_div(t, period);
            k = k.x >= 0.0 ? df_floor(k) : -df_floor(-k);
            t = df_sub(t, df_mul(k, period));
        }
        kp_put2(KP_T, t);
    }
    // phase 1: f32 Laguerre from s = t / r0
    {
        float r0f = kp_get(KP_R0N), etaf = kp_get(KP_ETA), zetaf = kp_get(KP_ZETA), alphaf = kp_get(KP_ALPHA);
        float tf = kp_get(KP_T) + kp_get(KP_T + 1u);
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
        kp_put2(KP_SD, vec2(s, 0.0));
    }
    return true;
}
// phase 2: df64 residual, f32 correction: f = ((r0n sd + (eta s2) c2) + ((zeta s2) sd) c3) - t,
// f' = (r0n + (eta sd) c1) + (zeta s2) c2, f'' = eta c0 + zeta sd c1 (dust_kepler_coef's
// expressions), each result into the stash as soon as it is made
#ifndef DUST_KEPLER_ITERS_DF_LR
#define DUST_KEPLER_ITERS_DF_LR DUST_KEPLER_ITERS_DF
#endif
void kp_solve_df() {
    for (uint it = 0u; it < DUST_KEPLER_ITERS_DF_LR; it++) {
        kp_stumpff_put();
        kp_put(KP_FPP, kp_get(KP_ETA) * kp_get(KP_C0) + kp_get(KP_ZETA) * kp_sd().x * kp_get(KP_C1));
        kp_put(KP_F, df_sub(df_add(df_add(df_mul(kp_get2(KP_R0N), kp_sd()),
                                          df_mul(df_mul(kp_get2(KP_ETA), kp_s2()), kp_get2(KP_C2))),
                                   df_mul(df_mul(df_mul(kp_get2(KP_ZETA), kp_s2()), kp_sd()), kp_get2(KP_C3))),
                            kp_get2(KP_T)).x);
        kp_put(KP_FP, df_add(df_add(kp_get2(KP_R0N), df_mul(df_mul(kp_get2(KP_ETA), kp_sd()), kp_get2(KP_C1))),
                             df_mul(df_mul(kp_get2(KP_ZETA), kp_s2()), kp_get2(KP_C2))).x);
        float ds;
        if (!dust_laguerre(kp_get(KP_F), kp_get(KP_FP), kp_get(KP_FPP), ds)) break;
        vec2 sd = df_add_f(kp_sd(), -ds);
        kp_put2(KP_SD, sd);
        if (!(abs(ds) > DUST_KEPLER_TOL_DF * abs(sd.x))) break;
    }
}
// kp_solve_df's loop body after kp_stumpff_put (a solve in passes): true = the loop is over
// (dust_laguerre failed: sd unchanged; or the step converged: sd updated)
bool kp_solve_df_step() {
    kp_put(KP_FPP, kp_get(KP_ETA) * kp_get(KP_C0) + kp_get(KP_ZETA) * kp_sd().x * kp_get(KP_C1));
    kp_put(KP_F, df_sub(df_add(df_add(df_mul(kp_get2(KP_R0N), kp_sd()),
                                      df_mul(df_mul(kp_get2(KP_ETA), kp_s2()), kp_get2(KP_C2))),
                               df_mul(df_mul(df_mul(kp_get2(KP_ZETA), kp_s2()), kp_sd()), kp_get2(KP_C3))),
                        kp_get2(KP_T)).x);
    kp_put(KP_FP, df_add(df_add(kp_get2(KP_R0N), df_mul(df_mul(kp_get2(KP_ETA), kp_sd()), kp_get2(KP_C1))),
                         df_mul(df_mul(kp_get2(KP_ZETA), kp_s2()), kp_get2(KP_C2))).x);
    float ds;
    if (!dust_laguerre(kp_get(KP_F), kp_get(KP_FP), kp_get(KP_FPP), ds)) return true;
    vec2 sd = df_add_f(kp_sd(), -ds);
    kp_put2(KP_SD, sd);
    return !(abs(ds) > DUST_KEPLER_TOL_DF * abs(sd.x));
}
// kp_solve_prep's result again from the stash (its early return's test on r0n, written first, and
// dt): valid while KP_R0N and dt stand
bool kp_moving() { return !(!(kp_get(KP_R0N) > 0.0) || kp_dt().x == 0.0); }
// final: the Lagrange coefficients (g1 = sd c1, g2 = s2 c2, g3 = (s2 sd) c3, recomputed where used),
// each written to the stash as soon as it is made, over a word no later expression reads: gg over
// c3, ff over f and f', fd over eta, gd over zeta (words 11-12, 19-20, 15-16, 17-18; words 0-10,
// 13-14 and 21 are free once the coefficients are made: a shader may write its results there while
// it still reads the coefficients)
const uint KP_FF = KP_F, KP_GG = KP_C3, KP_FD = KP_ETA, KP_GD = KP_ZETA;
void kp_solve_final_rest();
void kp_solve_final() {
    kp_stumpff_put();
    kp_solve_final_rest();
}
// kp_solve_final after its kp_stumpff_put (a pass of its own in a multi-pass solve)
void kp_solve_final_rest() {
    kp_put2(KP_GG, df_sub(kp_get2(KP_T), df_mul(kp_mu(), df_mul(df_mul(kp_s2(), kp_sd()), kp_get2(KP_C3)))));
    kp_put2(KP_FF, df_sub(vec2(1.0, 0.0), df_div(df_mul(kp_mu(), df_mul(kp_s2(), kp_get2(KP_C2))), kp_get2(KP_R0N))));
    vec2 rn = df_add(df_add(kp_get2(KP_R0N), df_mul(kp_get2(KP_ETA), df_mul(kp_sd(), kp_get2(KP_C1)))),
                     df_mul(kp_get2(KP_ZETA), df_mul(kp_s2(), kp_get2(KP_C2))));
    kp_put2(KP_FD, -df_div(df_mul(kp_mu(), df_mul(kp_sd(), kp_get2(KP_C1))), df_mul(rn, kp_get2(KP_R0N))));
    kp_put2(KP_GD, df_sub(vec2(1.0, 0.0), df_div(df_mul(kp_mu(), df_mul(kp_s2(), kp_get2(KP_C2))), rn)));
}
// dust_kepler_coef(kp_r0(), kp_v0(), kp_mu(), kp_dt(), still, ff, gg, fd, gd), bit for bit, the four
// coefficients left in the stash (KP_FF, KP_GG, KP_FD, KP_GD); false = still (r = r0, v = v0).
bool dust_kepler_coef_lr() {
    if (!kp_solve_prep()) return false;
    kp_solve_df();
    kp_solve_final();
    return true;
}

// dust_kepler_apply one component at a time from the stash's coefficients:
// r_k = r0_k ff + v0_k gg (kp_apply(k, KP_FF, KP_GG)), v_k = r0_k fd + v0_k gd (kp_apply(k, KP_FD, KP_GD))
vec2 kp_apply(uint k, uint wf, uint wg) { return df_add(df_mul(kp_r0c(k), kp_get2(wf)), df_mul(kp_v0c(k), kp_get2(wg))); }

#endif // DUST_KEPLER_LR_GLSL
