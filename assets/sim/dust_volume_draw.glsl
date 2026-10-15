// @assets/sim/dust_volume_draw.glsl
//
// The volumetric packet draw shared by dust_splat.comp (a packet drawn in its cluster's thread:
// the volume queue off, or full) and dust_draw_volume.comp (a lane per queued record): the same
// arithmetic on the same packet and seed, so the pyramid is the same word for word. Both shaders
// declare the same push-constant block `pc` (dust::DustSplatPushConstants) before including this,
// and set the per-packet globals it reads (dust_phot_support / _sun / _rh / _albedo / _lut,
// dust_pyr_split_word). Include after dust_scatter.glsl.

// whether this frame's in-frame gates are on (bit 22 of DUST_PYR_SPLIT_DEPTH clear; dust::frame_gates_on)
bool dust_frame_gates_on() {
    return ((pc.pyramid.v[DUST_PYR_SPLIT_DEPTH] >> 22u) & 1u) == 0u;
}

// Timing probes (bits 30-31 of DUST_PYR_SPLIT_DEPTH, host `AETHERVK_DUST_TIMING_PROBE`;
// diagnostics only, never set in production, no CPU mirror): 1 = plain_add (dust_atomic_add_sat),
// 2 = no_volume_writes (the volumetric texels are evaluated but not written: one add per packet
// into its level's first texel), 3 = no_volume (volumetric packets are dropped)
uint dust_probe() { return (dust_pyr_split_word >> 30u) & 3u; }

// the level a packet past an in-frame gate is drawn at least at (dust::gate_level)
uint dust_gate_level(uint levels) {
    return levels >= 3u ? levels - 3u : 0u;
}

// the texels of a piece's range at a level (dust::volume_evals, one piece)
uint dust_volume_texels(DustVolumePiece q, uint level, uint lw, uint lh) {
    ivec4 r = dust_volume_range(q, level, lw, lh);
    return uint(max(r.y - r.x + 1, 0) * max(r.w - r.z + 1, 0));
}

// Draws a packet by the exact ray integral of its density with its rounding seed already drawn
// (dust_volume_emit draws it, in the producer's order; dust::volume_scatter, where the
// derivation, the units and the choices are documented): per texel of one level, the optical
// depth along the ray from the eye through the texel centre,
// tau = k / sqrt(a) * exp(-|q' - (b/a) u'|^2 / 2) * Phi(-b / sqrt(a)), u' = L^-1 u, q' = L^-1 (eye -
// mean), a = |u'|^2, b = u'.q'; a chord as Gaussian pieces DUST_VOLUME_PIECE_SIGMAS apart; the
// EWA's pixel filter added at the packet's distance. The work is planned first
// (dust::volume_plan): the level from the angular sigma (>= DUST_VOLUME_MIN_LEVEL when a piece
// covers the frame) plus the frame's bias, coarsened until the evaluations (sum over the pieces
// of their texel ranges) fit DUST_VOLUME_MAX_EVALS, then the pieces merged in pairs; each piece is
// evaluated over its own range only, so the work is exactly the plan's (DUST_PYR_VOLUME_EVALS).
// The pieces are never kept in an array (profile 2026-10-15: a 16-piece local array indexed per
// piece was the plan's cost): pass A builds each piece in registers for its flags and its texels
// at the two levels the flags can lead to, the rare cap loops rebuild them, and pass B rebuilds
// each piece and draws it at once. The decisions use the same integers as the array did.
#define DUST_VOLUME_BUILD(q_, k_) \
    q_ = dust_volume_piece(k_, p.flux, p.rho, exposure, axMean, axS, segA, nA, cvarA, segB, nB, cvarB, \
                           hasB, W, H, r0, r1, r3, tv, eye);
#define DUST_VOLUME_EVALS_AT(out, level_) { \
    uint lw_ = pc.pyramid.v[DUST_PYR_TABLE + 3u * (level_) + 1u]; \
    uint lh_ = pc.pyramid.v[DUST_PYR_TABLE + 3u * (level_) + 2u]; \
    out = 0u; \
    for (uint k_ = 0u; k_ < nA * nB; ++k_) { \
        DustVolumePiece q_; \
        DUST_VOLUME_BUILD(q_, k_) \
        out += dust_volume_texels(q_, (level_), lw_, lh_); \
    } \
}

void dust_volume_draw(DustPacket p, uint seed, float exposure, vec3 color, uint levels, float W, float H, uint bias, uint axis) {
    DUST_PROF_BEGIN(vplan)
    vec3 r0 = vec3(pc.mvp[0][0], pc.mvp[1][0], pc.mvp[2][0]);
    vec3 r1 = vec3(pc.mvp[0][1], pc.mvp[1][1], pc.mvp[2][1]);
    vec3 r3 = vec3(pc.mvp[0][3], pc.mvp[1][3], pc.mvp[2][3]);
    vec3 tv = vec3(pc.mvp[3][0], pc.mvp[3][1], pc.mvp[3][3]);
    vec3 eye = pc.eyeLocal.xyz;
    DustVolumePlan pl;
    if (!dust_volume_plan_axes(p, r0, r1, r3, eye, W, H, levels, axis, pl)) { DUST_PROF_END(vplan) return; }
    vec3 axMean = pl.axMean;
    float axS[6] = pl.axS;
    vec3 segA = pl.segA, segB = pl.segB;
    bool hasB = pl.hasB;
    uint nA = pl.nA, nB = pl.nB;
    float cvarA = pl.cvarA, cvarB = pl.cvarB;
    uint level = pl.level;
    // pass A: the flags, and the texels at the level the plan gives and at the one a full piece
    // raises it to (both with the frame's bias)
    uint levelA = min(level + bias, levels - 1u);
    uint levelB = min(min(max(level, DUST_VOLUME_MIN_LEVEL), levels - 1u) + bias, levels - 1u);
    uint lwA = pc.pyramid.v[DUST_PYR_TABLE + 3u * levelA + 1u], lhA = pc.pyramid.v[DUST_PYR_TABLE + 3u * levelA + 2u];
    uint lwB = pc.pyramid.v[DUST_PYR_TABLE + 3u * levelB + 1u], lhB = pc.pyramid.v[DUST_PYR_TABLE + 3u * levelB + 2u];
    bool anyFull = false, anyOn = false;
    uint evalsA = 0u, evalsB = 0u;
    for (uint k = 0u; k < nA * nB; ++k) {
        DustVolumePiece q;
        DUST_VOLUME_BUILD(q, k)
        anyFull = anyFull || q.full;
        anyOn = anyOn || q.on;
        evalsA += dust_volume_texels(q, levelA, lwA, lhA);
        evalsB += dust_volume_texels(q, levelB, lwB, lhB);
    }
    if (!anyOn) { DUST_PROF_END(vplan) return; }
    level = anyFull ? levelB : levelA;
    uint evals = anyFull ? evalsB : evalsA;
    // the evaluation cap: a coarser level first, then fewer pieces (along seg_t first); both rare
    while (!(evals <= DUST_VOLUME_MAX_EVALS || level + 1u >= levels)) {
        level++;
        DUST_VOLUME_EVALS_AT(evals, level)
    }
    for (;;) {
        if (evals <= DUST_VOLUME_MAX_EVALS) break;
        if (nB > 1u) { nB = (nB + 1u) / 2u; cvarB = DUST_VOLUME_PIECE_VAR_CAPPED; }
        else if (nA > 1u) { nA = (nA + 1u) / 2u; cvarA = DUST_VOLUME_PIECE_VAR_CAPPED; }
        else break;
        DUST_VOLUME_EVALS_AT(evals, level)
    }
    uint n = nA * nB;
    // the in-frame gate (dust::volume_scatter): past DUST_FRAME_HARD_FACTOR budgets this frame,
    // at least dust_gate_level (the evaluations only shrink)
    uint before = atomicAdd(pc.pyramid.v[DUST_PYR_VOLUME_EVALS], evals);
    if (dust_frame_gates_on() && before > DUST_FRAME_HARD_FACTOR * DUST_VOLUME_FRAME_BUDGET) level = max(level, dust_gate_level(levels));
    uint off = pc.pyramid.v[DUST_PYR_TABLE + 3u * level];
    uint lw = pc.pyramid.v[DUST_PYR_TABLE + 3u * level + 1u];
    uint lh = pc.pyramid.v[DUST_PYR_TABLE + 3u * level + 2u];
    uint depthBits = floatBitsToUint(max(sqrt(dot(eye - p.mean, eye - p.mean)) / DUST_AU_M, 0.0));
    // the support's cut: the flux over the kept mass (dust::volume_scatter)
    float renorm = dust_has_support(dust_phot_support)
        ? 1.0 / max(dust_support_mass(p, dust_phot_support), DUST_SUPPORT_MASS_MIN) : 1.0;
    DUST_PROF_END(vplan)
    DUST_PROF_BEGIN(vdraw)
    // pass B: each piece rebuilt and drawn, its row segments one at a time (dust_volume_segment)
    float peak = 0.0;
    bool any_ = false;
    float sz = float(1u << level);
    for (uint k = 0u; k < n; ++k) {
        DustVolumePiece q;
        DUST_VOLUME_BUILD(q, k)
        ivec4 r = dust_volume_range(q, level, lw, lh);
        if (r.y < r.x) continue;
        uint pieceSeed = dust_hash32(seed ^ k);
        for (int ty = r.z; ty <= r.w; ++ty) {
            for (int xa = r.x; xa <= r.y; xa += int(DUST_SEGMENT_TEXELS)) {
                int xb = min(xa + int(DUST_SEGMENT_TEXELS) - 1, r.y);
                DUST_PROF_COUNT(volume_texels, xb - xa + 1)
                any_ = dust_volume_segment(pc.pyramid, pc.invUnit, q, renorm, p.mean, eye, r0, r1, r3, W, H, sz,
                                           color, depthBits, off, lw, pieceSeed, ty, xa, xb, peak) || any_;
            }
        }
    }
    if (peak > 0.0) atomicMax(pc.pyramid.v[DUST_PYR_TAU_MAX], floatBitsToUint(peak));
    if (any_) atomicOr(pc.pyramid.v[DUST_PYR_LEVEL_MASK], 1u << level);
    DUST_PROF_END(vdraw)
}
