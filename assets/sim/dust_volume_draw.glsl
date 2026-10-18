// @assets/sim/dust_volume_draw.glsl
//
// The volumetric packet draw of dust_draw_volume.comp (a lane per record of the chunk's volume
// queue, or of its volume spill: dust_split.comp never draws a volumetric packet itself), on the
// packet and seed the splitter emitted, so the pyramid is the same word for word whichever queue
// held the record. The shader declares the push-constant block `pc`
// (dust::DustSplatPushConstants) before including this, and sets the per-packet globals it reads
// (dust_phot_support / _sun / _rh / _albedo / _lut, dust_pyr_split_word). Include after
// dust_scatter.glsl.

#include "dust_frame_gates.glsl"

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
                           hasB, W, H, r0, r1, r3, tv, eye, fPx, levels, true);
// the piece without its own level (pass B, its class packed by pass A: step D3)
#define DUST_VOLUME_BUILD_FAST(q_, k_) \
    q_ = dust_volume_piece(k_, p.flux, p.rho, exposure, axMean, axS, segA, nA, cvarA, segB, nB, cvarB, \
                           hasB, W, H, r0, r1, r3, tv, eye, fPx, levels, false);
// a piece's class, orders and level packed in 16 bits (class | (kw - 1) << 2 | (kb - 1) << 4 |
// (line axis - 1) << 6 | level << 8; kw the order its level asks for, kb its order bound), two
// pieces a word (step F10)
uint dust_volume_class_code(DustVolumePiece q, uvec2 ck, uint kb, uint lev) {
    uint ax = (ck.x == DUST_VOLUME_CLASS_LINE && q.line != 0u) ? q.line - 1u : 0u;
    return ck.x | ((ck.y - 1u) << 2u) | ((kb - 1u) << 4u) | (ax << 6u) | (lev << 8u);
}
void dust_pack_code(inout uint p0, inout uint p1, inout uint p2, inout uint p3, inout uint p4, inout uint p5,
                    inout uint p6, inout uint p7, uint k, uint code) {
    uint sh = (k & 1u) * 16u;
    uint w = k >> 1u;
    if (w == 0u) p0 |= code << sh;
    else if (w == 1u) p1 |= code << sh;
    else if (w == 2u) p2 |= code << sh;
    else if (w == 3u) p3 |= code << sh;
    else if (w == 4u) p4 |= code << sh;
    else if (w == 5u) p5 |= code << sh;
    else if (w == 6u) p6 |= code << sh;
    else p7 |= code << sh;
}
uint dust_unpack_code(uint p0, uint p1, uint p2, uint p3, uint p4, uint p5, uint p6, uint p7, uint k) {
    uint w = k >> 1u;
    uint v = w == 0u ? p0 : (w == 1u ? p1 : (w == 2u ? p2 : (w == 3u ? p3 : (w == 4u ? p4 : (w == 5u ? p5 : (w == 6u ? p6 : p7))))));
    return (v >> ((k & 1u) * 16u)) & 0xFFFFu;
}
// the texels of a piece's range at a level (the table read)
uint dust_volume_texels_at(DustVolumePiece q, uint lev) {
    return dust_volume_texels(q, lev, pc.pyramid.v[DUST_PYR_TABLE + 3u * lev + 1u], pc.pyramid.v[DUST_PYR_TABLE + 3u * lev + 2u]);
}
// a piece's level (dust::volume_piece_level, steps F8 / F10): its own lp plus the bias (the cap's
// uniform extra included), at most the last level
uint dust_volume_piece_level(DustVolumePiece q, uint bias, uint levels) {
    return min(q.lp + bias, levels - 1u);
}
// the evaluations one texel of a class and order costs (dust::volume_texel_cost)
uint dust_volume_texel_cost(uint c, uint k) {
    return c == DUST_VOLUME_CLASS_SUBRAY ? k * k : (c == DUST_VOLUME_CLASS_LINE ? k : 1u);
}
// the largest sub-ray / line order (<= 4, at least 1) whose evaluations at `lev` stay within the
// piece's centre-class work one level above its own, cost(k) t(lev) <= t(lp + 1)
// (dust::volume_order_bound)
uint dust_volume_order_bound(DustVolumePiece q, uint lev, uint c, uint levels) {
    if (c != DUST_VOLUME_CLASS_SUBRAY && c != DUST_VOLUME_CLASS_LINE) return 1u;
    uint t1 = dust_volume_texels_at(q, min(q.lp + 1u, levels - 1u));
    uint t = dust_volume_texels_at(q, lev);
    uint k = DUST_VOLUME_SUBRAY_ORDER_MAX;
    while (k > 1u && dust_volume_texel_cost(c, k) * t > t1) k--;
    // the one stated exception (dust::volume_order_bound): a full sub-ray piece at its first coarse
    // level (c = 2) keeps k = 2 when its 2 x 2 sub-rays stay within its own-level centre work,
    // 4 t(lev) <= t(lp) (the eye-inside gate case fell to one ray per texel across its own core)
    if (k < 2u && c == DUST_VOLUME_CLASS_SUBRAY && q.full && lev == q.lp + 2u
        && 4u * t <= dust_volume_texels_at(q, q.lp)) k = 2u;
    return k;
}
// a piece's class and order under the packet's sub-ray limit K and its bound kb: a non-full sub-ray
// piece whose order exceeds K is drawn small (dust::volume_piece_class_k)
uvec2 dust_volume_class_k(DustVolumePiece q, uvec2 ck, uint K, uint kb) {
    if (ck.x == DUST_VOLUME_CLASS_SUBRAY && !q.full && ck.y > K) return uvec2(DUST_VOLUME_CLASS_SMALL, 1u);
    if (ck.x != DUST_VOLUME_CLASS_SUBRAY && ck.x != DUST_VOLUME_CLASS_LINE) return uvec2(ck.x, 1u);
    return uvec2(ck.x, min(min(max(min(ck.y, K), 1u), kb), DUST_VOLUME_SUBRAY_ORDER_MAX));
}
// every piece at its own level with bias_ (dust::volume_evals_pp / volume_subray_limit): the texels
// (out), the evaluations at the sub-ray limits 2, 3, 4 (e2_, e3_, e4_; at limit 1 the texels), the
// pieces' codes (pk0..pk7), whether any is on screen and whether all are at the last level
#define DUST_VOLUME_EVALS_PP(out, e2_, e3_, e4_, anyOn_, allTop_, bias_) { \
    out = 0u; e2_ = 0u; e3_ = 0u; e4_ = 0u; \
    pk0 = 0u; pk1 = 0u; pk2 = 0u; pk3 = 0u; pk4 = 0u; pk5 = 0u; pk6 = 0u; pk7 = 0u; \
    anyOn_ = false; allTop_ = true; \
    for (uint k_ = 0u; k_ < nA * nB; ++k_) { \
        DustVolumePiece q_; \
        DUST_VOLUME_BUILD(q_, k_) \
        uint lev_ = dust_volume_piece_level(q_, (bias_), levels); \
        uint t_ = dust_volume_texels_at(q_, lev_); \
        uvec2 ck_ = dust_volume_class(q_, lev_, levels); \
        uint kb_ = dust_volume_order_bound(q_, lev_, ck_.x, levels); \
        anyOn_ = anyOn_ || q_.on; \
        allTop_ = allTop_ && lev_ + 1u >= levels; \
        out += t_; \
        uvec2 c2_ = dust_volume_class_k(q_, ck_, 2u, kb_), c3_ = dust_volume_class_k(q_, ck_, 3u, kb_), \
              c4_ = dust_volume_class_k(q_, ck_, 4u, kb_); \
        e2_ += t_ * dust_volume_texel_cost(c2_.x, c2_.y); \
        e3_ += t_ * dust_volume_texel_cost(c3_.x, c3_.y); \
        e4_ += t_ * dust_volume_texel_cost(c4_.x, c4_.y); \
        dust_pack_code(pk0, pk1, pk2, pk3, pk4, pk5, pk6, pk7, k_, dust_volume_class_code(q_, ck_, kb_, lev_)); \
    } \
}

// the row segments of a texel range r_ (DUST_SEGMENT_TEXELS columns each, from its first column),
// CALL_ drawing one (it sees ty, xa, xb) and returning whether it wrote
#define DUST_VOLUME_ROWS(r_, CALL_) \
    for (int ty = (r_).z; ty <= (r_).w; ++ty) { \
        for (int xa = (r_).x; xa <= (r_).y; xa += int(DUST_SEGMENT_TEXELS)) { \
            int xb = min(xa + int(DUST_SEGMENT_TEXELS) - 1, (r_).y); \
            DUST_PROF_COUNT(volume_texels, xb - xa + 1) \
            any_ = CALL_ || any_; \
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
    float fPx = pl.fPx;
    DustVolumeBound bound = pl.bound;
    // pass A (steps F8 / F10): every piece at its own level with the frame's bias — the texels, the
    // evaluations at the sub-ray limits, the pieces' classes, orders and levels packed
    uint pk0, pk1, pk2, pk3, pk4, pk5, pk6, pk7;
    uint evals, e2, e3, e4;
    bool anyOn, allTop;
    DUST_PROF_BEGIN(vclass)
    DUST_VOLUME_EVALS_PP(evals, e2, e3, e4, anyOn, allTop, bias)
    DUST_PROF_END(vclass)
    if (!anyOn) { DUST_PROF_END(vplan) return; }
    // the evaluation cap on the texels: every piece one level coarser (dust::volume_plan_pieces,
    // the uniform extra), then fewer pieces (along seg_t first); both rare
    uint extra = 0u;
    while (!(evals <= DUST_VOLUME_MAX_EVALS || allTop)) {
        extra++;
        DUST_VOLUME_EVALS_PP(evals, e2, e3, e4, anyOn, allTop, bias + extra)
    }
    for (;;) {
        if (evals <= DUST_VOLUME_MAX_EVALS) break;
        if (nB > 1u) { nB = (nB + 1u) / 2u; cvarB = DUST_VOLUME_PIECE_VAR_CAPPED; }
        else if (nA > 1u) { nA = (nA + 1u) / 2u; cvarA = DUST_VOLUME_PIECE_VAR_CAPPED; }
        else break;
        DUST_VOLUME_EVALS_PP(evals, e2, e3, e4, anyOn, allTop, bias + extra)
    }
    uint n = nA * nB;
    // the packet's sub-ray limit K: the largest of 4, 3, 2, 1 whose evaluations fit the cap
    // (dust::volume_subray_limit); the evaluations counted are those at K
    uint K = 4u;
    if (e4 > DUST_VOLUME_MAX_EVALS) K = 3u;
    if (K == 3u && e3 > DUST_VOLUME_MAX_EVALS) K = 2u;
    if (K == 2u && e2 > DUST_VOLUME_MAX_EVALS) K = 1u;
    evals = K == 4u ? e4 : (K == 3u ? e3 : (K == 2u ? e2 : evals));
    // the in-frame gate (dust::volume_scatter): past DUST_FRAME_HARD_FACTOR budgets this frame,
    // every piece at least at dust_gate_level (the evaluations only shrink)
    uint before = atomicAdd(pc.pyramid.v[DUST_PYR_VOLUME_EVALS], evals);
    bool gated = dust_frame_gates_on() && before > DUST_FRAME_HARD_FACTOR * DUST_VOLUME_FRAME_BUDGET;
    uint gate = dust_gate_level(levels);
    uint depthBits = floatBitsToUint(max(sqrt(dot(eye - p.mean, eye - p.mean)) / DUST_AU_M, 0.0));
    // the support's cut: the flux over the kept mass (dust::volume_scatter)
    float renorm = dust_has_support(dust_phot_support)
        ? 1.0 / max(dust_support_mass(p, dust_phot_support), DUST_SUPPORT_MASS_MIN) : 1.0;
    DUST_PROF_END(vplan)
    DUST_PROF_BEGIN(vdraw)
    // pass B: each piece rebuilt and drawn at its own level, its row segments one at a time; the
    // class and level from pass A's codes (the piece rebuilt without its own level), or, past the
    // gate, computed again at the gated level
    float peak = 0.0;
    uint maskBits = 0u;
    for (uint k = 0u; k < n; ++k) {
        DustVolumePiece q;
        uint code = dust_unpack_code(pk0, pk1, pk2, pk3, pk4, pk5, pk6, pk7, k);
        uint lev = code >> 8u;
        uvec2 ck;
        uint kb;
        if (gated && lev < gate) {
            // past the gate: the piece at the gate level, its class and bound there
            DUST_VOLUME_BUILD(q, k)
            lev = gate;
            ck = dust_volume_class(q, lev, levels);
            kb = dust_volume_order_bound(q, lev, ck.x, levels);
        } else {
            DUST_VOLUME_BUILD_FAST(q, k)
            ck = uvec2(code & 3u, ((code >> 2u) & 3u) + 1u);
            kb = ((code >> 4u) & 3u) + 1u;
            if (ck.x == DUST_VOLUME_CLASS_LINE) q.line = ((code >> 6u) & 1u) + 1u;
        }
        uint off = pc.pyramid.v[DUST_PYR_TABLE + 3u * lev];
        uint lw = pc.pyramid.v[DUST_PYR_TABLE + 3u * lev + 1u];
        uint lh = pc.pyramid.v[DUST_PYR_TABLE + 3u * lev + 2u];
        float sz = float(1u << lev);
        ivec4 r = dust_volume_range(q, lev, lw, lh);
        if (r.y < r.x || r.w < r.z) continue;
        // the class and order under the limit K and the piece's bound; one loop per class
        uvec2 cks = dust_volume_class_k(q, ck, K, kb);
        uint cls = cks.x;
        uint kSub = max(cks.y, 1u);
        uint pieceSeed = dust_hash32(seed ^ k);
        bool any_ = false;
#ifdef DUST_K1C_OFF
        DUST_PROF_BEGIN(vcentre)
        DUST_VOLUME_ROWS(r, dust_volume_segment(pc.pyramid, pc.invUnit, q, bound, renorm, p.mean, eye, r0, r1, r3, W, H,
                                                sz, color, depthBits, off, lw, pieceSeed, ty, xa, xb, peak))
        DUST_PROF_END(vcentre)
#else
        if (cls == DUST_VOLUME_CLASS_CENTRE) {
            DUST_PROF_BEGIN(vcentre)
            DUST_VOLUME_ROWS(r, dust_volume_segment(pc.pyramid, pc.invUnit, q, bound, renorm, p.mean, eye, r0, r1, r3, W, H,
                                                    sz, color, depthBits, off, lw, pieceSeed, ty, xa, xb, peak))
            DUST_PROF_END(vcentre)
        } else if (cls == DUST_VOLUME_CLASS_SMALL) {
            DustSmallPoint s0, s1, s2, s3, s4;
            DUST_PROF_BEGIN(vsmallpts)
            dust_volume_small_points(q, eye, r0, r1, r3, tv, W, H, p.mean, renorm, s0, s1, s2, s3, s4);
            DUST_PROF_END(vsmallpts)
            // only the texels its points cover (dust::volume_scatter)
            ivec4 rs = dust_volume_small_range(s0, s1, s2, s3, s4, r, sz);
            if (rs.y < rs.x || rs.w < rs.z) continue;
            DUST_VOLUME_ROWS(rs, dust_volume_segment_small(pc.pyramid, pc.invUnit, W, H, sz, color, depthBits, off,
                                                           lw, pieceSeed, ty, xa, xb, s0, s1, s2, s3, s4,
                                                           peak))
        } else {
            bool lines = cls == DUST_VOLUME_CLASS_LINE;
            DUST_VOLUME_ROWS(r, dust_volume_segment_sub(pc.pyramid, pc.invUnit, q, bound, renorm, p.mean, eye, r0, r1, r3,
                                                        W, H, sz, color, depthBits, off, lw, pieceSeed, ty, xa, xb,
                                                        lines, kSub, peak))
        }
#endif
        if (any_) maskBits |= 1u << lev;
    }
    if (peak > 0.0) atomicMax(pc.pyramid.v[DUST_PYR_TAU_MAX], floatBitsToUint(peak));
    if (maskBits != 0u) atomicOr(pc.pyramid.v[DUST_PYR_LEVEL_MASK], maskBits);
    DUST_PROF_END(vdraw)
}
