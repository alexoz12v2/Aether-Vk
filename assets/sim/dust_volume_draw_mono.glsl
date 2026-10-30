// @assets/sim/dust_volume_draw_mono.glsl
//
// the monolithic volumetric draw, kept for the mono draw family (AETHERVK_DUST_DRAW_STAGED=0,
// the GTX 1070 default): dust_volume_draw.glsl as it was before the staged consumer, adapted only
// in the edge record layout (dust_small_edge_put_mono below).
//
// The volumetric packet draw of dust_draw_volume.comp (a lane per record of the chunk's volume
// queue, or of its volume spill: dust_split.comp never draws a volumetric packet itself), on the
// packet and seed the splitter emitted, so the pyramid is the same word for word whichever queue
// held the record. The shader declares the push-constant block `pc`
// (dust::DustSplatPushConstants) before including this, and sets the per-packet globals it reads
// (dust_phot_support / _sun / _rh / _albedo / _lut, dust_pyr_split_word). Include after
// dust_scatter.glsl.

#include "dust_frame_gates.glsl"

// (the mono draw family) the edge records' old layout: structure-of-arrays at the
// capacity's stride (word w of record i at the region's records + w * cap + i), the edge pass's
// indirect x in workgroups of 64 pieces (dust_vqueue_append). The shared dust_scatter.glsl's
// dust_small_edge_* now use the staged family's array-of-structures layout, so the mono consumer
// and the mono edge pass carry the old functions here, renamed `_mono`, bodies unchanged.
void dust_small_edge_put_mono(DustUintBuffer pyr, bool odd, DustSmallEdge ed, ivec4 r, uint lev, vec3 color,
                         uint depthBits, uint seed) {
    uint ebase = dust_vedge_base(pyr, odd);
    uint i = dust_vqueue_append(pyr, ebase);
    if (i == 0xFFFFFFFFu) return;
    uint cap = dust_vqueue_capacity(pyr, ebase);
    uint a = ebase + DUST_VQUEUE_HEADER_WORDS + i;
    pyr.v[a + 0u * cap] = floatBitsToUint(ed.mean.x);
    pyr.v[a + 1u * cap] = floatBitsToUint(ed.mean.y);
    pyr.v[a + 2u * cap] = floatBitsToUint(ed.mean.z);
    pyr.v[a + 3u * cap] = floatBitsToUint(ed.g.x);
    pyr.v[a + 4u * cap] = floatBitsToUint(ed.g.y);
    pyr.v[a + 5u * cap] = floatBitsToUint(ed.g.z);
    pyr.v[a + 6u * cap] = floatBitsToUint(ed.x1);
    pyr.v[a + 7u * cap] = floatBitsToUint(ed.x2);
    pyr.v[a + 8u * cap] = floatBitsToUint(ed.y1);
    pyr.v[a + 9u * cap] = floatBitsToUint(ed.y2);
    pyr.v[a + 10u * cap] = floatBitsToUint(ed.w1);
    pyr.v[a + 11u * cap] = floatBitsToUint(ed.w2);
    pyr.v[a + 12u * cap] = floatBitsToUint(ed.massW);
    pyr.v[a + 13u * cap] = floatBitsToUint(ed.wr0);
    pyr.v[a + 14u * cap] = floatBitsToUint(ed.wr1);
    pyr.v[a + 15u * cap] = floatBitsToUint(ed.wr2);
    pyr.v[a + 16u * cap] = uint(r.x) | (uint(r.y) << 16u);
    pyr.v[a + 17u * cap] = uint(r.z) | (uint(r.w) << 16u);
    pyr.v[a + 18u * cap] = lev;
    pyr.v[a + 19u * cap] = floatBitsToUint(color.x);
    pyr.v[a + 20u * cap] = floatBitsToUint(color.y);
    pyr.v[a + 21u * cap] = floatBitsToUint(color.z);
    pyr.v[a + 22u * cap] = depthBits;
    pyr.v[a + 23u * cap] = seed;
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
                           hasB, W, H, r0, r1, r3, tv, eye, fPx, levels, true);

// Pass B's state out of registers. The consumer is latency-bound at the
// 128-register cap: everything live across pass B's texel loops counts against the warps an SM
// holds, and most of it is only read when a piece is built. So each lane keeps the state its
// piece loop reads once a piece in its own slots of workgroup memory (word c of lane l at
// c * gl_WorkGroupSize.x + l: the lanes of a warp on consecutive banks): pass A's piece codes
// (DUST_VS_PK, written there by pass A, read by a k-dependent index), the pieces' covariance
// (dust_volume_piece_cov's Cholesky factor and diagonal: the same for every piece, built once
// after the cap loops), the packet's 3-sigma bound, fPx, and the loop's constants packed in one
// word (DUST_VS_CTRL). The loop-invariant words are read back by an atomic OR of 0: a plain load
// of a word the loop never stores could be hoisted out of the piece loop or forwarded from its
// store, which would put the value back in a register; an atomic is neither. The packet's own
// words (mean, segments, flux, rho, seed, exposure, colour, support, Sun, r_h) are re-read from
// its queue record (dust_vrec; the deposits' atomics on the same buffer keep those loads inside
// the loop). Bits and arithmetic unchanged: a word is stored and read back as its bits, and the
// pieces are built by the same operations (dust_volume_piece = _cov + _at).
// 31 words a lane: 7 936 B a workgroup of 64, 12 workgroups (24 warps) in an SM's 96 KB.
const uint DUST_VS_PK = 0u;      // 8 words: pass A's codes, two pieces a word (dust_volume_class_code)
const uint DUST_VS_L = 8u;       // 6: the pieces' Cholesky factor
const uint DUST_VS_SKD = 14u;    // 3: the pieces' covariance diagonal
const uint DUST_VS_BOUND = 17u;  // 12: the 3-sigma bound (l[6], qe, g)
const uint DUST_VS_FPX = 29u;    // 1: fPx (the gated path's level)
const uint DUST_VS_CTRL = 30u;   // 1: nA | nB << 5 | K << 10 | axis mode << 13 | hasB << 15 | gated << 16 | levels << 17 (8 bits) | top << 25 (26-31 clear)
const uint DUST_VS_WORDS = 31u;
uint dust_pk[8];  // XID31-V2: the codes in private memory
shared uint dust_vs[DUST_VS_WORDS * gl_WorkGroupSize.x];
uint dust_vs_at(uint c) { return c * gl_WorkGroupSize.x + gl_LocalInvocationIndex; }
void dust_vs_put(uint c, float v) { dust_vs[dust_vs_at(c)] = floatBitsToUint(v); }
uint dust_vs_word(uint c) { return atomicOr(dust_vs[dust_vs_at(c)], 0u); }
float dust_vs_get(uint c) { return uintBitsToFloat(dust_vs_word(c)); }
void dust_vs_put_bound(DustVolumeBound b) {
    dust_vs_put(DUST_VS_BOUND + 0u, b.l[0]); dust_vs_put(DUST_VS_BOUND + 1u, b.l[1]); dust_vs_put(DUST_VS_BOUND + 2u, b.l[2]);
    dust_vs_put(DUST_VS_BOUND + 3u, b.l[3]); dust_vs_put(DUST_VS_BOUND + 4u, b.l[4]); dust_vs_put(DUST_VS_BOUND + 5u, b.l[5]);
    dust_vs_put(DUST_VS_BOUND + 6u, b.qe.x); dust_vs_put(DUST_VS_BOUND + 7u, b.qe.y); dust_vs_put(DUST_VS_BOUND + 8u, b.qe.z);
    dust_vs_put(DUST_VS_BOUND + 9u, b.g.x); dust_vs_put(DUST_VS_BOUND + 10u, b.g.y); dust_vs_put(DUST_VS_BOUND + 11u, b.g.z);
}
// the bound for a piece whose rays the capsule clips (q.cap); zeros otherwise, never read
// (dust_volume_ray_tau reads the bound only for a cap piece)
DustVolumeBound dust_vs_bound(bool cap) {
    DustVolumeBound b;
    b.l[0] = 0.0; b.l[1] = 0.0; b.l[2] = 0.0; b.l[3] = 0.0; b.l[4] = 0.0; b.l[5] = 0.0;
    b.qe = vec3(0.0);
    b.g = vec3(0.0);
    if (cap) {
        b.l[0] = dust_vs_get(DUST_VS_BOUND + 0u); b.l[1] = dust_vs_get(DUST_VS_BOUND + 1u);
        b.l[2] = dust_vs_get(DUST_VS_BOUND + 2u); b.l[3] = dust_vs_get(DUST_VS_BOUND + 3u);
        b.l[4] = dust_vs_get(DUST_VS_BOUND + 4u); b.l[5] = dust_vs_get(DUST_VS_BOUND + 5u);
        b.qe = vec3(dust_vs_get(DUST_VS_BOUND + 6u), dust_vs_get(DUST_VS_BOUND + 7u), dust_vs_get(DUST_VS_BOUND + 8u));
        b.g = vec3(dust_vs_get(DUST_VS_BOUND + 9u), dust_vs_get(DUST_VS_BOUND + 10u), dust_vs_get(DUST_VS_BOUND + 11u));
    }
    return b;
}
// word w of the lane's queue record (ra = its first word, rc = the queue's capacity: the column
// stride; the word order of DUST_VQUEUE_PUT / DUST_VQUEUE_GET: 0-2 mean, 9-11 seg, 12 flux,
// 13-14 rho, 15-17 segT, 18 seed, 19 exposure, 20-22 colour, 23-26 support, 27-29 Sun, 30 r_h)
float dust_vrec(uint ra, uint rc, uint w) { return uintBitsToFloat(pc.pyramid.v[ra + w * rc]); }
vec3 dust_vrec3(uint ra, uint rc, uint w) { return vec3(dust_vrec(ra, rc, w), dust_vrec(ra, rc, w + 1u), dust_vrec(ra, rc, w + 2u)); }
// a piece's class, orders and level packed in 16 bits (class | (kw - 1) << 2 | (kb - 1) << 4 |
// (line axis - 1) << 6 | level << 8; kw the order its level asks for, kb its order bound), two
// pieces a word (step F10)
uint dust_volume_class_code(DustVolumePiece q, uvec2 ck, uint kb, uint lev) {
    uint ax = (ck.x == DUST_VOLUME_CLASS_LINE && q.line != 0u) ? q.line - 1u : 0u;
    return ck.x | ((ck.y - 1u) << 2u) | ((kb - 1u) << 4u) | (ax << 6u) | (lev << 8u);
}
// (the eight code words in private memory `dust_pk`: kept in the shared slots DUST_VS_PK.. they were overwritten by an unfound writer
//  on some frames (garbage levels, Xid 31); k < 16)
void dust_pack_code(uint k, uint code) {
    dust_pk[k >> 1u] |= code << ((k & 1u) * 16u);
}
uint dust_unpack_code(uint k) {
    return (dust_pk[k >> 1u] >> ((k & 1u) * 16u)) & 0xFFFFu;
}
// the texels of a piece's range at a level (the table read)
uint dust_volume_texels_at(DustVolumePiece q, uint lev) {
    return dust_volume_texels(q, lev, pc.pyramid.v[DUST_PYR_TABLE + 3u * lev + 1u], pc.pyramid.v[DUST_PYR_TABLE + 3u * lev + 2u]);
}
// a piece's level (dust::volume_piece_level): its own lp plus the bias (the cap's
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
// pieces' codes (the lane's slots DUST_VS_PK..), whether any is on screen and whether all are at
// the last level
#define DUST_VOLUME_EVALS_PP(out, e2_, e3_, e4_, anyOn_, allTop_, bias_) { \
    out = 0u; e2_ = 0u; e3_ = 0u; e4_ = 0u; \
    for (uint w_ = 0u; w_ < 8u; ++w_) dust_pk[w_] = 0u; \
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
        dust_pack_code(k_, dust_volume_class_code(q_, ck_, kb_, lev_)); \
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

// the frame's top-level regime (frame-uniform, from the pyramid header): the bias puts every piece
// on the last level and that level is 1 x 1 (topOnly); known to every path unless the reference
// mode DUST_FRAME_TOP_PASS_A asks for pass A in full (topKnown; dust_draw_volume.comp main sets the
// accumulate mode from it)
bool dust_volume_top_only(uint levels, uint bias) {
    uint topW = pc.pyramid.v[DUST_PYR_TABLE + 3u * (levels - 1u) + 1u];
    uint topH = pc.pyramid.v[DUST_PYR_TABLE + 3u * (levels - 1u) + 2u];
    return bias + 1u >= levels && topW == 1u && topH == 1u;
}
bool dust_volume_top_known(uint levels, uint bias) {
    return dust_volume_top_only(levels, bias) && (pc.pyramid.v[DUST_PYR_FLAGS] & DUST_FRAME_TOP_PASS_A) == 0u;
}
// the lane's end words under the accumulate mode (dust_draw_volume.comp main reduces them over the
// subgroup: DUST_PYR_TAU_MAX max, DUST_PYR_LEVEL_MASK or, DUST_PYR_VOLUME_EVALS add)
float dust_vacc_peak = 0.0;
uint  dust_vacc_mask = 0u, dust_vacc_evals = 0u;

void dust_volume_draw(DustPacket p, uint ra, uint rc, float exposure, uint levels, float W, float H, uint bias,
                      uint axis) {
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
    // (the bound into the lane's slots at once, the axis mode for pass B's rebuild: as
    // dust_volume_plan_axes chose it)
    dust_vs_put_bound(pl.bound);
    uint mode = (axis != DUST_VOLUME_AXIS_CHORD && dot(p.segT, p.segT) > 0.0) ? axis : DUST_VOLUME_AXIS_CHORD;
    // pass A: every piece at its own level with the frame's bias — the texels, the
    // evaluations at the sub-ray limits, the pieces' classes, orders and levels packed
    uint evals, e2, e3, e4;
    bool anyOn, allTop;
    // the top-level path: the bias puts every piece on the last level
    // (min(lp + bias, levels - 1), lp >= 0) and that level is 1 x 1, so a piece covers at most one
    // texel: the evaluations are at most 16 pieces x 16 (DUST_VOLUME_MAX_EVALS), the cap loops
    // never act and K is 4. Pass A then only asks whether any piece is on screen (the first one
    // found ends it); pass B classifies each piece at the top level (the gated path's build) and
    // sums the evaluations into the lane's code slots, which this path does not use. The same
    // decisions on the same integers: the pyramid is the same word for word (reference:
    // DUST_FRAME_TOP_PASS_A, dust::DUST_FRAME_TOP_PASS_A)
    bool topOnly = dust_volume_top_only(levels, bias);
    bool topKnown = dust_volume_top_known(levels, bias);
    // the volume detail (the draw's flags bits 8-10, dust::SplitView::volume_detail; 16 by
    // default, which no plan exceeds): a top-level packet cut into at most that many pieces,
    // fewer along seg_t first, as the evaluation cap (dust::volume_plan_pieces)
    if (topOnly) {
        uint detail = max(16u >> ((pc.flags >> DUST_FLAGS_VOLUME_DETAIL_SHIFT) & 7u), 1u);
        for (;;) {
            if (nA * nB <= detail) break;
            if (nB > 1u) { nB = (nB + 1u) / 2u; cvarB = DUST_VOLUME_PIECE_VAR_CAPPED; }
            else if (nA > 1u) { nA = (nA + 1u) / 2u; cvarA = DUST_VOLUME_PIECE_VAR_CAPPED; }
            else break;
        }
    }
    DUST_PROF_BEGIN(vclass)
    if (topKnown) {
        evals = 0u; e2 = 0u; e3 = 0u; e4 = 0u;
        anyOn = false; allTop = true;
        for (uint k_ = 0u; k_ < nA * nB && !anyOn; ++k_) {
            DustVolumePiece q_;
            DUST_VOLUME_BUILD(q_, k_)
            anyOn = q_.on;
        }
        dust_pk[0] = 0u;
    } else {
        DUST_VOLUME_EVALS_PP(evals, e2, e3, e4, anyOn, allTop, bias)
    }
    DUST_PROF_END(vclass)
#ifdef DUST_PROFILE
    // (PARTICLES_PERF: the profile pair, words DUST_PYR_PROF_PAIR / + 1, by the frame's mode;
    // the host logs them. 0: the packets found off screen here and their cycles / 256 so far;
    // 1 / 3: the allTop packets / those whose pieces are all SMALL too, their cycles / 256 up to
    // the end; 2: the allTop packets' pieces and the SMALL ones among them)
    uint profMode = (pc.pyramid.v[DUST_PYR_FLAGS] >> DUST_FRAME_PROF_PAIR_SHIFT) & 3u;
    if (profMode == 0u && !anyOn && !dust_counting) {
        atomicAdd(pc.pyramid.v[DUST_PYR_PROF_PAIR], 1u);
        atomicAdd(pc.pyramid.v[DUST_PYR_PROF_PAIR + 1u], (clock2x32ARB().x - dust_prof_t0_vplan) >> 8u);
    }
#endif
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
#ifdef DUST_PROFILE
    bool profHit = false;
    if (profMode >= 1u && allTop && !dust_counting && (profMode == 1u || !topKnown)) {
        uint nSmall = 0u;
        for (uint k = 0u; k < n; ++k) nSmall += (dust_unpack_code(k) & 3u) == DUST_VOLUME_CLASS_SMALL ? 1u : 0u;
        if (profMode == 2u) {
            atomicAdd(pc.pyramid.v[DUST_PYR_PROF_PAIR], n);
            atomicAdd(pc.pyramid.v[DUST_PYR_PROF_PAIR + 1u], nSmall);
        }
        profHit = profMode == 1u || (profMode == 3u && nSmall == n);
    }
#endif
    // the packet's sub-ray limit K: the largest of 4, 3, 2, 1 whose evaluations fit the cap
    // (dust::volume_subray_limit); the evaluations counted are those at K
    uint K = 4u;
    if (e4 > DUST_VOLUME_MAX_EVALS) K = 3u;
    if (K == 3u && e3 > DUST_VOLUME_MAX_EVALS) K = 2u;
    if (K == 2u && e2 > DUST_VOLUME_MAX_EVALS) K = 1u;
    evals = K == 4u ? e4 : (K == 3u ? e3 : (K == 2u ? e2 : evals));
    // the in-frame gate (dust::volume_scatter): past DUST_FRAME_HARD_FACTOR budgets this frame,
    // every piece at least at dust_gate_level (the evaluations only shrink)
    // (the top-level path: no piece is below the gate level, and every volumetric packet of the
    // frame takes this path, so none reads the running sum; pass B adds the evaluations)
    bool gated = false;
    if (!topKnown) {
        uint before = atomicAdd(pc.pyramid.v[DUST_PYR_VOLUME_EVALS], evals);
        gated = dust_frame_gates_on() && before > DUST_FRAME_HARD_FACTOR * DUST_VOLUME_FRAME_BUDGET;
    }
    uint depthBits = floatBitsToUint(max(sqrt(dot(eye - p.mean, eye - p.mean)) / DUST_AU_M, 0.0));
    // the support's cut: the flux over the kept mass (dust::volume_scatter; the packet's
    // covariance re-read from its record: not kept live across pass A)
    p.S[0] = dust_vrec(ra, rc, 3u); p.S[1] = dust_vrec(ra, rc, 4u); p.S[2] = dust_vrec(ra, rc, 5u);
    p.S[3] = dust_vrec(ra, rc, 6u); p.S[4] = dust_vrec(ra, rc, 7u); p.S[5] = dust_vrec(ra, rc, 8u);
    float renorm = dust_has_support(dust_phot_support)
        ? 1.0 / max(dust_support_mass(p, dust_phot_support), DUST_SUPPORT_MASS_MIN) : 1.0;
    // pass B's state into the lane's slots: the pieces' covariance at the final nA, nB
    // (after the cap loops), fPx, the loop's constants
    {
        float Lk[6];
        vec3 skd;
        dust_volume_piece_cov(axS, segA, nA, cvarA, segB, nB, cvarB, hasB, Lk, skd);
        dust_vs_put(DUST_VS_L + 0u, Lk[0]); dust_vs_put(DUST_VS_L + 1u, Lk[1]); dust_vs_put(DUST_VS_L + 2u, Lk[2]);
        dust_vs_put(DUST_VS_L + 3u, Lk[3]); dust_vs_put(DUST_VS_L + 4u, Lk[4]); dust_vs_put(DUST_VS_L + 5u, Lk[5]);
        dust_vs_put(DUST_VS_SKD + 0u, skd.x); dust_vs_put(DUST_VS_SKD + 1u, skd.y); dust_vs_put(DUST_VS_SKD + 2u, skd.z);
        dust_vs_put(DUST_VS_FPX, fPx);
        dust_vs[dust_vs_at(DUST_VS_CTRL)] = nA | (nB << 5u) | (K << 10u) | (mode << 13u) | ((hasB ? 1u : 0u) << 15u)
                                          | ((gated ? 1u : 0u) << 16u) | (levels << 17u) | ((topKnown ? 1u : 0u) << 25u);
    }
    DUST_PROF_END(vplan)
    DUST_PROF_BEGIN(vdraw)
    // pass B: each piece rebuilt and drawn at its own level, its row segments one at a time; the
    // class and level from pass A's codes (the piece rebuilt without its own level), or, past the
    // gate, computed again at the gated level. Live across the pieces: k, n, the record's address,
    // depthBits, renorm, W, H and the accumulators; the rest is re-read at each piece
    float peak = 0.0;
    uint maskBits = 0u;
    for (uint k = 0u; k < n; ++k) {
        DUST_PROF_BEGIN(vhead)
        uint ctl = dust_vs_word(DUST_VS_CTRL);
        uint nAk = ctl & 31u, nBk = (ctl >> 5u) & 31u, Kk = (ctl >> 10u) & 7u, modek = (ctl >> 13u) & 3u;
        bool hasBk = ((ctl >> 15u) & 1u) != 0u, gatedk = ((ctl >> 16u) & 1u) != 0u;
        uint levelsk = (ctl >> 17u) & 0xFFu;
        // the packet's words (dust_volume_plan_axes' axes: the chord, or seg_t as axis b). z = 0
        // (ctl's top bits are never set) but unknown to the compiler: the record's word offsets are
        // computed here at each piece, not hoisted out of the loop as twenty live addresses
        uint z = ctl >> 31u;
        uint raK = ra + z, rcK = rc + z;
        vec3 meanRef = dust_vrec3(raK, rcK, 0u);
        vec3 seg = dust_vrec3(raK, rcK, 9u);
        float flux = dust_vrec(raK, rcK, 12u);
        vec2 rho = vec2(dust_vrec(raK, rcK, 13u), dust_vrec(raK, rcK, 14u));
        vec3 segT = dust_vrec3(raK, rcK, 15u);
        uint seed = pc.pyramid.v[raK + 18u * rcK];
        float expo = dust_vrec(raK, rcK, 19u);
        vec3 color = dust_vrec3(raK, rcK, 20u);
        dust_phot_support = vec4(dust_vrec(raK, rcK, 23u), dust_vrec(raK, rcK, 24u), dust_vrec(raK, rcK, 25u),
                                 dust_vrec(raK, rcK, 26u));
        dust_phot_sun = dust_vrec3(raK, rcK, 27u);
        dust_phot_rh = dust_vrec(raK, rcK, 30u);
        dust_pyr_split_word = pc.pyramid.v[DUST_PYR_SPLIT_DEPTH];
        vec3 axMeanK = meanRef, segAK = seg, segBK = vec3(0.0);
        if (modek == DUST_VOLUME_AXIS_TIME) {
            float ubar = rho.x * 0.5 + (rho.y - rho.x) / 3.0;
            axMeanK = meanRef + seg * ubar;
            segAK = vec3(0.0);
            segBK = segT;
        } else if (modek == DUST_VOLUME_AXIS_GRID) {
            segBK = segT;
        }
        float Lk[6];
        Lk[0] = dust_vs_get(DUST_VS_L + 0u); Lk[1] = dust_vs_get(DUST_VS_L + 1u); Lk[2] = dust_vs_get(DUST_VS_L + 2u);
        Lk[3] = dust_vs_get(DUST_VS_L + 3u); Lk[4] = dust_vs_get(DUST_VS_L + 4u); Lk[5] = dust_vs_get(DUST_VS_L + 5u);
        vec3 skd = vec3(dust_vs_get(DUST_VS_SKD + 0u), dust_vs_get(DUST_VS_SKD + 1u), dust_vs_get(DUST_VS_SKD + 2u));
        float fPxK = dust_vs_get(DUST_VS_FPX);
        DUST_PROF_END(vhead)
        DustVolumePiece q;
        uint code = dust_unpack_code(k);
        uint lev = code >> 8u;
        uvec2 ck;
        uint kb;
        uint gate = dust_gate_level(levelsk);
        bool topk = ((ctl >> 25u) & 1u) != 0u;
        DUST_PROF_BEGIN(vbuild)
        if (topk) {
            // the top-level path: the piece at the last level, its class and bound there (pass
            // A's build and decisions on it)
            q = dust_volume_piece_at(k, flux, rho, expo, axMeanK, Lk, skd, segAK, nAk, segBK, nBk, hasBk, W, H, r0, r1,
                                     r3, tv, eye, fPxK, levelsk, true);
            lev = levelsk - 1u;
            ck = dust_volume_class(q, lev, levelsk);
            kb = dust_volume_order_bound(q, lev, ck.x, levelsk);
        } else if (gatedk && lev < gate) {
            // past the gate: the piece at the gate level, its class and bound there
            q = dust_volume_piece_at(k, flux, rho, expo, axMeanK, Lk, skd, segAK, nAk, segBK, nBk, hasBk, W, H, r0, r1,
                                     r3, tv, eye, fPxK, levelsk, true);
            lev = gate;
            ck = dust_volume_class(q, lev, levelsk);
            kb = dust_volume_order_bound(q, lev, ck.x, levelsk);
        } else {
            // the piece without its own level (its class packed by pass A)
            q = dust_volume_piece_at(k, flux, rho, expo, axMeanK, Lk, skd, segAK, nAk, segBK, nBk, hasBk, W, H, r0, r1,
                                     r3, tv, eye, fPxK, levelsk, false);
            // (a level past the pyramid would read an empty table row and run the range loop away; the
            // codes in the lane's slots were read back out of range in the Xid 31 investigation)
            if (lev >= levelsk) {
                atomicAdd(pc.pyramid.v[DUST_PYR_BAD_CODE], 1u);
                atomicCompSwap(pc.pyramid.v[DUST_PYR_BAD_CODE_FIRST], 0u, (code & 0xFFFFu) | (k << 16u) | (n << 24u));
            }
            lev = min(lev, levelsk - 1u);
            ck = uvec2(code & 3u, ((code >> 2u) & 3u) + 1u);
            kb = ((code >> 4u) & 3u) + 1u;
            if (ck.x == DUST_VOLUME_CLASS_LINE) q.line = ((code >> 6u) & 1u) + 1u;
        }
        DUST_PROF_END(vbuild)
        DUST_PROF_BEGIN(vdecide)
        uint off = pc.pyramid.v[DUST_PYR_TABLE + 3u * lev];
        uint lw = pc.pyramid.v[DUST_PYR_TABLE + 3u * lev + 1u];
        uint lh = pc.pyramid.v[DUST_PYR_TABLE + 3u * lev + 2u];
        if (lw == 0u || lh == 0u) {
            atomicAdd(pc.pyramid.v[DUST_PYR_BAD_CODE], 0x10000u);
            continue;
        }
        float sz = float(1u << lev);
        ivec4 r = dust_volume_range(q, lev, lw, lh);
        DUST_PROF_END(vdecide)
        if (r.y < r.x || r.w < r.z) continue;
        // the class and order under the limit K and the piece's bound; one loop per class
        uvec2 cks = dust_volume_class_k(q, ck, Kk, kb);
        // (the top-level path: the piece's evaluations, pass A's t x cost at K = 4)
        if (topk) {
            uint tk = uint((r.y - r.x + 1) * (r.w - r.z + 1));
            dust_pk[0] += tk * dust_volume_texel_cost(cks.x, cks.y);
        }
        uint cls = cks.x;
        uint kSub = max(cks.y, 1u);
        uint pieceSeed = dust_hash32(seed ^ k);
        bool any_ = false;
#ifdef DUST_K1C_OFF
        DUST_PROF_BEGIN(vcentre)
        DustVolumeBound bound = dust_vs_bound(q.cap);
        DUST_VOLUME_ROWS(r, dust_volume_segment(pc.pyramid, pc.invUnit, q, bound, renorm, meanRef, eye, r0, r1, r3, W, H,
                                                sz, color, depthBits, off, lw, pieceSeed, ty, xa, xb, peak))
        DUST_PROF_END(vcentre)
#else
        if (cls == DUST_VOLUME_CLASS_CENTRE) {
            DUST_PROF_BEGIN(vcentre)
            DustVolumeBound bound = dust_vs_bound(q.cap);
            DUST_VOLUME_ROWS(r, dust_volume_segment(pc.pyramid, pc.invUnit, q, bound, renorm, meanRef, eye, r0, r1, r3, W, H,
                                                    sz, color, depthBits, off, lw, pieceSeed, ty, xa, xb, peak))
            DUST_PROF_END(vcentre)
        } else if (cls == DUST_VOLUME_CLASS_SMALL) {
            DustSmallPoint s0, s1, s2, s3, s4;
            DustSmallEdge ed;
            DUST_PROF_BEGIN(vsmallpts)
            bool edge = dust_volume_small_points(q, eye, r0, r1, r3, tv, W, H, meanRef, renorm, s0, s1, s2, s3, s4, ed);
            DUST_PROF_END(vsmallpts)
            // an edge piece is drawn by the edge pass (dust_draw_small_edge.comp)
            // from its record: its inputs, range, level, colour, depth bits and seed
            if (edge) {
                dust_small_edge_put_mono(pc.pyramid, (pc.flags & DUST_VIEW_EDGE_ODD) != 0u, ed, r, lev, color, depthBits,
                                         pieceSeed);
                continue;
            }
            // only the texels its points cover (dust::volume_scatter)
            ivec4 rs = dust_volume_small_range(s0, s1, s2, s3, s4, r, sz);
            if (rs.y < rs.x || rs.w < rs.z) continue;
            DUST_VOLUME_ROWS(rs, dust_volume_segment_small(pc.pyramid, pc.invUnit, W, H, sz, color, depthBits, off,
                                                           lw, pieceSeed, ty, xa, xb, s0, s1, s2, s3, s4,
                                                           peak))
        } else {
            bool lines = cls == DUST_VOLUME_CLASS_LINE;
            DustVolumeBound bound = dust_vs_bound(q.cap);
            DUST_VOLUME_ROWS(r, dust_volume_segment_sub(pc.pyramid, pc.invUnit, q, bound, renorm, meanRef, eye, r0, r1, r3,
                                                        W, H, sz, color, depthBits, off, lw, pieceSeed, ty, xa, xb,
                                                        lines, kSub, peak))
        }
#endif
        if (any_) maskBits |= 1u << lev;
    }
    if (dust_vacc) {
        // the end words to the subgroup flush; dust_vacc = topKnown = ctl bit 25
        dust_vacc_peak = peak;
        dust_vacc_mask = maskBits;
        dust_vacc_evals = dust_pk[0];
    } else {
        if (peak > 0.0) atomicMax(pc.pyramid.v[DUST_PYR_TAU_MAX], floatBitsToUint(peak));
        if (maskBits != 0u) atomicOr(pc.pyramid.v[DUST_PYR_LEVEL_MASK], maskBits);
        if (((dust_vs_word(DUST_VS_CTRL) >> 25u) & 1u) != 0u) {
            uint ev = dust_pk[0];
            if (ev != 0u) atomicAdd(pc.pyramid.v[DUST_PYR_VOLUME_EVALS], ev);
        }
    }
    DUST_PROF_END(vdraw)
#ifdef DUST_PROFILE
    if (profHit) {
        atomicAdd(pc.pyramid.v[DUST_PYR_PROF_PAIR], 1u);
        atomicAdd(pc.pyramid.v[DUST_PYR_PROF_PAIR + 1u], (clock2x32ARB().x - dust_prof_t0_vplan) >> 8u);
    }
#endif
}
