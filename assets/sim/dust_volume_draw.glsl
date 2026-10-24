// @assets/sim/dust_volume_draw.glsl
//
// The volumetric packet draw of dust_draw_volume.comp (perf9_O: a workgroup of 64 lanes draws its
// 64 records one after another, the lanes cooperating on each; dust_split.comp never draws a
// volumetric packet itself), on the packet and seed the splitter emitted, so the pyramid is the
// same word for word whichever queue held the record. The shader declares the push-constant block
// `pc` (dust::DustSplatPushConstants) before including this. Include after dust_scatter.glsl and
// dust_draw_texel.glsl.
//
// The arithmetic is dust::volume_scatter's, as the former one-lane draw ran it (the derivation, the
// units and the choices are documented there): the plan (dust::volume_plan: the axes, the pieces per
// axis, the level from the angular sigma plus the frame's bias, coarsened until the evaluations fit
// DUST_VOLUME_MAX_EVALS, then the pieces merged in pairs), pass A (every piece at its own level:
// its class, order and texels), pass B (each piece at its level, its row segments drawn by the
// texel's exact ray integral, its sub-rays, its lines or its small points). Every value is computed
// by the same expressions in the same order; only which lane computes it, and where it waits
// between the stages (a shared word, or a word of the workgroup's scratch rows), change. The
// deposits are order-free atomics (dust_atomic_add_sat's proof; the colour dither is hashed per
// texel word, the depth a minimum), the peak a maximum, the level mask an OR and the evaluation
// counts integer sums, so the pyramid and the frame words are the one-lane draw's. Stages (DESIGN.md):
//   phase 0   a lane per record: the plan's pieces per axis (dust_volume_plan_axes) and the support's
//             renorm, into the workgroup's scratch rows;
//   R1        per record, three lanes: the pieces' covariance (dust_volume_piece_cov), the packet's
//             3-sigma bound, the axes (shared words);
//   pass A    a lane per piece: dust_volume_piece_at in four steps through the piece's scratch row
//             (A1 the mean, its whitened offset and k; A2 the projected box; A3 on / full / cap;
//             A4 its own level), A5 its level, class, order and texels at the bias; lane 0 sums them
//             and runs the cap loops (A5 again at a coarser bias, or the covariance and A1-A5 again
//             with fewer pieces), then the sub-ray limit K and the in-frame gate;
//   pass B    a lane per piece: its drawn level, class, order and range (pass A's code, or the top /
//             gated level's class), a small piece's inputs, weights and points (S1-S3; an edge piece
//             to the edge pass);
//   texels    groups of up to DUST_T_SEGS row segments of one piece (dust_volume_segment*'s rows and
//             segments): the units (a texel, a texel's sub-ray, or a texel's line node) a lane each,
//             the texel's tau reduced in the loop's order, the segment's carry walk, the deposits.

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
// a piece's class, orders and level packed in 16 bits (class | (kw - 1) << 2 | (kb - 1) << 4 |
// (line axis - 1) << 6 | level << 8; kw the order its level asks for, kb its order bound), two
// pieces a word (step F10)
uint dust_volume_class_code(DustVolumePiece q, uvec2 ck, uint kb, uint lev) {
    uint ax = (ck.x == DUST_VOLUME_CLASS_LINE && q.line != 0u) ? q.line - 1u : 0u;
    return ck.x | ((ck.y - 1u) << 2u) | ((kb - 1u) << 4u) | (ax << 6u) | (lev << 8u);
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
    [[unroll]] for (uint i5_ = 0u; i5_ < DUST_VOLUME_SUBRAY_ORDER_MAX - 1u; ++i5_)
        if (k > 1u && dust_volume_texel_cost(c, k) * t > t1) k--;
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
// the frame's top-level regime (frame-uniform, from the pyramid header): the bias puts every piece
// on the last level and that level is 1 x 1 (topOnly); known to every path unless the reference
// mode DUST_FRAME_TOP_PASS_A asks for pass A in full (topKnown; dust_draw_volume.comp main sets the
// accumulate mode from it: round 8 step F2)
bool dust_volume_top_only(uint levels, uint bias) {
    uint topW = pc.pyramid.v[DUST_PYR_TABLE + 3u * (levels - 1u) + 1u];
    uint topH = pc.pyramid.v[DUST_PYR_TABLE + 3u * (levels - 1u) + 2u];
    return bias + 1u >= levels && topW == 1u && topH == 1u;
}
bool dust_volume_top_known(uint levels, uint bias) {
    return dust_volume_top_only(levels, bias) && (pc.pyramid.v[DUST_PYR_FLAGS] & DUST_FRAME_TOP_PASS_A) == 0u;
}

// ---- the workgroup's state ------------------------------------------------------------------
// Shared words (1 KB a workgroup; above 1 KB RADV's scheduler spends VGPRs up to the LDS-limited
// wave count). A workgroup-uniform value is read from them at each use: held in a register, it costs
// every lane of Mali-G52 a register. Every branch around a barrier tests words no lane writes
// between the barriers enclosing the test.
const uint DUST_VW_LANES = 64u;                       // the workgroup (local_size_x)
shared uint dust_v_unit[DUST_VW_LANES];               // pass A's codes and flags; the texels' unit values
shared uint dust_v_rec[32];                           // the record's words (DUST_VQUEUE_GET's order)
shared uint dust_v_st[32];                            // the record's state (DUST_VS_*)
shared uint dust_v_seg[2u * 2u * DUST_T_SEGS];        // two groups: [par * 8 + s] key row << 16 | col, [par * 8 + 4 + s] texels
shared uint dust_v_sp[2u * DUST_T_SEGS];              // the walked group's segments: [s] peak bits, [4 + s] written
#ifdef DUST_VW_STAGE
shared uint dust_v_ctl[17];                           // DUST_VC_* (staged: RES, SR)
#else
shared uint dust_v_ctl[15];                           // DUST_VC_*
#endif
shared uint dust_v_gx[28];                            // the group's piece (DUST_VG_*)
// dust_v_st: the pieces' Cholesky factor and diagonal (dust_volume_piece_cov), the packet's bound,
// the renorm, the depth bits, the control word, the levels, the cap loops' stage, the sums
const uint DUST_VS_L = 0u;        // 6
const uint DUST_VS_SKD = 6u;      // 3
const uint DUST_VS_BOUND = 9u;    // 12: l[6], qe, g
const uint DUST_VS_RENORM = 21u;
const uint DUST_VS_DEPTH = 22u;
const uint DUST_VS_CTL = 23u;     // nA | nB << 5 | K << 10 | mode << 13 | hasB << 15 | gated << 16 | cappedA << 17 | cappedB << 18 | top << 19
const uint DUST_VS_LEVELS = 24u;
const uint DUST_VS_STAGE = 25u;   // extra | stage << 8 (0 first pass A, 1 the bias loop, 2 the piece loop)
const uint DUST_VS_EVALS = 26u;   // the evaluations at K (the gate's add)
const uint DUST_VS_PEAK = 27u;    // the largest tau per px^2 (float bits, 0: none)
const uint DUST_VS_MASK = 28u;    // the levels written
const uint DUST_VS_TOPEV = 29u;   // the top-level path's evaluations
const uint DUST_VS_STATE = 30u;   // the cap loops' decision: 2 A5 again, 3 not drawn, 4 fewer pieces, 5 drawn
const uint DUST_VS_FPX = 31u;
// dust_v_ctl
const uint DUST_VC_COUNT = 0u;    // the workgroup's records
const uint DUST_VC_RA = 1u;       // the workgroup's first record's word 0 (base + header + r0)
const uint DUST_VC_CAP = 3u;      // the queue's capacity (the records' column stride)
const uint DUST_VC_J = 4u;        // the record being drawn (read at each use: a loop counter of the
                                  // record loop would be strength-reduced into address registers)
const uint DUST_VC_CP = 5u;       // the cursor's piece
const uint DUST_VC_CY = 6u;       // its row (0xFFFFFFFF: the piece's first)
const uint DUST_VC_CX = 7u;       // its next column
const uint DUST_VC_GP = 8u;       // the group's piece
const uint DUST_VC_N0 = 9u;       // the groups' segment counts (2: par 0, 1)
const uint DUST_VC_NEW = 11u;     // the group's piece is not the one in dust_v_gx
const uint DUST_VC_G = 12u;       // the texel phase's group
const uint DUST_VC_CH = 13u;      // the group's pass's first slot
const uint DUST_VC_PH = 14u;      // the driver's step (DUST_VW_PH_*)
const uint DUST_VC_RES = 15u;     // staged: the phase the record resumes at (DUST_VW_PH_NEXT: done)
const uint DUST_VC_SR = 16u;      // staged: the workgroup's first scratch row
// dust_v_gx: pass A's axes (0-2 axMean, 3-5 segA, 6-8 segB); in the texel phase the group's
// piece: its scratch words 0-8 (DUST_VP_*), then a small piece's three points (mx my ix iy rho m)
const uint DUST_VG_PT = 9u;
const uint DUST_VG_ZERO = 27u;    // always 0 (set once): a header word's address through it is not a
                                  // loop-invariant constant (Mali-G52 would hold each hoisted address
                                  // in two registers of every lane through the record loop)

#define DUST_VRW(w) dust_v_rec[w]
#define DUST_VRF(w) uintBitsToFloat(dust_v_rec[w])
#define DUST_VSF(w) uintBitsToFloat(dust_v_st[w])
#define DUST_VGF(w) uintBitsToFloat(dust_v_gx[w])
#define DUST_V_R0 vec3(pc.mvp[0][0], pc.mvp[1][0], pc.mvp[2][0])
#define DUST_V_R1 vec3(pc.mvp[0][1], pc.mvp[1][1], pc.mvp[2][1])
#define DUST_V_R3 vec3(pc.mvp[0][3], pc.mvp[1][3], pc.mvp[2][3])
#define DUST_V_TV vec3(pc.mvp[3][0], pc.mvp[3][1], pc.mvp[3][3])
#define DUST_V_W float(DUST_PW(DUST_PYR_WIDTH))
#define DUST_V_H float(DUST_PW(DUST_PYR_HEIGHT))

// The workgroup's scratch, in shared memory (pointer-free: the Adreno 619 compiler crashed on every
// buffer reference built from a computed address in the edge pass; no scratch rows in the buffer, so
// no host sizing): a row a piece of the record being drawn (dust_v_pr), the stash (dust_v_stash: the
// small pieces' points in pass B, the line units' four words a lane in the texel phase), the
// record's axS (dust_v_gx 9-14 from R1 to the last covariance: the texel phase's points reuse them),
// the accumulate mode's sums (dust_v_acc, the .sg / .prof blobs). Workgroup memory: a word written
// before a barrier() is visible to every lane after it (OpControlBarrier Workgroup/Workgroup,
// AcquireRelease | WorkgroupMemory); no buffer word passes between lanes.
const uint DUST_VW_PIECE_WORDS = 25u;
shared uint dust_v_pr[DUST_VOLUME_PIECES_MAX * DUST_VW_PIECE_WORDS];
shared uint dust_v_stash[4u * DUST_VW_LANES];
const uint DUST_VG_AXS = 9u;
// a piece's row
const uint DUST_VP_QW = 0u;       // 3: L^-1 (eye - mean)
const uint DUST_VP_K = 3u;        // its k
const uint DUST_VP_FLAGS = 4u;    // full | on << 1 | cap << 2 | line << 3 | (ls valid) << 5 | lp << 8 | ls << 16
const uint DUST_VP_CODE = 5u;     // pass A's code (dust_volume_class_code)
const uint DUST_VP_DESC = 6u;     // pass B's level | class << 8 | order << 10 | drawn << 13
const uint DUST_VP_RX = 7u;       // its range's x0 | x1 << 16
const uint DUST_VP_RY = 8u;       // y0 | y1 << 16
const uint DUST_VP_MEAN = 9u;     // 3: its mean (pass A)
const uint DUST_VP_BBOX = 12u;    // 4: its projected box (pass A, pass B's range)
const uint DUST_VP_BLEV = 20u;    // 4: pass B's level, class, order, order bound (its first step)
const uint DUST_VP_SMALL = 9u;    // 16: a small piece's inputs, dust_small_edge_put's words 0-15 (pass B)
// a header word, a plain read (its index through the shared zero: pyramid + 4 w is then not a
// loop-invariant address the compiler hoists into a register pair)
#define DUST_PW(w) pc.pyramid.v[(w) + dust_v_gx[DUST_VG_ZERO]]
#define DUST_VWP(k, w) dust_v_pr[(k) * DUST_VW_PIECE_WORDS + (w)]
#define DUST_VWPF(k, w) uintBitsToFloat(DUST_VWP(k, w))
#ifdef DUST_VW_STAGE
#ifdef DUST_PROFILE
#error "the staged form has no profile variant (the .prof blob is the single-dispatch form)"
#endif
// The staged form's scratch row of a record (DUST_VW_SX_WORDS words, the record's index in its
// drain's queue: the main queue's rows, then the spill's, after the deferred edge regions; host:
// dust::dust_queue_offsets' vscratch): the state words, the resume phase, the axes, pass B's
// codes and evaluations (S3 -> S4), the piece rows (S2 -> S3 in shared memory between; S4 and S5
// read and write them here: a lane touches only its own piece's row there, or reads)
const uint DUST_VW_SX_ST = 0u;      // 32: dust_v_st
const uint DUST_VW_SX_RES = 32u;    // the resume phase
const uint DUST_VW_SX_GX = 36u;     // 15: dust_v_gx 0-14 (the axes, axS)
const uint DUST_VW_SX_UNIT = 64u;   // 16: dust_v_unit 0-15 (pass B's codes)
const uint DUST_VW_SX_TROW = 80u;   // 16: dust_t_row 0-15 (pass B's evaluations)
const uint DUST_VW_SX_PR = 96u;     // 16 x 25: the piece rows
const uint DUST_VW_SX_PT = 496u;    // 16 x 3 x 6: the small pieces' points (PBSc; mx my ix iy m rho)
const uint DUST_VW_SX_WORDS = 784u;
#define DUST_VW_ROW (dust_v_ctl[DUST_VC_SR] + dust_v_ctl[DUST_VC_J] * DUST_VW_SX_WORDS)
#if DUST_VW_STAGE == 3 && defined(DUST_VW_PART) && DUST_VW_PART == 2
#define DUST_VW_S3B 1  // S3 part 2: the piece rows in the scratch row too
#endif
#if DUST_VW_STAGE >= 4 || defined(DUST_VW_S3B)
#undef DUST_VWP
#define DUST_VWP(k, w) pc.pyramid.v[DUST_VW_ROW + DUST_VW_SX_PR + (k) * DUST_VW_PIECE_WORDS + (w)]
#endif
#endif
// dust_vqueue_base, dust_vqueue_capacity, dust_vspill_base, dust_vedge_base without a buffer
// reference parameter (plain pc.pyramid reads; a helper taking a DustUintBuffer is private memory
// on the Adreno 619)
uint dust_vw_vqueue_base() { return pc.pyramid.v[DUST_PYR_QUEUE_BASE] + DUST_QUEUE_WORDS; }
uint dust_vw_vqueue_capacity(uint base) {
    uint c = pc.pyramid.v[base + 4u];
    return c == 0u ? DUST_VQUEUE_CAPACITY : min(c, DUST_VQUEUE_CAPACITY);
}
uint dust_vw_vspill_base() {
    uint vbase = dust_vw_vqueue_base();
    return vbase + pc.pyramid.v[vbase + DUST_VQUEUE_SPILL_OFFSET];
}
uint dust_vw_vedge_base(bool odd) {
    uint vbase = dust_vw_vqueue_base();
    uint n = pc.pyramid.v[vbase + DUST_VQUEUE_RECORDS];
    n = n == 0u ? DUST_VQUEUE_CAPACITY : min(n, DUST_VQUEUE_CAPACITY);
    uint e0 = dust_vw_vspill_base() + DUST_VQUEUE_HEADER_WORDS + dust_vspill_records_of(n) * DUST_VQUEUE_RECORD_WORDS;
    return odd ? e0 + DUST_VQUEUE_HEADER_WORDS + dust_vedge_records_of(n) * DUST_VEDGE_RECORD_WORDS : e0;
}
#ifdef DUST_VW_STAGE
// the drain's scratch rows: the main queue's n rows after edge region 1, then the spill's
uint dust_vw_vscratch_base(bool spill) {
    uint vbase = dust_vw_vqueue_base();
    uint n = pc.pyramid.v[vbase + DUST_VQUEUE_RECORDS];
    n = n == 0u ? DUST_VQUEUE_CAPACITY : min(n, DUST_VQUEUE_CAPACITY);
    uint s0 = dust_vw_vedge_base(true) + DUST_VQUEUE_HEADER_WORDS + dust_vedge_records_of(n) * DUST_VEDGE_RECORD_WORDS;
    return spill ? s0 + n * DUST_VW_SX_WORDS : s0;
}
#endif

// the record's packet from its shared words
DustPacket dust_vw_packet() {
    DustPacket p;
    p.mean = vec3(DUST_VRF(0u), DUST_VRF(1u), DUST_VRF(2u));
    p.S[0] = DUST_VRF(3u); p.S[1] = DUST_VRF(4u); p.S[2] = DUST_VRF(5u);
    p.S[3] = DUST_VRF(6u); p.S[4] = DUST_VRF(7u); p.S[5] = DUST_VRF(8u);
    p.seg = vec3(DUST_VRF(9u), DUST_VRF(10u), DUST_VRF(11u));
    p.flux = DUST_VRF(12u);
    p.rho = vec2(DUST_VRF(13u), DUST_VRF(14u));
    p.segT = vec3(DUST_VRF(15u), DUST_VRF(16u), DUST_VRF(17u));
    p.openStart = false;
    p.openEnd = false;
    return p;
}
// the record's photometry without the dust_phot_* globals (a global written in a branch and read
// later crashed the Adreno 619 compiler): dust_volume_weight, dust_small_weights with the record's
// Sun direction (words 27-29), r_h (30), pc.albedo, the phase table read plainly
float dust_vw_lut(uint i) { return uintBitsToFloat(pc.pyramid.v[DUST_PYR_PHASE_LUT + i]); }
float dust_vw_radiance_weight(float albedo, vec3 sunDir, vec3 toEye, float rhAu) {
    float n = dot(toEye, toEye);
    float cosA = n > 0.0 ? dot(sunDir, toEye) / sqrt(n) : 1.0;
    float alpha = acos(clamp(cosA, -1.0, 1.0));
    float r = max(rhAu, 1e-6);
    float deg = clamp(alpha * (180.0 / DUST_PI_F32), 0.0, 180.0);
    uint i = min(uint(deg), 179u);
    float f = deg - float(i);
    float a = dust_vw_lut(i);
    float b = dust_vw_lut(i + 1u);
    return albedo * (a + (b - a) * f) / (r * r);
}
#define DUST_VW_SUN vec3(DUST_VRF(27u), DUST_VRF(28u), DUST_VRF(29u))
float dust_vw_volume_weight(vec3 meanRef, vec3 x, vec3 u) {
    vec3 d = x - meanRef;
    vec3 rVec = DUST_VW_SUN * (-DUST_VRF(30u) * DUST_AU_M) + d;
    float rn = max(sqrt(dot(rVec, rVec)), 1.0);
    return dust_vw_radiance_weight(pc.albedo, rVec * (-1.0 / rn), -u, rn / DUST_AU_M);
}
void dust_vw_small_sun(vec3 x, vec3 meanRef, out vec3 sunDir, out float rAu) {
    vec3 rVec = DUST_VW_SUN * (-DUST_VRF(30u) * DUST_AU_M) + (x - meanRef);
    float rn = max(sqrt(dot(rVec, rVec)), 1.0);
    sunDir = rVec * (-1.0 / rn);
    rAu = rn / DUST_AU_M;
}
void dust_vw_small_weights(vec3 x0, vec3 x1, vec3 x2, vec3 eye, vec3 meanRef, out float wr0, out float wr1,
                           out float wr2, out vec3 sun0, out float r20) {
    wr0 = 1.0; wr1 = 1.0; wr2 = 1.0; sun0 = vec3(0.0); r20 = 1.0;
    float albedo = pc.albedo;
    if (!(albedo > 0.0)) return;
    vec3 sun1, sun2;
    float rAu0, rAu1, rAu2;
    dust_vw_small_sun(x0, meanRef, sun0, rAu0);
    sun1 = sun0; sun2 = sun0; rAu1 = rAu0; rAu2 = rAu0;
    float r0_ = max(rAu0, 1e-6), r1_ = max(rAu1, 1e-6), r2_ = max(rAu2, 1e-6);
    r20 = r0_ * r0_;
    float g0 = dust_small_phase_deg(x0, eye, sun0);
    float g1 = dust_small_phase_deg(x1, eye, sun1);
    float g2 = dust_small_phase_deg(x2, eye, sun2);
    uint i0 = min(uint(g0), 179u), i1 = min(uint(g1), 179u), i2 = min(uint(g2), 179u);
    float a0 = dust_vw_lut(i0), b0 = dust_vw_lut(i0 + 1u);
    float a1 = a0, b1 = b0, a2 = a0, b2 = b0;
    if (i1 != i0) { a1 = dust_vw_lut(i1); b1 = dust_vw_lut(i1 + 1u); }
    if (i2 != i0) { a2 = dust_vw_lut(i2); b2 = dust_vw_lut(i2 + 1u); }
    wr0 = albedo * (a0 + (b0 - a0) * (g0 - float(i0))) / r20;
    wr1 = albedo * (a1 + (b1 - a1) * (g1 - float(i1))) / (r1_ * r1_);
    wr2 = albedo * (a2 + (b2 - a2) * (g2 - float(i2))) / (r2_ * r2_);
}
// the pieces' Cholesky factor (dust_volume_piece_cov's Lk) from the shared words
#define DUST_VW_LK(L_) float L_[6]; L_[0] = DUST_VSF(DUST_VS_L + 0u); L_[1] = DUST_VSF(DUST_VS_L + 1u); \
    L_[2] = DUST_VSF(DUST_VS_L + 2u); L_[3] = DUST_VSF(DUST_VS_L + 3u); L_[4] = DUST_VSF(DUST_VS_L + 4u); \
    L_[5] = DUST_VSF(DUST_VS_L + 5u);



// ---- R1: the record's axes, plan, covariance and bound ---------------------------------------
// dust_volume_plan_axes in two lane-0 steps through the shared words (the same expressions on the
// same values): AXES its projection test, fPx, the axis mode and the axes (axMean, segA, segB into
// dust_v_gx 0-8, axS into dust_v_gx 9-14, an element at a time: the three candidate arrays held
// together cost Mali-G52 its 32-register mode), the record's state words; PLAN the pieces per axis
// from the axes, the top-level path's detail cap, the support's renorm (dust_volume_draw's)
// axS element i: lessT_i = S_i - kf ht_a ht_b; time: lessT_i + kf hc_a hc_b; grid: lessT_i; chord: S_i
#define DUST_VW_AXS_EL(i, s_, a_, b_) { float s = s_; float lt = s - kf * ht.a_ * ht.b_; \
    float v = mode == DUST_VOLUME_AXIS_TIME ? lt + kf * hc.a_ * hc.b_ : (mode == DUST_VOLUME_AXIS_GRID ? lt : s); \
    dust_v_gx[DUST_VG_AXS + (i)] = floatBitsToUint(v); }
void dust_vw_axes() {
    vec3 r0 = DUST_V_R0, r1 = DUST_V_R1, r3 = DUST_V_R3;
    uint levels = DUST_PW(DUST_PYR_LEVELS);
    uint split = DUST_PW(DUST_PYR_SPLIT_DEPTH);
    bool topOnly = dust_volume_top_only(levels, (split >> 8u) & 0xFFu);
    bool top = topOnly && (DUST_PW(DUST_PYR_FLAGS) & DUST_FRAME_TOP_PASS_A) == 0u;  // dust_volume_top_known
    uint axis = (split >> 16u) & 0x3u;
    vec3 segT = vec3(DUST_VRF(15u), DUST_VRF(16u), DUST_VRF(17u));
    uint mode = (axis != DUST_VOLUME_AXIS_CHORD && dot(segT, segT) > 0.0) ? axis : DUST_VOLUME_AXIS_CHORD;
    dust_v_st[DUST_VS_CTL] = (mode << 13u) | ((top ? 1u : 0u) << 19u) | ((topOnly ? 1u : 0u) << 20u);
    dust_v_st[DUST_VS_STAGE] = 0u;
    dust_v_st[DUST_VS_PEAK] = 0u;
    dust_v_st[DUST_VS_MASK] = 0u;
    dust_v_st[DUST_VS_TOPEV] = 0u;
    dust_v_st[DUST_VS_LEVELS] = levels;
    float W = DUST_V_W, H = DUST_V_H;
    float n3 = dot(r3, r3);
    float fPx = 0.5 * min(W * sqrt(dot(r0, r0)), H * sqrt(dot(r1, r1))) / sqrt(n3);
    // (dust_volume_plan_axes' false: no projection, the packet not drawn)
    dust_v_st[DUST_VS_STATE] = (n3 > 0.0 && fPx > 0.0) ? 1u : 3u;
    vec3 mean = vec3(DUST_VRF(0u), DUST_VRF(1u), DUST_VRF(2u));
    vec3 seg = vec3(DUST_VRF(9u), DUST_VRF(10u), DUST_VRF(11u));
    vec3 q0 = pc.eyeLocal.xyz - mean;
    vec3 rSeg = q0 - seg * dust_segment_param(q0, seg);
    float dist = sqrt(dot(rSeg, rSeg));
    float rel = dist / fPx;
    float varF = DUST_PIXEL_FILTER_VAR * rel * rel;
    float kf = DUST_CHORD_VAR_FACTOR;
    vec3 ht = segT * 0.5;
    vec3 hc = seg * 0.5;
    vec3 axMean = mean;
    vec3 segA = seg, segB = vec3(0.0);
    if (mode == DUST_VOLUME_AXIS_TIME) {
        vec2 rho = vec2(DUST_VRF(13u), DUST_VRF(14u));
        float ubar = rho.x * 0.5 + (rho.y - rho.x) / 3.0;
        axMean = mean + seg * ubar;
        segA = vec3(0.0);
        segB = segT;
    } else if (mode == DUST_VOLUME_AXIS_GRID) {
        segB = segT;
    }
    dust_v_gx[0] = floatBitsToUint(axMean.x); dust_v_gx[1] = floatBitsToUint(axMean.y); dust_v_gx[2] = floatBitsToUint(axMean.z);
    dust_v_gx[3] = floatBitsToUint(segA.x); dust_v_gx[4] = floatBitsToUint(segA.y); dust_v_gx[5] = floatBitsToUint(segA.z);
    dust_v_gx[6] = floatBitsToUint(segB.x); dust_v_gx[7] = floatBitsToUint(segB.y); dust_v_gx[8] = floatBitsToUint(segB.z);
    dust_v_st[DUST_VS_FPX] = floatBitsToUint(fPx);
    DUST_VW_AXS_EL(0u, DUST_VRF(3u) + varF, x, x)
    DUST_VW_AXS_EL(1u, DUST_VRF(4u), x, y)
    DUST_VW_AXS_EL(2u, DUST_VRF(5u), x, z)
    DUST_VW_AXS_EL(3u, DUST_VRF(6u) + varF, y, y)
    DUST_VW_AXS_EL(4u, DUST_VRF(7u), y, z)
    DUST_VW_AXS_EL(5u, DUST_VRF(8u) + varF, z, z)
}
#define DUST_VW_AXSF(i) uintBitsToFloat(dust_v_gx[DUST_VG_AXS + (i)])
#define DUST_VW_GX3(w) vec3(DUST_VGF(w), DUST_VGF((w) + 1u), DUST_VGF((w) + 2u))
// lane 0: dust_volume_plan_axes' pieces per axis (packed into the control word: nA | nB << 5 |
// hasB << 15 | cappedA << 17 | cappedB << 18), the top-level path's detail cap, the renorm
void dust_vw_plan() {
    uint ctl = dust_v_st[DUST_VS_CTL];
    uint mode = (ctl >> 13u) & 3u;
    bool hasB = mode == DUST_VOLUME_AXIS_TIME || mode == DUST_VOLUME_AXIS_GRID;
    float axS[6];
    axS[0] = DUST_VW_AXSF(0u); axS[1] = DUST_VW_AXSF(1u); axS[2] = DUST_VW_AXSF(2u);
    axS[3] = DUST_VW_AXSF(3u); axS[4] = DUST_VW_AXSF(4u); axS[5] = DUST_VW_AXSF(5u);
    float Lb[6];
    dust_cholesky3(axS, Lb);
    vec3 dw = dust_cholesky_solve3(Lb, DUST_VW_GX3(3u));
    float lenW = sqrt(dot(dw, dw)) / DUST_VOLUME_PIECE_SIGMAS;
    bool capped = lenW > float(DUST_VOLUME_PIECES_MAX);
    uint nA = capped ? DUST_VOLUME_PIECES_MAX : max(uint(ceil(lenW)), 1u);
    float cvarA = capped ? DUST_VOLUME_PIECE_VAR_CAPPED : 1.0 / 12.0;
    uint nB = 1u;
    float cvarB = 1.0 / 12.0;
    if (hasB) {
        vec3 dwb = dust_cholesky_solve3(Lb, DUST_VW_GX3(6u));
        float lenB = sqrt(dot(dwb, dwb)) / DUST_VOLUME_PIECE_SIGMAS;
        bool cappedB = lenB > float(DUST_VOLUME_PIECES_MAX);
        nB = cappedB ? DUST_VOLUME_PIECES_MAX : max(uint(ceil(lenB)), 1u);
        cvarB = cappedB ? DUST_VOLUME_PIECE_VAR_CAPPED : 1.0 / 12.0;
    }
    [[unroll]] for (uint i1_ = 0u; i1_ < 4u; ++i1_) {
        if (nA * nB > DUST_VOLUME_PIECES_MAX) {
            if (nB > 1u) { nB = (nB + 1u) / 2u; cvarB = DUST_VOLUME_PIECE_VAR_CAPPED; }
            else { nA = (nA + 1u) / 2u; cvarA = DUST_VOLUME_PIECE_VAR_CAPPED; }
        }
    }
    bool cA = cvarA == DUST_VOLUME_PIECE_VAR_CAPPED, cB = cvarB == DUST_VOLUME_PIECE_VAR_CAPPED;
    // the volume detail (dust_volume_draw's: a top-level packet cut into at most that many pieces,
    // fewer along seg_t first)
    if (((ctl >> 20u) & 1u) != 0u) {  // (dust_volume_top_only, from AXES)
        uint detail = max(16u >> ((pc.flags >> DUST_FLAGS_VOLUME_DETAIL_SHIFT) & 7u), 1u);
        bool dn_ = false;
        [[unroll]] for (uint i2_ = 0u; i2_ < 8u; ++i2_) {
            if (!dn_) {
                if (nA * nB <= detail) dn_ = true;
                else if (nB > 1u) { nB = (nB + 1u) / 2u; cB = true; }
                else if (nA > 1u) { nA = (nA + 1u) / 2u; cA = true; }
                else dn_ = true;
            }
        }
    }
    dust_v_st[DUST_VS_CTL] = (ctl & ~(1u << 20u)) | nA | (nB << 5u) | ((hasB ? 1u : 0u) << 15u) | ((cA ? 1u : 0u) << 17u)
                           | ((cB ? 1u : 0u) << 18u);
}
// dust_support_mass (dust_common.glsl), its 16 samples marked [[unroll]]
float dust_vw_support_mass(DustPacket p, vec4 s) {
    if (!dust_has_support(s)) return 1.0;
    float sig = sqrt(max(dust_var_along(p.S, s.xyz), 0.0));
    float m0 = dot(s.xyz, p.mean) - s.w, ds = dot(s.xyz, p.seg);
    float acc = 0.0, wsum = 0.0;
    [[unroll]] for (int k = 0; k < 16; ++k) {
        float u = (float(k) + 0.5) / 16.0;
        float w = p.rho.x + (p.rho.y - p.rho.x) * u;
        float x = m0 + ds * u;
        float f = sig > 0.0 ? 0.5 * (1.0 + dust_erf(x / (sig * 1.4142135623730951))) : (x >= 0.0 ? 1.0 : 0.0);
        acc += w * f;
        wsum += w;
    }
    return wsum > 0.0 ? acc / wsum : 1.0;
}
// lane 2 of COV: the support's renorm (PLAN's tail before)
void dust_vw_renorm() {
    DustPacket p = dust_vw_packet();
    vec4 support = vec4(DUST_VRF(23u), DUST_VRF(24u), DUST_VRF(25u), DUST_VRF(26u));
    float renorm = dust_has_support(support) ? 1.0 / max(dust_vw_support_mass(p, support), DUST_SUPPORT_MASS_MIN) : 1.0;
    dust_v_st[DUST_VS_RENORM] = floatBitsToUint(renorm);
}
// lane 0: the pieces' covariance at the control word's nA, nB (dust_volume_piece_cov on the
// record's axes) into DUST_VS_L / DUST_VS_SKD
void dust_vw_cov() {
    float axS[6];
    axS[0] = DUST_VW_AXSF(0u); axS[1] = DUST_VW_AXSF(1u); axS[2] = DUST_VW_AXSF(2u);
    axS[3] = DUST_VW_AXSF(3u); axS[4] = DUST_VW_AXSF(4u); axS[5] = DUST_VW_AXSF(5u);
    uint ctl = dust_v_st[DUST_VS_CTL];
    float cvarA = ((ctl >> 17u) & 1u) != 0u ? DUST_VOLUME_PIECE_VAR_CAPPED : 1.0 / 12.0;
    float cvarB = ((ctl >> 18u) & 1u) != 0u ? DUST_VOLUME_PIECE_VAR_CAPPED : 1.0 / 12.0;
    float Lk[6];
    vec3 skd;
    dust_volume_piece_cov(axS, DUST_VW_GX3(3u), ctl & 31u, cvarA, DUST_VW_GX3(6u), (ctl >> 5u) & 31u, cvarB,
                          ((ctl >> 15u) & 1u) != 0u, Lk, skd);
    dust_v_st[DUST_VS_L + 0u] = floatBitsToUint(Lk[0]); dust_v_st[DUST_VS_L + 1u] = floatBitsToUint(Lk[1]);
    dust_v_st[DUST_VS_L + 2u] = floatBitsToUint(Lk[2]); dust_v_st[DUST_VS_L + 3u] = floatBitsToUint(Lk[3]);
    dust_v_st[DUST_VS_L + 4u] = floatBitsToUint(Lk[4]); dust_v_st[DUST_VS_L + 5u] = floatBitsToUint(Lk[5]);
    dust_v_st[DUST_VS_SKD + 0u] = floatBitsToUint(skd.x); dust_v_st[DUST_VS_SKD + 1u] = floatBitsToUint(skd.y);
    dust_v_st[DUST_VS_SKD + 2u] = floatBitsToUint(skd.z);
}
// lane 1: the packet's 3-sigma bound (dust_volume_plan_axes' second half: pl.bound by the axis
// mode) into DUST_VS_BOUND
void dust_vw_bound() {
    float bs[6];
    bs[0] = DUST_VW_AXSF(0u); bs[1] = DUST_VW_AXSF(1u); bs[2] = DUST_VW_AXSF(2u);
    bs[3] = DUST_VW_AXSF(3u); bs[4] = DUST_VW_AXSF(4u); bs[5] = DUST_VW_AXSF(5u);
    uint mode = (dust_v_st[DUST_VS_CTL] >> 13u) & 3u;
    vec3 start = DUST_VW_GX3(0u), sg = DUST_VW_GX3(3u);
    if (mode == DUST_VOLUME_AXIS_TIME) {
        sg = DUST_VW_GX3(6u);
        start = DUST_VW_GX3(0u) - sg * 0.5;
    } else if (mode == DUST_VOLUME_AXIS_GRID) {
        float kf = DUST_CHORD_VAR_FACTOR;
        vec3 hb = DUST_VW_GX3(6u) * 0.5;
        bs[0] = bs[0] + kf * hb.x * hb.x;
        bs[1] = bs[1] + kf * hb.x * hb.y;
        bs[2] = bs[2] + kf * hb.x * hb.z;
        bs[3] = bs[3] + kf * hb.y * hb.y;
        bs[4] = bs[4] + kf * hb.y * hb.z;
        bs[5] = bs[5] + kf * hb.z * hb.z;
    }
    DustVolumeBound bd = dust_volume_bound(bs, start, sg, pc.eyeLocal.xyz);
    dust_v_st[DUST_VS_BOUND + 0u] = floatBitsToUint(bd.l[0]); dust_v_st[DUST_VS_BOUND + 1u] = floatBitsToUint(bd.l[1]);
    dust_v_st[DUST_VS_BOUND + 2u] = floatBitsToUint(bd.l[2]); dust_v_st[DUST_VS_BOUND + 3u] = floatBitsToUint(bd.l[3]);
    dust_v_st[DUST_VS_BOUND + 4u] = floatBitsToUint(bd.l[4]); dust_v_st[DUST_VS_BOUND + 5u] = floatBitsToUint(bd.l[5]);
    dust_v_st[DUST_VS_BOUND + 6u] = floatBitsToUint(bd.qe.x); dust_v_st[DUST_VS_BOUND + 7u] = floatBitsToUint(bd.qe.y);
    dust_v_st[DUST_VS_BOUND + 8u] = floatBitsToUint(bd.qe.z); dust_v_st[DUST_VS_BOUND + 9u] = floatBitsToUint(bd.g.x);
    dust_v_st[DUST_VS_BOUND + 10u] = floatBitsToUint(bd.g.y); dust_v_st[DUST_VS_BOUND + 11u] = floatBitsToUint(bd.g.z);
}

// ---- the accumulate mode's and the profile's sums (shared dust_v_acc) ---------------------------
// The accumulate mode (the .sg / .prof blobs, the top-level path: every deposit at the 1 x 1 top
// texel): the deposits' sums mod 2^32 and their wrap bits, the nearest depth, the split adds, the
// records' peak, mask and evaluations; lane 0 writes them once a workgroup (dust_vw_flush_acc, the
// former subgroup flush's semantics without a subgroup operation). The profile (.prof): lane 0's
// stage cycles, the records, the texels, the small pieces' fast path; flushed once a workgroup.
const uint DUST_VA_SUM = 0u;      // 4: counts, r, g, b (mod 2^32)
const uint DUST_VA_WRAP = 4u;     // bit w: word w passed 2^32
const uint DUST_VA_DEPTH = 5u;    // the nearest depth (0xFFFFFFFF: none)
const uint DUST_VA_SPLITS = 6u;
const uint DUST_VA_PEAK = 7u;
const uint DUST_VA_MASK = 8u;
const uint DUST_VA_EVALS = 9u;
const uint DUST_VA_THREAD = 10u;  // the profile: lane 0's cycles (the workgroup's records)
const uint DUST_VA_VPLAN = 11u;
const uint DUST_VA_VDRAW = 12u;
const uint DUST_VA_TEXELS = 13u;
const uint DUST_VA_SMALL_FAST = 14u;
const uint DUST_VA_T0 = 15u;      // the record's start
const uint DUST_VA_T1 = 16u;      // its texel phase's start
const uint DUST_VA_ALLTOP = 17u;  // the last pass A's allTop
const uint DUST_VA_MAX = 18u;     // the slowest record's cycles
const uint DUST_VA_WORDS = 19u;
#if defined(DUST_SUBGROUP) || defined(DUST_PROFILE)
shared uint dust_v_acc[DUST_VA_WORDS];
#endif
#define DUST_VA(w) dust_v_acc[w]
#ifdef DUST_PROFILE
#undef DUST_PROF_COUNT
#define DUST_PROF_COUNT(c, n) atomicAdd(DUST_VA(DUST_VA_##c), uint(n));
const uint DUST_VA_volume_texels = DUST_VA_TEXELS, DUST_VA_small_fast = DUST_VA_SMALL_FAST;
#define DUST_VW_CLOCK clock2x32ARB().x
#endif

// ---- pass A: a lane per piece ----------------------------------------------------------------
// dust_volume_piece_at(k, ..., withLevel = true) in four steps through the piece's row (each value
// loaded again at its use: a step's live set is its own), then A5 at the bias. The same operations
// on the same values in the same order.
// A1: the piece's mean, flux, whitened offset, k and full test
void dust_vw_a1(uint k) {
    uint ctl = dust_v_st[DUST_VS_CTL];
    uint nA = ctl & 31u, nB = (ctl >> 5u) & 31u;
    bool hasB = ((ctl >> 15u) & 1u) != 0u;
    float invN = 1.0 / float(nA);
    float invB = 1.0 / float(nB);
    uint i = k / nB, j = k % nB;
    float uk = (float(i) + 0.5) * invN;
    vec3 axMean = vec3(DUST_VGF(0u), DUST_VGF(1u), DUST_VGF(2u));
    vec3 segA = vec3(DUST_VGF(3u), DUST_VGF(4u), DUST_VGF(5u));
    vec3 meanK = axMean + segA * uk;
    vec2 rho = vec2(DUST_VRF(13u), DUST_VRF(14u));
    float fluxK = DUST_VRF(12u) * (rho.x + (rho.y - rho.x) * uk) * invN;
    if (hasB) {
        float vk = (float(j) + 0.5) * invB - 0.5;
        meanK = meanK + vec3(DUST_VGF(6u), DUST_VGF(7u), DUST_VGF(8u)) * vk;
        fluxK *= invB;
    }
    DUST_VW_LK(Lk)
    vec3 qw = dust_cholesky_solve3(Lk, pc.eyeLocal.xyz - meanK);
    bool full = dot(qw, qw) < DUST_VOLUME_SIGMAS * DUST_VOLUME_SIGMAS;
    float kq = DUST_VRF(19u) * fluxK / (2.0 * DUST_PI_F32 * (Lk[0] * Lk[3] * Lk[5]));
    DUST_VWP(k, DUST_VP_MEAN + 0u) = floatBitsToUint(meanK.x);
    DUST_VWP(k, DUST_VP_MEAN + 1u) = floatBitsToUint(meanK.y);
    DUST_VWP(k, DUST_VP_MEAN + 2u) = floatBitsToUint(meanK.z);
    DUST_VWP(k, DUST_VP_QW + 0u) = floatBitsToUint(qw.x);
    DUST_VWP(k, DUST_VP_QW + 1u) = floatBitsToUint(qw.y);
    DUST_VWP(k, DUST_VP_QW + 2u) = floatBitsToUint(qw.z);
    DUST_VWP(k, DUST_VP_K) = floatBitsToUint(kq);
    DUST_VWP(k, DUST_VP_FLAGS) = full ? 1u : 0u;
}
#define DUST_VW_MEANK(k) vec3(DUST_VWPF(k, DUST_VP_MEAN + 0u), DUST_VWPF(k, DUST_VP_MEAN + 1u), DUST_VWPF(k, DUST_VP_MEAN + 2u))
// a corner of A2's box (sx, sy, sz: the signs of ext's components, the loop's -ext / ext)
#define DUST_VW_CORNER(sx, sy, sz) if (!full) { \
    vec3 pt = vec3(meanK.x + sx ext.x, meanK.y + sy ext.y, meanK.z + sz ext.z); \
    float w = dot(r3, pt) + t3; \
    if (!(w > 0.0)) { full = true; } else { \
        float px = ((dot(r0, pt) + t0) / w * 0.5 + 0.5) * W; \
        float py = ((dot(r1, pt) + t1) / w * 0.5 + 0.5) * H; \
        x0 = min(x0, px); x1 = max(x1, px); \
        y0 = min(y0, py); y1 = max(y1, py); } }
// A2: the projected 3-sigma box (unless full; a corner not in front of the eye makes it full)
void dust_vw_a2(uint k) {
    bool full = (DUST_VWP(k, DUST_VP_FLAGS) & 1u) != 0u;
    vec3 skd = vec3(DUST_VSF(DUST_VS_SKD + 0u), DUST_VSF(DUST_VS_SKD + 1u), DUST_VSF(DUST_VS_SKD + 2u));
    vec3 ext = vec3(3.0 * sqrt(skd.x), 3.0 * sqrt(skd.y), 3.0 * sqrt(skd.z));
    float x0 = 1e30, x1 = -1e30, y0 = 1e30, y1 = -1e30;
    if (!full) {
        vec3 meanK = DUST_VW_MEANK(k);
        vec3 r0 = DUST_V_R0, r1 = DUST_V_R1, r3 = DUST_V_R3, t = DUST_V_TV;
        float t0 = t.x, t1 = t.y, t3 = t.z;
        float W = DUST_V_W, H = DUST_V_H;
        // the eight corners in the loop's order (c = 0 .. 7: x by bit 0, y bit 1, z bit 2), written
        // out (a corner after the first behind the eye is skipped, the loop's break)
        // (corners 4 .. 7 in A2B)
        DUST_VW_CORNER(-, -, -) DUST_VW_CORNER(+, -, -) DUST_VW_CORNER(-, +, -) DUST_VW_CORNER(+, +, -)
    }
    DUST_VWP(k, DUST_VP_BBOX + 0u) = floatBitsToUint(x0);
    DUST_VWP(k, DUST_VP_BBOX + 1u) = floatBitsToUint(x1);
    DUST_VWP(k, DUST_VP_BBOX + 2u) = floatBitsToUint(y0);
    DUST_VWP(k, DUST_VP_BBOX + 3u) = floatBitsToUint(y1);
    DUST_VWP(k, DUST_VP_FLAGS) = full ? 1u : 0u;
}
// A2B: corners 4 .. 7 of A2's box, from A2's box and full flag
void dust_vw_a2b(uint k) {
    bool full = (DUST_VWP(k, DUST_VP_FLAGS) & 1u) != 0u;
    vec3 skd = vec3(DUST_VSF(DUST_VS_SKD + 0u), DUST_VSF(DUST_VS_SKD + 1u), DUST_VSF(DUST_VS_SKD + 2u));
    vec3 ext = vec3(3.0 * sqrt(skd.x), 3.0 * sqrt(skd.y), 3.0 * sqrt(skd.z));
    float x0 = DUST_VWPF(k, DUST_VP_BBOX + 0u), x1 = DUST_VWPF(k, DUST_VP_BBOX + 1u);
    float y0 = DUST_VWPF(k, DUST_VP_BBOX + 2u), y1 = DUST_VWPF(k, DUST_VP_BBOX + 3u);
    if (!full) {
        vec3 meanK = DUST_VW_MEANK(k);
        vec3 r0 = DUST_V_R0, r1 = DUST_V_R1, r3 = DUST_V_R3, t = DUST_V_TV;
        float t0 = t.x, t1 = t.y, t3 = t.z;
        float W = DUST_V_W, H = DUST_V_H;
        DUST_VW_CORNER(-, -, +) DUST_VW_CORNER(+, -, +) DUST_VW_CORNER(-, +, +) DUST_VW_CORNER(+, +, +)
    }
    DUST_VWP(k, DUST_VP_BBOX + 0u) = floatBitsToUint(x0);
    DUST_VWP(k, DUST_VP_BBOX + 1u) = floatBitsToUint(x1);
    DUST_VWP(k, DUST_VP_BBOX + 2u) = floatBitsToUint(y0);
    DUST_VWP(k, DUST_VP_BBOX + 3u) = floatBitsToUint(y1);
    DUST_VWP(k, DUST_VP_FLAGS) = full ? 1u : 0u;
}
// A3: on screen, wholly behind the eye plane, the capsule's pieces
void dust_vw_a3(uint k) {
    bool full = (DUST_VWP(k, DUST_VP_FLAGS) & 1u) != 0u;
    float x0 = DUST_VWPF(k, DUST_VP_BBOX + 0u), x1 = DUST_VWPF(k, DUST_VP_BBOX + 1u);
    float y0 = DUST_VWPF(k, DUST_VP_BBOX + 2u), y1 = DUST_VWPF(k, DUST_VP_BBOX + 3u);
    bool on = full || !(x1 < 0.0 || y1 < 0.0 || x0 > DUST_V_W || y0 > DUST_V_H || !(x0 <= x1) || !(y0 <= y1));
    {
        DUST_VW_LK(Lk)
        vec3 r3 = DUST_V_R3;
        vec3 lt30 = dust_lt_mul(Lk, r3);
        if (dot(r3, DUST_VW_MEANK(k)) + DUST_V_TV.z + DUST_VOLUME_SIGMAS * sqrt(dot(lt30, lt30)) < 0.0) { on = false; full = false; }
    }
    uint ctl = dust_v_st[DUST_VS_CTL];
    uint nA = ctl & 31u, nB = (ctl >> 5u) & 31u;
    bool hasB = ((ctl >> 15u) & 1u) != 0u;
    uint i = k / nB, j = k % nB;
    bool cap = nA <= 4u || i < 2u || i + 2u >= nA || (hasB && (nB <= 4u || j < 2u || j + 2u >= nB));
    DUST_VWP(k, DUST_VP_FLAGS) = (full ? 1u : 0u) | (on ? 2u : 0u) | (cap ? 4u : 0u);
}
// A4: its own level (dust_volume_piece_at's K1c block) in two steps (row words 18, 19): A4a the
// screen sigma and the band axis, A4b the full piece's angular sigma and the levels
const uint DUST_VP_A4S = 18u;
const uint DUST_VP_A4L = 19u;
void dust_vw_a4(uint k) {
    uint fl = DUST_VWP(k, DUST_VP_FLAGS);
    bool full = (fl & 1u) != 0u;
    vec3 meanK = DUST_VW_MEANK(k);
    vec3 r3 = DUST_V_R3, t = DUST_V_TV;
    float t3 = t.z;
    DUST_VW_LK(Lk)
    float W = DUST_V_W, H = DUST_V_H;
    uint levels = dust_v_st[DUST_VS_LEVELS];
    float wM = dot(r3, meanK) + t3;
    float sigScr = 3.4e38;
    uint line = 0u;
    if (wM > 0.0) {
        vec2 sa = dust_volume_screen_at(Lk, meanK, wM, vec3(0.0), 0.0, DUST_V_R0, DUST_V_R1, r3, t, W, H);
        sigScr = sa.x;
        line = uint(sa.y);
    } else if (full) {
        vec3 lt3 = dust_lt_mul(Lk, r3);
        float sw2 = dot(lt3, lt3);
        float sw = sqrt(sw2);
        bool okA = sw > 0.0;
        float alpha = okA ? -wM / sw : 0.0;
        if (okA && alpha < 3.0) {
            float tail = 0.5 * (1.0 - dust_erf(alpha * DUST_FRAC_1_SQRT_2));
            float lam = exp(-0.5 * alpha * alpha) / DUST_SQRT_2PI / tail;
            float wF = wM + sw * lam;
            vec3 g = dust_l_mul(Lk, lt3);
            vec3 xF = meanK + g * ((wF - wM) / sw2);
            if (wF > 0.0) {
                vec2 sa = dust_volume_screen_at(Lk, xF, wF, g, 1.0 / sw2, DUST_V_R0, DUST_V_R1, r3, t, W, H);
                sigScr = sa.x;
                line = uint(sa.y);
            }
        }
    }
    DUST_VWP(k, DUST_VP_A4S) = floatBitsToUint(sigScr);
    DUST_VWP(k, DUST_VP_A4L) = line;
}
// A4b: the full piece's angular sigma, the levels
void dust_vw_a4b(uint k) {
    uint fl = DUST_VWP(k, DUST_VP_FLAGS);
    bool full = (fl & 1u) != 0u;
    float sigScr = DUST_VWPF(k, DUST_VP_A4S);
    uint line = DUST_VWP(k, DUST_VP_A4L);
    uint levels = dust_v_st[DUST_VS_LEVELS];
    float sigPx = sigScr;
    if (full) {
        DUST_VW_LK(Lk)
        vec3 meanK = DUST_VW_MEANK(k);
        vec3 skd = vec3(DUST_VSF(DUST_VS_SKD + 0u), DUST_VSF(DUST_VS_SKD + 1u), DUST_VSF(DUST_VS_SKD + 2u));
        float sigK = 2.0 * (Lk[0] * Lk[3] * Lk[5]) / (skd.x + skd.y + skd.z);
        vec3 dq = pc.eyeLocal.xyz - meanK;
        float denK = max(sqrt(dot(dq, dq)), sigK);
        sigPx = min((denK > 0.0 ? sigK / denK : 1.0) * DUST_VSF(DUST_VS_FPX), sigScr);
    }
    uint lp = dust_volume_level(min(sigPx, 1e30), levels);
    uint ls = line != 0u ? dust_volume_level(min(sigScr, 1e30), levels) : 0xFFFFFFFFu;
    DUST_VWP(k, DUST_VP_FLAGS) = (fl & 7u) | (line << 3u) | (ls != 0xFFFFFFFFu ? (1u << 5u) | ((ls & 0xFFu) << 16u) : 0u)
                               | ((lp & 0xFFu) << 8u);
}
// the piece as the class and range decisions read it (from its row; the factor and the mean unused)
DustVolumePiece dust_vw_piece(uint k) {
    DustVolumePiece q;
    q.l[0] = 0.0; q.l[1] = 0.0; q.l[2] = 0.0; q.l[3] = 0.0; q.l[4] = 0.0; q.l[5] = 0.0;
    q.qw = vec3(DUST_VWPF(k, DUST_VP_QW + 0u), DUST_VWPF(k, DUST_VP_QW + 1u), DUST_VWPF(k, DUST_VP_QW + 2u));
    q.k = 0.0;
    q.bbox = vec4(DUST_VWPF(k, DUST_VP_BBOX + 0u), DUST_VWPF(k, DUST_VP_BBOX + 1u), DUST_VWPF(k, DUST_VP_BBOX + 2u),
                  DUST_VWPF(k, DUST_VP_BBOX + 3u));
    uint fl = DUST_VWP(k, DUST_VP_FLAGS);
    q.full = (fl & 1u) != 0u;
    q.on = (fl & 2u) != 0u;
    q.cap = (fl & 4u) != 0u;
    q.line = (fl >> 3u) & 3u;
    q.ls = ((fl >> 5u) & 1u) != 0u ? (fl >> 16u) & 0xFFu : 0xFFFFFFFFu;
    q.lp = (fl >> 8u) & 0xFFu;
    return q;
}
// A5: the piece at its own level with the bias (DUST_VOLUME_EVALS_PP's body) in three steps
// through its row (word 16: level | class << 8 | order << 12 | bound << 16; 17: texels): A5a its
// level, texels and class; A5b its order bound; A5c the evaluations at the sub-ray limits 2, 3, 4
// into dust_t_row (k, 16 + k, 32 + k, 48 + k), on screen and at the last level into dust_v_unit[k]
// (bits 0, 1), its code into its row
const uint DUST_VP_A5 = 16u;
const uint DUST_VP_A5T = 17u;
void dust_vw_a5a(uint k, uint bias) {
    DustVolumePiece q = dust_vw_piece(k);
    uint levels = dust_v_st[DUST_VS_LEVELS];
    uint lev = dust_volume_piece_level(q, bias, levels);
    uint t = dust_volume_texels_at(q, lev);
    uvec2 ck = dust_volume_class(q, lev, levels);
    DUST_VWP(k, DUST_VP_A5) = lev | (ck.x << 8u) | (ck.y << 12u);
    DUST_VWP(k, DUST_VP_A5T) = t;
    dust_v_unit[k] = (q.on ? 1u : 0u) | (lev + 1u >= levels ? 2u : 0u);
}
void dust_vw_a5b(uint k) {
    DustVolumePiece q = dust_vw_piece(k);
    uint w = DUST_VWP(k, DUST_VP_A5);
    uint kb = dust_volume_order_bound(q, w & 0xFFu, (w >> 8u) & 15u, dust_v_st[DUST_VS_LEVELS]);
    DUST_VWP(k, DUST_VP_A5) = w | (kb << 16u);
}
void dust_vw_a5c(uint k) {
    DustVolumePiece q = dust_vw_piece(k);
    uint w = DUST_VWP(k, DUST_VP_A5);
    uint lev = w & 0xFFu, kb = w >> 16u;
    uvec2 ck = uvec2((w >> 8u) & 15u, (w >> 12u) & 15u);
    uint t = DUST_VWP(k, DUST_VP_A5T);
    uvec2 c2 = dust_volume_class_k(q, ck, 2u, kb), c3 = dust_volume_class_k(q, ck, 3u, kb),
          c4 = dust_volume_class_k(q, ck, 4u, kb);
    dust_t_row[k] = t;
    dust_t_row[16u + k] = t * dust_volume_texel_cost(c2.x, c2.y);
    dust_t_row[32u + k] = t * dust_volume_texel_cost(c3.x, c3.y);
    dust_t_row[48u + k] = t * dust_volume_texel_cost(c4.x, c4.y);
    DUST_VWP(k, DUST_VP_CODE) = dust_volume_class_code(q, ck, kb, lev);
}

// lane 0 after a pass A: the sums over the pieces (DUST_VOLUME_EVALS_PP's: integer sums, an OR and
// an AND, order-free), then dust_volume_draw's next step: the !anyOn return (after the first pass
// only), the cap loops (the bias loop: A5 again at bias + extra; the piece loop: fewer pieces, the
// covariance and A1-A5 again), or the end: K, the evaluations at K, the in-frame gate, the depth
void dust_vw_decide() {
    uint ctl = dust_v_st[DUST_VS_CTL];
    uint n = (ctl & 31u) * ((ctl >> 5u) & 31u);
    bool topKnown = ((ctl >> 19u) & 1u) != 0u;
    uint evals = 0u, e2 = 0u, e3 = 0u, e4 = 0u;
    bool anyOn = false, allTop = true;
    [[unroll]] for (uint k = 0u; k < DUST_VOLUME_PIECES_MAX; ++k) {
      if (k < n) {
        uint u = dust_v_unit[k];
        anyOn = anyOn || (u & 1u) != 0u;
        if (!topKnown) {
            allTop = allTop && (u & 2u) != 0u;
            evals += dust_t_row[k];
            e2 += dust_t_row[16u + k];
            e3 += dust_t_row[32u + k];
            e4 += dust_t_row[48u + k];
        }
      }
    }
#ifdef DUST_PROFILE
    DUST_VA(DUST_VA_ALLTOP) = allTop ? 1u : 0u;
#endif
    uint stage = dust_v_st[DUST_VS_STAGE];
    uint extra = stage & 0xFFu;
    stage >>= 8u;
    uint state = 5u;
    if (stage == 0u) {
        if (!anyOn) { dust_v_st[DUST_VS_STATE] = 3u; return; }
        stage = 1u;
    }
    if (stage == 1u) {
        if (!(evals <= DUST_VOLUME_MAX_EVALS || allTop)) { extra++; state = 2u; }
        else stage = 2u;
    }
    if (stage == 2u && evals > DUST_VOLUME_MAX_EVALS) {
        uint nA = ctl & 31u, nB = (ctl >> 5u) & 31u;
        if (nB > 1u) { nB = (nB + 1u) / 2u; ctl |= 1u << 18u; state = 4u; }
        else if (nA > 1u) { nA = (nA + 1u) / 2u; ctl |= 1u << 17u; state = 4u; }
        ctl = (ctl & ~1023u) | nA | (nB << 5u);
    }
    dust_v_st[DUST_VS_STAGE] = extra | (stage << 8u);
    if (state == 5u) {
        // the packet's sub-ray limit K and the evaluations at K (dust::volume_subray_limit)
        uint K = 4u;
        if (e4 > DUST_VOLUME_MAX_EVALS) K = 3u;
        if (K == 3u && e3 > DUST_VOLUME_MAX_EVALS) K = 2u;
        if (K == 2u && e2 > DUST_VOLUME_MAX_EVALS) K = 1u;
        evals = K == 4u ? e4 : (K == 3u ? e3 : (K == 2u ? e2 : evals));
        // the in-frame gate (the top-level path: none)
        bool gated = false;
        if (!topKnown) {
            uint before = atomicAdd(DUST_PW(DUST_PYR_VOLUME_EVALS), evals);
            gated = dust_frame_gates_on() && before > DUST_FRAME_HARD_FACTOR * DUST_VOLUME_FRAME_BUDGET;
        }
        vec3 eye = pc.eyeLocal.xyz, mean = vec3(DUST_VRF(0u), DUST_VRF(1u), DUST_VRF(2u));
        dust_v_st[DUST_VS_DEPTH] = floatBitsToUint(max(sqrt(dot(eye - mean, eye - mean)) / DUST_AU_M, 0.0));
        dust_v_st[DUST_VS_EVALS] = evals;
        ctl = (ctl & ~((7u << 10u) | (1u << 16u))) | (K << 10u) | ((gated ? 1u : 0u) << 16u);
    }
    dust_v_st[DUST_VS_CTL] = ctl;
    dust_v_st[DUST_VS_STATE] = state;
}

// ---- pass B: a lane per piece ----------------------------------------------------------------
// The piece's drawn level, class and order (dust_volume_draw's pass B head: the top level's class,
// the gated level's, or pass A's code), its range at that level and its class under K into its row
// (DUST_VP_DESC / _RX / _RY: drawn = it has texels to draw here); the top-level path's evaluations
// into dust_t_row[k]; a code past the pyramid into dust_v_unit[k] (lane 0 counts it, in piece
// order). A small piece's points then (S1-S3): an edge piece goes to the edge pass.
void dust_vw_pass_b(uint k) {
    uint ctl = dust_v_st[DUST_VS_CTL];
    uint levels = dust_v_st[DUST_VS_LEVELS];
    bool topk = ((ctl >> 19u) & 1u) != 0u, gatedk = ((ctl >> 16u) & 1u) != 0u;
    uint code = DUST_VWP(k, DUST_VP_CODE);
    uint lev = code >> 8u;
    uvec2 ck;
    uint kb;
    uint gate = dust_gate_level(levels);
    uint bad = 0u;
    dust_t_row[k] = 0u;
    if (topk || (gatedk && lev < gate)) {
        DustVolumePiece q = dust_vw_piece(k);
        lev = topk ? levels - 1u : gate;
        ck = dust_volume_class(q, lev, levels);
        kb = 0u;  // (dust_vw_pass_b1: the order bound)
    } else {
        if (lev >= levels) bad = 1u | ((code & 0xFFFFu) << 1u);
        lev = min(lev, levels - 1u);
        ck = uvec2(code & 3u, ((code >> 2u) & 3u) + 1u);
        kb = ((code >> 4u) & 3u) + 1u;
    }
    dust_v_unit[k] = bad;
    DUST_VWP(k, DUST_VP_BLEV) = lev;
    DUST_VWP(k, DUST_VP_BLEV + 1u) = ck.x;
    DUST_VWP(k, DUST_VP_BLEV + 2u) = ck.y;
    DUST_VWP(k, DUST_VP_BLEV + 3u) = kb;
}
// pass B's order bound at the top / gated level (dust_vw_pass_b's kb = 0)
void dust_vw_pass_b1(uint k) {
    if (DUST_VWP(k, DUST_VP_BLEV + 3u) != 0u) return;
    DustVolumePiece q = dust_vw_piece(k);
    DUST_VWP(k, DUST_VP_BLEV + 3u) = dust_volume_order_bound(q, DUST_VWP(k, DUST_VP_BLEV), DUST_VWP(k, DUST_VP_BLEV + 1u),
                                                             dust_v_st[DUST_VS_LEVELS]);
}
// pass B's second step: the range at the level, the class's order, the descriptor
void dust_vw_pass_b2(uint k) {
    uint ctl = dust_v_st[DUST_VS_CTL];
    uint levels = dust_v_st[DUST_VS_LEVELS];
    bool topk = ((ctl >> 19u) & 1u) != 0u;
    DustVolumePiece q = dust_vw_piece(k);
    uint lev = DUST_VWP(k, DUST_VP_BLEV);
    uvec2 ck = uvec2(DUST_VWP(k, DUST_VP_BLEV + 1u), DUST_VWP(k, DUST_VP_BLEV + 2u));
    uint kb = DUST_VWP(k, DUST_VP_BLEV + 3u);
    uint lw = DUST_PW(DUST_PYR_TABLE + 3u * lev + 1u);
    uint lh = DUST_PW(DUST_PYR_TABLE + 3u * lev + 2u);
    uint drawn = 0u;
    ivec4 r = ivec4(0, -1, 0, -1);
    uint cls = 0u, kSub = 1u;
    if (lw == 0u || lh == 0u) {
        atomicAdd(DUST_PW(DUST_PYR_BAD_CODE), 0x10000u);
    } else {
        r = dust_volume_range(q, lev, lw, lh);
        if (!(r.y < r.x || r.w < r.z)) {
            uvec2 cks = dust_volume_class_k(q, ck, (ctl >> 10u) & 7u, kb);
            if (topk) {
                uint tk = uint((r.y - r.x + 1) * (r.w - r.z + 1));
                dust_t_row[k] = tk * dust_volume_texel_cost(cks.x, cks.y);
            }
            cls = cks.x;
            kSub = max(cks.y, 1u);
            drawn = 1u;
        }
    }
    DUST_VWP(k, DUST_VP_DESC) = lev | (cls << 8u) | (kSub << 10u) | (drawn << 13u);
    DUST_VWP(k, DUST_VP_RX) = uint(r.x) | (uint(r.y) << 16u);
    DUST_VWP(k, DUST_VP_RY) = uint(r.z) | (uint(r.w) << 16u);
}
// ---- a small piece's points (dust_volume_small_points in three steps through its row) -------------
#define DUST_VW_SIN(k, w) DUST_VWPF(k, DUST_VP_SMALL + (w))
// S1, the prelude: the depth axis, the lateral axes' dots, the mass (false: no axis, no points)
bool dust_vw_s1(uint k) {
    DUST_VW_LK(L)
    vec3 qw = vec3(DUST_VWPF(k, DUST_VP_QW + 0u), DUST_VWPF(k, DUST_VP_QW + 1u), DUST_VWPF(k, DUST_VP_QW + 2u));
    vec3 lq = dust_l_mul(L, qw);
    vec3 mean = pc.eyeLocal.xyz - lq;
    float dn = sqrt(dot(lq, lq));
    if (!(dn > 0.0)) return false;
    vec3 e = lq * (-1.0 / dn);
    vec3 lte = dust_lt_mul(L, e);
    float se = sqrt(dot(lte, lte));
    if (!(se > 0.0)) return false;
    vec3 z = lte * (1.0 / se);
    vec3 g = dust_l_mul(L, z);
    float sgn = z.z >= 0.0 ? 1.0 : -1.0;
    float ia = -1.0 / (sgn + z.z);
    float ib = z.x * z.y * ia;
    float sb = sqrt(DUST_BOUND_VAR);
    g = g * sb;
    vec3 lat1 = dust_l_mul(L, vec3(1.0 + sgn * z.x * z.x * ia, sgn * ib, -sgn * z.x)) * sb;
    vec3 lat2 = dust_l_mul(L, vec3(ib, sgn + z.y * z.y * ia, -z.y)) * sb;
    vec3 r0 = DUST_V_R0, r1 = DUST_V_R1, r3 = DUST_V_R3;
    DUST_VWP(k, DUST_VP_SMALL + 6u) = floatBitsToUint(dot(r0, lat1));
    DUST_VWP(k, DUST_VP_SMALL + 7u) = floatBitsToUint(dot(r0, lat2));
    DUST_VWP(k, DUST_VP_SMALL + 8u) = floatBitsToUint(dot(r1, lat1));
    DUST_VWP(k, DUST_VP_SMALL + 9u) = floatBitsToUint(dot(r1, lat2));
    DUST_VWP(k, DUST_VP_SMALL + 10u) = floatBitsToUint(dot(r3, lat1));
    DUST_VWP(k, DUST_VP_SMALL + 11u) = floatBitsToUint(dot(r3, lat2));
    vec4 sp = vec4(DUST_VRF(23u), DUST_VRF(24u), DUST_VRF(25u), DUST_VRF(26u));
    float kept = 1.0;
    if (dust_has_support(sp)) {
        vec3 ltn = dust_lt_mul(L, sp.xyz);
        float sn = sqrt(dot(ltn, ltn));
        float mn = dot(sp.xyz, mean) - sp.w;
        kept = sn > 0.0 ? 0.5 * (1.0 + dust_erf(mn / (sn * DUST_SQRT_2))) : (mn >= 0.0 ? 1.0 : 0.0);
    }
    float mass = 2.0 * DUST_PI_F32 * DUST_VWPF(k, DUST_VP_K) * (L[0] * L[3] * L[5]) * DUST_VSF(DUST_VS_RENORM) * kept;
    DUST_VWP(k, DUST_VP_SMALL + 0u) = floatBitsToUint(mean.x);
    DUST_VWP(k, DUST_VP_SMALL + 1u) = floatBitsToUint(mean.y);
    DUST_VWP(k, DUST_VP_SMALL + 2u) = floatBitsToUint(mean.z);
    DUST_VWP(k, DUST_VP_SMALL + 3u) = floatBitsToUint(g.x);
    DUST_VWP(k, DUST_VP_SMALL + 4u) = floatBitsToUint(g.y);
    DUST_VWP(k, DUST_VP_SMALL + 5u) = floatBitsToUint(g.z);
    DUST_VWP(k, DUST_VP_SMALL + 12u) = floatBitsToUint(mass);
    return true;
}
#define DUST_VW_SMEAN(k) vec3(DUST_VW_SIN(k, 0u), DUST_VW_SIN(k, 1u), DUST_VW_SIN(k, 2u))
#define DUST_VW_SG(k) vec3(DUST_VW_SIN(k, 3u), DUST_VW_SIN(k, 4u), DUST_VW_SIN(k, 5u))
// S2: the slices' radiance weights (dust_small_weights)
void dust_vw_s2(uint k) {
    float wr0, wr1, wr2;
    vec3 sun0;
    float r20;
    vec3 mean = DUST_VW_SMEAN(k), g = DUST_VW_SG(k);
    dust_vw_small_weights(mean + g * dust_small_node(0u, false).x, mean + g * dust_small_node(1u, false).x,
                       mean + g * dust_small_node(2u, false).x, pc.eyeLocal.xyz, vec3(DUST_VRF(0u), DUST_VRF(1u), DUST_VRF(2u)),
                          wr0, wr1, wr2, sun0, r20);
    DUST_VWP(k, DUST_VP_SMALL + 13u) = floatBitsToUint(wr0);
    DUST_VWP(k, DUST_VP_SMALL + 14u) = floatBitsToUint(wr1);
    DUST_VWP(k, DUST_VP_SMALL + 15u) = floatBitsToUint(wr2);
}
// slice q's point by one projection (dust_volume_small_fast on the row's inputs)
DustSmallPoint dust_vw_small_point(uint k, uint q) {
    return dust_volume_small_fast(q, false, DUST_VW_SMEAN(k), DUST_VW_SG(k), DUST_VW_SIN(k, 6u), DUST_VW_SIN(k, 7u),
                                  DUST_VW_SIN(k, 8u), DUST_VW_SIN(k, 9u), DUST_VW_SIN(k, 10u), DUST_VW_SIN(k, 11u),
                                  DUST_VW_SIN(k, 12u), DUST_VW_SIN(k, 13u + q), DUST_V_R0, DUST_V_R1, DUST_V_R3,
                                  DUST_V_TV, DUST_V_W, DUST_V_H);
}
// dust_volume_small_range's bounds, one point at a time
void dust_vw_cover(DustSmallPoint p, inout vec4 b) {
    if (!(p.m > 0.0)) return;
    float ex = DUST_SMALL_COVER / (p.ix * DUST_SQRT_2), ey = DUST_SMALL_COVER / (p.iy * DUST_SQRT_2);
    b.x = min(b.x, p.mx - ex);
    b.y = max(b.y, p.mx + ex);
    b.z = min(b.z, p.my - ey);
    b.w = max(b.w, p.my + ey);
}
// S1-S3 of the small pieces in four steps (dust_vw_small_a .. _d): the prelude (no axis: not
// drawn), the weights, the slices' points (a lane a point, into the stash's words: piece k's
// point q at dust_v_stash[5 (3 k + q)], mx my ix iy m), then a lane a piece: the edge decision
// on slice 0's point (an edge piece marked for dust_vw_small_e, a fifth step), else the texels
// its points cover as its range
#define DUST_VW_SMALL_ON(k) ((DUST_VWP(k, DUST_VP_DESC) & ((1u << 13u) | (3u << 8u))) == ((1u << 13u) | (DUST_VOLUME_CLASS_SMALL << 8u)))
#ifdef DUST_VW_STAGE
#define DUST_VW_SPT(k, q, w) uintBitsToFloat(pc.pyramid.v[DUST_VW_ROW + DUST_VW_SX_PT + 6u * (3u * (k) + (q)) + (w)])
#else
#define DUST_VW_SPT(k, q, w) uintBitsToFloat(dust_v_stash[5u * (3u * (k) + (q)) + (w)])
#endif
void dust_vw_small_a(uint k) {
    if (!DUST_VW_SMALL_ON(k)) return;
    if (!dust_vw_s1(k)) DUST_VWP(k, DUST_VP_DESC) = DUST_VWP(k, DUST_VP_DESC) & ~(1u << 13u);
}
void dust_vw_small_b(uint k) {
    if (DUST_VW_SMALL_ON(k)) dust_vw_s2(k);
}
void dust_vw_small_c(uint l) {
    uint k = l / 3u, q = l % 3u;
    if (!DUST_VW_SMALL_ON(k)) return;
    DustSmallPoint p = dust_vw_small_point(k, q);
#ifdef DUST_VW_STAGE
    // into the scratch row (PBSd reads them there, the texel loop's CTX copies them)
    uint o = DUST_VW_ROW + DUST_VW_SX_PT + 6u * l;
    pc.pyramid.v[o + 0u] = floatBitsToUint(p.mx);
    pc.pyramid.v[o + 1u] = floatBitsToUint(p.my);
    pc.pyramid.v[o + 2u] = floatBitsToUint(p.ix);
    pc.pyramid.v[o + 3u] = floatBitsToUint(p.iy);
    pc.pyramid.v[o + 4u] = floatBitsToUint(p.m);
    pc.pyramid.v[o + 5u] = floatBitsToUint(p.rho);
#else
    uint o = 5u * l;
    dust_v_stash[o + 0u] = floatBitsToUint(p.mx);
    dust_v_stash[o + 1u] = floatBitsToUint(p.my);
    dust_v_stash[o + 2u] = floatBitsToUint(p.ix);
    dust_v_stash[o + 3u] = floatBitsToUint(p.iy);
    dust_v_stash[o + 4u] = floatBitsToUint(p.m);
#endif
}
// dust_vw_cover on point q's stash words
void dust_vw_cover_q(uint k, uint q, inout vec4 b) {
    DustSmallPoint p;
    p.mx = DUST_VW_SPT(k, q, 0u); p.my = DUST_VW_SPT(k, q, 1u);
    p.ix = DUST_VW_SPT(k, q, 2u); p.iy = DUST_VW_SPT(k, q, 3u);
    p.rho = 0.0; p.m = DUST_VW_SPT(k, q, 4u);
    dust_vw_cover(p, b);
}
void dust_vw_small_d(uint k) {
    if (!DUST_VW_SMALL_ON(k)) return;
    uint desc = DUST_VWP(k, DUST_VP_DESC);
    uint lev = desc & 0xFFu;
    uint rx = DUST_VWP(k, DUST_VP_RX), ry = DUST_VWP(k, DUST_VP_RY);
    ivec4 r = ivec4(int(rx & 0xFFFFu), int(rx >> 16u), int(ry & 0xFFFFu), int(ry >> 16u));
    uint drawn = 0u;
    bool edge = true;
    if (DUST_VW_SPT(k, 0u, 4u) > 0.0) {
        float ex = 3.0 / (DUST_VW_SPT(k, 0u, 2u) * DUST_SQRT_2), ey = 3.0 / (DUST_VW_SPT(k, 0u, 3u) * DUST_SQRT_2);
        float W = DUST_V_W, H = DUST_V_H;
        float mx = DUST_VW_SPT(k, 0u, 0u), my = DUST_VW_SPT(k, 0u, 1u);
        edge = mx - ex < 0.0 || mx + ex > W || my - ey < 0.0 || my + ey > H;
    }
    if (edge) {
        // to dust_vw_small_e (drawn here: no; edge pending: bit 14)
        DUST_VWP(k, DUST_VP_DESC) = (desc & ~(1u << 13u)) | (1u << 14u);
        return;
    } else {
        DUST_PROF_COUNT(small_fast, 1u)
        vec4 b = vec4(3.4e38, -3.4e38, 3.4e38, -3.4e38);
        dust_vw_cover_q(k, 0u, b);
        dust_vw_cover_q(k, 1u, b);
        dust_vw_cover_q(k, 2u, b);
        if (b.x <= b.y && b.z <= b.w) {
            float sz = float(1u << lev);
            ivec4 rs = ivec4(int(max(floor(b.x / sz), float(r.x))), int(min(floor(b.y / sz), float(r.y))),
                             int(max(floor(b.z / sz), float(r.z))), int(min(floor(b.w / sz), float(r.w))));
            if (!(rs.y < rs.x || rs.w < rs.z)) {
                drawn = 1u;
                DUST_VWP(k, DUST_VP_RX) = uint(rs.x) | (uint(rs.y) << 16u);
                DUST_VWP(k, DUST_VP_RY) = uint(rs.z) | (uint(rs.w) << 16u);
            }
        }
    }
    if (drawn == 0u) DUST_VWP(k, DUST_VP_DESC) = desc & ~(1u << 13u);
}
// an edge piece (dust_vw_small_d's bit 14) to the edge pass, from its row's inputs
void dust_vw_small_e(uint k) {
    uint desc = DUST_VWP(k, DUST_VP_DESC);
    if ((desc & ((1u << 14u) | (3u << 8u))) != ((1u << 14u) | (DUST_VOLUME_CLASS_SMALL << 8u))) return;
    DUST_VWP(k, DUST_VP_DESC) = desc & ~(1u << 14u);
    // dust_small_edge_put's append and words, a word at a time (the row's inputs are its words 0-15;
    // its range words are the row's: r's coordinates are their 16-bit fields)
    uint ebase = dust_vw_vedge_base((pc.flags & DUST_VIEW_EDGE_ODD) != 0u);
    uint i = atomicAdd(pc.pyramid.v[ebase + 3u], 1u);
    if (i >= dust_vw_vqueue_capacity(ebase)) return;
    atomicMax(pc.pyramid.v[ebase], (i >> DUST_VEDGE_GROUP_SHIFT) + 1u);
    uint a = ebase + DUST_VQUEUE_HEADER_WORDS + i * DUST_VEDGE_RECORD_WORDS;
    [[unroll]] for (uint w = 0u; w < 16u; ++w) pc.pyramid.v[a + w] = DUST_VWP(k, DUST_VP_SMALL + w);
    pc.pyramid.v[a + 16u] = DUST_VWP(k, DUST_VP_RX);
    pc.pyramid.v[a + 17u] = DUST_VWP(k, DUST_VP_RY);
    pc.pyramid.v[a + 18u] = desc & 0xFFu;
    pc.pyramid.v[a + 19u] = DUST_VRW(20u);
    pc.pyramid.v[a + 20u] = DUST_VRW(21u);
    pc.pyramid.v[a + 21u] = DUST_VRW(22u);
    pc.pyramid.v[a + 22u] = dust_v_st[DUST_VS_DEPTH];
    pc.pyramid.v[a + 23u] = dust_hash32(DUST_VRW(18u) ^ k);
}

// ---- the texel phase ---------------------------------------------------------------------------
// The group's piece's words in dust_v_gx: 0-2 qw, 3 k, 4 flags, 5 code, 6 desc, 7-8 range; a
// small piece's points at DUST_VG_PT + 6 q (mx my ix iy rho m)
#define DUST_VG_DESC dust_v_gx[DUST_VP_DESC]
#define DUST_VG_LEV (DUST_VG_DESC & 0xFFu)
#define DUST_VG_CLS ((DUST_VG_DESC >> 8u) & 3u)
#define DUST_VG_KSUB ((DUST_VG_DESC >> 10u) & 7u)

// the optical depth per px^2 of the group's piece along the ray on the planes aRow, bRow
// (dust_volume_ray_tau's arithmetic; the capsule's interval first, every word read at its use)
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
float dust_vw_ray(vec3 aRow, vec3 bRow, vec3 r3) {
    precise vec3 v = cross(aRow, bRow);
    if (dot(v, r3) < 0.0) v = -v;
    float vn = sqrt(dot(v, v));
    if (!(vn > 0.0)) return -1.0;
    vec3 u = v * (1.0 / vn);
    vec2 cap = vec2(-3.4e38, 3.4e38);
    if ((dust_v_gx[DUST_VP_FLAGS] & 4u) != 0u) {
        DustVolumeBound bd;
        bd.l[0] = DUST_VSF(DUST_VS_BOUND + 0u); bd.l[1] = DUST_VSF(DUST_VS_BOUND + 1u); bd.l[2] = DUST_VSF(DUST_VS_BOUND + 2u);
        bd.l[3] = DUST_VSF(DUST_VS_BOUND + 3u); bd.l[4] = DUST_VSF(DUST_VS_BOUND + 4u); bd.l[5] = DUST_VSF(DUST_VS_BOUND + 5u);
        bd.qe = vec3(DUST_VSF(DUST_VS_BOUND + 6u), DUST_VSF(DUST_VS_BOUND + 7u), DUST_VSF(DUST_VS_BOUND + 8u));
        bd.g = vec3(DUST_VSF(DUST_VS_BOUND + 9u), DUST_VSF(DUST_VS_BOUND + 10u), DUST_VSF(DUST_VS_BOUND + 11u));
        cap = dust_capsule_interval(bd, u);
    }
    if (!(cap.y > max(cap.x, 0.0))) return 0.0;
    DUST_VW_LK(L)
    vec3 uw = dust_cholesky_solve3(L, u);
    float a = dot(uw, uw);
    vec3 qw = vec3(DUST_VGF(0u), DUST_VGF(1u), DUST_VGF(2u));
    float b = dot(uw, qw);
    precise vec3 rr = qw - uw * (b / a);
    float rr2 = dot(rr, rr);
    float sa = sqrt(a);
    vec3 eye = pc.eyeLocal.xyz;
    float tauPx = DUST_VGF(3u) * DUST_VSF(DUST_VS_RENORM) / sa * exp(-0.5 * rr2)
                * dust_ray_kept(a, b, rr2, cap.x, cap.y, eye, u, vec4(DUST_VRF(23u), DUST_VRF(24u), DUST_VRF(25u), DUST_VRF(26u)));
    if (pc.albedo > 0.0) {
        vec3 x = eye + u * max(-b / a, 0.0);
        tauPx *= dust_vw_volume_weight(vec3(DUST_VRF(0u), DUST_VRF(1u), DUST_VRF(2u)), x, u);
    }
    return tauPx;
}
#endif
// dust_volume_line_tau on the group's piece
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
float dust_vw_line_tau(bool alongX, float fixed_, float t) {
    vec3 r0 = DUST_V_R0, r1 = DUST_V_R1, r3 = DUST_V_R3;
    float W = DUST_V_W, H = DUST_V_H;
    vec3 aR = alongX ? r0 - r3 * (2.0 * t / W - 1.0) : r0 - r3 * (2.0 * fixed_ / W - 1.0);
    vec3 bR = alongX ? r1 - r3 * (2.0 * fixed_ / H - 1.0) : r1 - r3 * (2.0 * t / H - 1.0);
    float v = dust_vw_ray(aR, bR, r3);
    return (v >= 0.0 && !isinf(v)) ? v : 0.0;
}
#endif
// a texel's in-frame rectangle (dust_volume_segment_small / _sub): xa_, xb_, ya, yb
vec4 dust_vw_rect(int ty, int tx, float sz) {
    return vec4(max(float(tx) * sz, 0.0), min(float(tx + 1) * sz, DUST_V_W),
                max(float(ty) * sz, 0.0), min(float(ty + 1) * sz, DUST_V_H));
}
#define DUST_VW_GPT(q) DustSmallPoint(DUST_VGF(DUST_VG_PT + 6u * (q)), DUST_VGF(DUST_VG_PT + 6u * (q) + 1u), \
    DUST_VGF(DUST_VG_PT + 6u * (q) + 2u), DUST_VGF(DUST_VG_PT + 6u * (q) + 3u), DUST_VGF(DUST_VG_PT + 6u * (q) + 4u), \
    DUST_VGF(DUST_VG_PT + 6u * (q) + 5u))
// a unit of a one-unit class: the texel's tau (the centre class's ray; the small class's five
// rectangle integrals, NaN for a texel without in-frame area: the segment loop's `continue`)
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
float dust_vw_unit1(int ty, int tx) {
    float sz = float(1u << DUST_VG_LEV);
    if (DUST_VG_CLS == DUST_VOLUME_CLASS_CENTRE) {
        vec3 r3 = DUST_V_R3;
        float yc = (float(ty) + 0.5) * sz;
        float ny = 2.0 * yc / DUST_V_H - 1.0;
        vec3 bRow = DUST_V_R1 - r3 * ny;
        float xc = (float(tx) + 0.5) * sz;
        float nx = 2.0 * xc / DUST_V_W - 1.0;
        vec3 aRow = DUST_V_R0 - r3 * nx;
        return dust_vw_ray(aRow, bRow, r3);
    }
    vec4 rc = dust_vw_rect(ty, tx, sz);
    float xa_ = rc.x, xb_ = rc.y, ya = rc.z, yb = rc.w;
    if (!(xb_ > xa_ && yb > ya)) return uintBitsToFloat(0x7FC00000u);
    DustSmallPoint s3 = dust_small_none(), s4 = dust_small_none();
    float acc = dust_small_rect(DUST_VW_GPT(0u), xa_, xb_, ya, yb) + dust_small_rect(DUST_VW_GPT(1u), xa_, xb_, ya, yb)
              + dust_small_rect(DUST_VW_GPT(2u), xa_, xb_, ya, yb) + dust_small_rect(s3, xa_, xb_, ya, yb)
              + dust_small_rect(s4, xa_, xb_, ya, yb);
    return max(acc, 0.0) / ((xb_ - xa_) * (yb - ya));
}
#endif
// the line units' stash (a lane's four words in dust_v_stash, across the barrier between the
// two steps of a unit: the line's closed form, then the one ray): the ray's line coordinate, the
// mode (0 the Gauss-Legendre fallback's node, 1 the unit's value is word 2, 2 the closed form),
// sg, the erf difference
#define DUST_VW_ST(w) dust_v_stash[4u * gl_LocalInvocationIndex + (w)]
const uint DUST_VW_SENTINEL = 0xFFFFFFFFu;
// step 1 of a line unit (sub = 4 j + i): line j's dust_volume_line_integral up to its ray: the
// closed form's parameters on node i = 0 (node 1 the sentinel, the others 0), or the fallback's
// node i, into the stash
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
void dust_vw_unit_line(int ty, int tx, uint sub) {
    float sz = float(1u << DUST_VG_LEV);
    vec4 rc = dust_vw_rect(ty, tx, sz);
    float xa_ = rc.x, xb_ = rc.y, ya = rc.z, yb = rc.w;
    DUST_VW_ST(1u) = 1u;
    DUST_VW_ST(2u) = 0u;
    if (!(xb_ > xa_ && yb > ya)) return;
    uint kSub = DUST_VG_KSUB;
    bool alongX = ((dust_v_gx[DUST_VP_FLAGS] >> 3u) & 3u) == 1u;
    uint i = sub & 3u;
    float c = alongX ? 0.5 * (ya + yb) : 0.5 * (xa_ + xb_);
    float h = alongX ? 0.5 * (yb - ya) : 0.5 * (xb_ - xa_);
    float lo = alongX ? xa_ : ya, hi = alongX ? xb_ : yb;
    vec2 nwl = dust_gl_node(kSub, sub >> 2u);
    float fixed_ = c + h * nwl.x;
    vec3 r0 = DUST_V_R0, r1 = DUST_V_R1, r3 = DUST_V_R3;
    float W = DUST_V_W, H = DUST_V_H;
    float mid = 0.5 * (lo + hi);
    vec3 aM = alongX ? r0 - r3 * (2.0 * mid / W - 1.0) : r0 - r3 * (2.0 * fixed_ / W - 1.0);
    vec3 bM = alongX ? r1 - r3 * (2.0 * fixed_ / H - 1.0) : r1 - r3 * (2.0 * mid / H - 1.0);
    vec3 vm = cross(aM, bM);
    vec3 v1 = alongX ? cross(r3, bM) * (-2.0 / W) : cross(aM, r3) * (-2.0 / H);
    if (dot(vm, r3) < 0.0) { vm = -vm; v1 = -v1; }
    DUST_VW_LK(L)
    vec3 wm = dust_cholesky_solve3(L, vm);
    vec3 w1 = dust_cholesky_solve3(L, v1);
    float a0 = dot(wm, wm), a1 = dot(wm, w1), a2 = dot(w1, w1);
    vec3 qw = vec3(DUST_VGF(0u), DUST_VGF(1u), DUST_VGF(2u));
    float b0 = dot(wm, qw), b1 = dot(w1, qw);
    float den = b1 * a1 - b0 * a2;
    bool ok = den != 0.0;
    float ts = 0.0, d2 = 0.0;
    if (ok) {
        ts = (b0 * a1 - b1 * a0) / den;
        float at = a0 + 2.0 * a1 * ts + a2 * ts * ts;
        float bt = b0 + b1 * ts;
        d2 = 2.0 * (a2 * bt * bt - b1 * b1 * at) / (at * at);
        ok = d2 > 0.0 && !isinf(ts) && !isnan(ts);
    }
    if (!ok) {
        float hh = 0.5 * (hi - lo);
        vec2 nw = dust_gl_node(DUST_VOLUME_SUBRAY_ORDER_MAX, i);
        DUST_VW_ST(0u) = floatBitsToUint(mid + hh * nw.x);
        DUST_VW_ST(1u) = 0u;
        return;
    }
    if (i == 1u) { DUST_VW_ST(2u) = DUST_VW_SENTINEL; return; }
    if (i != 0u) return;
    float sg = sqrt(2.0 / d2);
    float t = mid + ts;
    float atS = a0 + 2.0 * a1 * ts + a2 * ts * ts;
    float btS = b0 + b1 * ts;
    float r2 = dot(qw, qw) - btS * btS / atS;
    if (!(r2 < DUST_BOUND_R2)) { DUST_VW_ST(2u) = floatBitsToUint(0.0); return; }
    float half_ = sg * sqrt(DUST_BOUND_R2 - r2);
    lo = max(lo, t - half_);
    hi = min(hi, t + half_);
    if (!(hi > lo)) { DUST_VW_ST(2u) = floatBitsToUint(0.0); return; }
    float k = 1.0 / (sg * DUST_SQRT_2);
    DUST_VW_ST(0u) = floatBitsToUint(t);
    DUST_VW_ST(1u) = 2u;
    DUST_VW_ST(2u) = floatBitsToUint(sg);
    DUST_VW_ST(3u) = floatBitsToUint(dust_erf((hi - t) * k) - dust_erf((lo - t) * k));
}
#endif
// step 2 of a unit: the sub-ray class's (sub = j kSub + i) sub-ray (i, j) tau (dust_volume_segment_sub's
// inner loop body before its accumulation); the line class's one ray from the stash and the
// closed form's factors (dust_volume_line_integral's return), or the stashed value
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
uint dust_vw_unit(int ty, int tx, uint sub) {
    float sz = float(1u << DUST_VG_LEV);
    vec4 rc = dust_vw_rect(ty, tx, sz);
    float xa_ = rc.x, xb_ = rc.y, ya = rc.z, yb = rc.w;
    if (!(xb_ > xa_ && yb > ya)) return 0u;
    uint kSub = DUST_VG_KSUB;
    vec3 r3 = DUST_V_R3;
    vec3 aR, bR;
    bool line = DUST_VG_CLS == DUST_VOLUME_CLASS_LINE;
    if (!line) {
        float cx = 0.5 * (xa_ + xb_), cy = 0.5 * (ya + yb);
        float hx = 0.5 * (xb_ - xa_), hy = 0.5 * (yb - ya);
        vec2 nwj = dust_gl_node(kSub, sub / kSub);
        float ys = cy + hy * nwj.x;
        bR = DUST_V_R1 - r3 * (2.0 * ys / DUST_V_H - 1.0);
        vec2 nwi = dust_gl_node(kSub, sub % kSub);
        float xs = cx + hx * nwi.x;
        aR = DUST_V_R0 - r3 * (2.0 * xs / DUST_V_W - 1.0);
    } else {
        if (DUST_VW_ST(1u) == 1u) return DUST_VW_ST(2u);
        // dust_volume_line_tau(alongX, fixed_, t)
        bool alongX = ((dust_v_gx[DUST_VP_FLAGS] >> 3u) & 3u) == 1u;
        float c = alongX ? 0.5 * (ya + yb) : 0.5 * (xa_ + xb_);
        float h = alongX ? 0.5 * (yb - ya) : 0.5 * (xb_ - xa_);
        float fixed_ = c + h * dust_gl_node(kSub, sub >> 2u).x;
        float t = uintBitsToFloat(DUST_VW_ST(0u));
        float W = DUST_V_W, H = DUST_V_H;
        aR = alongX ? DUST_V_R0 - r3 * (2.0 * t / W - 1.0) : DUST_V_R0 - r3 * (2.0 * fixed_ / W - 1.0);
        bR = alongX ? DUST_V_R1 - r3 * (2.0 * fixed_ / H - 1.0) : DUST_V_R1 - r3 * (2.0 * t / H - 1.0);
    }
    float v = dust_vw_ray(aR, bR, r3);
    if (!line) return floatBitsToUint(v);
    v = (v >= 0.0 && !isinf(v)) ? v : 0.0;
    if (DUST_VW_ST(1u) == 0u) return floatBitsToUint(v);
    return floatBitsToUint(v * uintBitsToFloat(DUST_VW_ST(2u)) * DUST_SQRT_2PI * 0.5 * uintBitsToFloat(DUST_VW_ST(3u)));
}
#endif
// the reduce's loop bodies written out: a sub-ray unit (j, i), a sub-ray row j, a line node j
// (its four-node sum i = 0 .. 3 unless the sentinel)
#define DUST_VW_RI(j, i) if ((i) < kSub) { \
    vec2 nwi = dust_gl_node(kSub, i); \
    float tt = uintBitsToFloat(dust_v_unit[u0 + (j) * kSub + (i)]); \
    if (tt >= 0.0 && !isinf(tt)) acc += (nwi.y * nwj.y) * tt; }
#define DUST_VW_RJ(j) if ((j) < kSub) { \
    vec2 nwj = dust_gl_node(kSub, j); \
    DUST_VW_RI(j, 0u) DUST_VW_RI(j, 1u) DUST_VW_RI(j, 2u) DUST_VW_RI(j, 3u) }
#define DUST_VW_R4(i) acc4 += dust_gl_node(DUST_VOLUME_SUBRAY_ORDER_MAX, i).y * uintBitsToFloat(dust_v_unit[b + (i)]);
#define DUST_VW_RL(j) if ((j) < kSub) { \
    vec2 nw = dust_gl_node(kSub, j); \
    uint b = u0 + 4u * (j); \
    float li; \
    if (dust_v_unit[b + 1u] == DUST_VW_SENTINEL) { \
        li = uintBitsToFloat(dust_v_unit[b]); \
    } else { \
        float acc4 = 0.0; \
        DUST_VW_R4(0u) DUST_VW_R4(1u) DUST_VW_R4(2u) DUST_VW_R4(3u) \
        li = acc4 * (hi - lo); \
    } \
    acc += nw.y * li; }
// a texel's tau from its units at dust_v_unit[u0 ..] (the loops' accumulations in their order)
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
float dust_vw_reduce(int ty, int tx, uint u0) {
    float sz = float(1u << DUST_VG_LEV);
    vec4 rc = dust_vw_rect(ty, tx, sz);
    float xa_ = rc.x, xb_ = rc.y, ya = rc.z, yb = rc.w;
    if (!(xb_ > xa_ && yb > ya)) return uintBitsToFloat(0x7FC00000u);
    uint kSub = DUST_VG_KSUB;
    float acc = 0.0;
    if (DUST_VG_CLS == DUST_VOLUME_CLASS_SUBRAY) {
        DUST_VW_RJ(0u) DUST_VW_RJ(1u) DUST_VW_RJ(2u) DUST_VW_RJ(3u)
        return acc;
    }
    bool alongX = ((dust_v_gx[DUST_VP_FLAGS] >> 3u) & 3u) == 1u;
    float lo = alongX ? xa_ : ya, hi = alongX ? xb_ : yb;
    DUST_VW_RL(0u) DUST_VW_RL(1u) DUST_VW_RL(2u) DUST_VW_RL(3u)
    return acc / (hi - lo);
}
#endif

#ifdef DUST_PROFILE
// lane 0 after pass B: vplan (the record's start to here), the profile pair's allTop modes
// (dust_volume_draw's: 2 the pieces and the SMALL ones, 1 / 3 the hit flag for the end)
void dust_vw_prof_pass_b(uint n) {
    uint now = DUST_VW_CLOCK;
    atomicAdd(DUST_VA(DUST_VA_VPLAN), now - DUST_VA(DUST_VA_T0));
    DUST_VA(DUST_VA_T1) = now;
    uint profMode = (DUST_PW(DUST_PYR_FLAGS) >> DUST_FRAME_PROF_PAIR_SHIFT) & 3u;
    bool topKnown = ((dust_v_st[DUST_VS_CTL] >> 19u) & 1u) != 0u;
    bool hit = false;
    if (profMode >= 1u && DUST_VA(DUST_VA_ALLTOP) != 0u && (profMode == 1u || !topKnown)) {
        uint nSmall = 0u;
        [[unroll]] for (uint k = 0u; k < DUST_VOLUME_PIECES_MAX; ++k) if (k < n) nSmall += (DUST_VWP(k, DUST_VP_CODE) & 3u) == DUST_VOLUME_CLASS_SMALL ? 1u : 0u;
        if (profMode == 2u) {
            atomicAdd(DUST_PW(DUST_PYR_PROF_PAIR), n);
            atomicAdd(DUST_PW(DUST_PYR_PROF_PAIR + 1u), nSmall);
        }
        hit = profMode == 1u || (profMode == 3u && nSmall == n);
    }
    DUST_VA(DUST_VA_ALLTOP) = hit ? 2u : 0u;
}
// lane 0 at the record's end: vdraw, the record's cycles, the slowest, the profile pair's hit
void dust_vw_prof_end() {
    uint now = DUST_VW_CLOCK;
    atomicAdd(DUST_VA(DUST_VA_VDRAW), now - DUST_VA(DUST_VA_T1));
    uint dt = now - DUST_VA(DUST_VA_T0);
    atomicAdd(DUST_VA(DUST_VA_THREAD), dt);
    atomicMax(DUST_VA(DUST_VA_MAX), dt);
    if (DUST_VA(DUST_VA_ALLTOP) == 2u) {
        atomicAdd(DUST_PW(DUST_PYR_PROF_PAIR), 1u);
        atomicAdd(DUST_PW(DUST_PYR_PROF_PAIR + 1u), dt >> 8u);
    }
}
#endif

// pass 2 of a segment (dust_t_vwalk's arithmetic: dust_volume_deposit's prefix in column order),
// whether it writes (a texel's counts >= 1 and writes on: dust_volume_deposit's true)
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
bool dust_vw_walk(uint row, uint n, float carry, float area, float invUnit, bool noWrites, inout float peak) {
    bool any_ = false;
    [[unroll]] for (uint j = 0u; j < DUST_SEGMENT_TEXELS; ++j) {
      if (j < n) {
        float tauPx = uintBitsToFloat(dust_t_row[row + j]);
        float out_ = 0.0;
        if ((tauPx >= 0.0) && !isinf(tauPx)) {
            peak = max(peak, tauPx);
            float tau = tauPx * area * invUnit;
            float tauC = tau + carry;
            float counts = min(floor(tauC + 0.5), DUST_COUNT_TEXEL_MAX);
            carry = tauC - counts;
            if ((counts >= 1.0) && !noWrites) out_ = counts;
        }
        any_ = any_ || out_ != 0.0;
        dust_t_row[row + j] = floatBitsToUint(out_);
      }
    }
    return any_;
}
#endif

// lane 0: the walked group's peaks and written flags into the record's (the group's piece still in
// dust_v_gx), then the next group: up to DUST_T_SEGS row segments of one drawn piece from the
// cursor, in DUST_VOLUME_ROWS' order (the pieces in order, a piece's rows, a row's segments from
// its first column), into dust_v_seg[par * 8 ..]
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
void dust_vw_take(uint par, uint g) {
    if (g > 0u) {
        uint pb = (par ^ 1u) * 8u;
        [[unroll]] for (uint s = 0u; s < DUST_T_SEGS; ++s) {
            if (dust_v_seg[pb + 4u + s] != 0u) {
                dust_v_st[DUST_VS_PEAK] = floatBitsToUint(max(DUST_VSF(DUST_VS_PEAK), uintBitsToFloat(dust_v_sp[s])));
                if (dust_v_sp[4u + s] != 0u) dust_v_st[DUST_VS_MASK] |= 1u << DUST_VG_LEV;
            }
        }
    }
    uint ctl = dust_v_st[DUST_VS_CTL];
    uint n = (ctl & 31u) * ((ctl >> 5u) & 31u);
    uint p = dust_v_ctl[DUST_VC_CP];
    bool fresh = dust_v_ctl[DUST_VC_CY] == 0xFFFFFFFFu;
    if (fresh) {
        while (p < n && (DUST_VWP(p, DUST_VP_DESC) & (1u << 13u)) == 0u) ++p;
    }
    uint cnt = 0u;
    uint b = par * 8u;
    if (p < n) {
        uint rx = DUST_VWP(p, DUST_VP_RX), ry = DUST_VWP(p, DUST_VP_RY);
        int x0 = int(rx & 0xFFFFu), x1 = int(rx >> 16u), y1 = int(ry >> 16u);
        int ty = fresh ? int(ry & 0xFFFFu) : int(dust_v_ctl[DUST_VC_CY]);
        int xa = fresh ? x0 : int(dust_v_ctl[DUST_VC_CX]);
        while (cnt < DUST_T_SEGS && ty <= y1) {
            int xb = min(xa + int(DUST_SEGMENT_TEXELS) - 1, x1);
            dust_v_seg[b + cnt] = (uint(ty) << 16u) | uint(xa);
            dust_v_seg[b + 4u + cnt] = uint(xb - xa + 1);
            DUST_PROF_COUNT(volume_texels, xb - xa + 1)
            ++cnt;
            xa += int(DUST_SEGMENT_TEXELS);
            if (xa > x1) { xa = x0; ++ty; }
        }
        dust_v_ctl[DUST_VC_NEW] = (g == 0u || fresh) ? 1u : 0u;
        dust_v_ctl[DUST_VC_GP] = p;
        if (ty > y1) { dust_v_ctl[DUST_VC_CP] = p + 1u; dust_v_ctl[DUST_VC_CY] = 0xFFFFFFFFu; }
        else { dust_v_ctl[DUST_VC_CP] = p; dust_v_ctl[DUST_VC_CY] = uint(ty); dust_v_ctl[DUST_VC_CX] = uint(xa); }
    } else {
        dust_v_ctl[DUST_VC_CP] = p;
    }
    [[unroll]] for (uint c = 0u; c < DUST_T_SEGS; ++c) if (c >= cnt) dust_v_seg[b + 4u + c] = 0u;
    dust_v_ctl[DUST_VC_N0 + par] = cnt;
}
#endif

// the group's piece into dust_v_gx (a lane a word; a small piece's three points, lanes 9-11)
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
void dust_vw_context(uint lane) {
    uint p = dust_v_ctl[DUST_VC_GP];
    if (lane < 9u) dust_v_gx[lane] = DUST_VWP(p, lane);
    if (lane >= 9u && lane < 12u && ((DUST_VWP(p, DUST_VP_DESC) >> 8u) & 3u) == DUST_VOLUME_CLASS_SMALL) {
        uint q = lane - 9u;
#ifdef DUST_VW_STAGE
        // PBSc computed them (the same call on the same words: no small-piece input changes after
        // PBSb, and bit 13 is only ever cleared after PBSc): its row words mx my ix iy m rho
        uint o = DUST_VW_ROW + DUST_VW_SX_PT + 18u * p + 6u * q;
        dust_v_gx[DUST_VG_PT + 6u * q + 0u] = pc.pyramid.v[o + 0u];
        dust_v_gx[DUST_VG_PT + 6u * q + 1u] = pc.pyramid.v[o + 1u];
        dust_v_gx[DUST_VG_PT + 6u * q + 2u] = pc.pyramid.v[o + 2u];
        dust_v_gx[DUST_VG_PT + 6u * q + 3u] = pc.pyramid.v[o + 3u];
        dust_v_gx[DUST_VG_PT + 6u * q + 4u] = pc.pyramid.v[o + 5u];
        dust_v_gx[DUST_VG_PT + 6u * q + 5u] = pc.pyramid.v[o + 4u];
#else
        DustSmallPoint s = dust_vw_small_point(p, q);
        dust_v_gx[DUST_VG_PT + 6u * q + 0u] = floatBitsToUint(s.mx);
        dust_v_gx[DUST_VG_PT + 6u * q + 1u] = floatBitsToUint(s.my);
        dust_v_gx[DUST_VG_PT + 6u * q + 2u] = floatBitsToUint(s.ix);
        dust_v_gx[DUST_VG_PT + 6u * q + 3u] = floatBitsToUint(s.iy);
        dust_v_gx[DUST_VG_PT + 6u * q + 4u] = floatBitsToUint(s.rho);
        dust_v_gx[DUST_VG_PT + 6u * q + 5u] = floatBitsToUint(s.m);
#endif
    }
}
#endif

// the texel passes of group par (slot = segment * 16 + column, the units into dust_t_row's taus):
// the centre and small classes one pass, a lane a texel (dust_vw_unit1); the sub-ray (kSub^2
// units) and line (4 kSub) classes in passes of 64 / c texels from slot dust_v_ctl[DUST_VC_CH]:
// the line's closed forms (dust_vw_units_line), the units into dust_v_unit (dust_vw_units_eval),
// then a lane per texel reduces them in the loops' order (dust_vw_units_reduce). One step each
#define DUST_VW_COST (DUST_VG_CLS == DUST_VOLUME_CLASS_SUBRAY ? DUST_VG_KSUB * DUST_VG_KSUB : 4u * DUST_VG_KSUB)
#define DUST_VW_TP (DUST_VW_LANES / DUST_VW_COST)
#define DUST_VW_SLOTS(par) (dust_v_ctl[DUST_VC_N0 + (par)] * DUST_SEGMENT_TEXELS)
#define DUST_VW_ONE_PASS (DUST_VG_CLS == DUST_VOLUME_CLASS_CENTRE || DUST_VG_CLS == DUST_VOLUME_CLASS_SMALL)
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
void dust_vw_units_line(uint lane, uint par) {
    uint b = par * 8u;
    uint slot = dust_v_ctl[DUST_VC_CH] + lane / DUST_VW_COST;
    uint s = slot / DUST_SEGMENT_TEXELS, x = slot % DUST_SEGMENT_TEXELS;
    if (lane < DUST_VW_TP * DUST_VW_COST && slot < DUST_VW_SLOTS(par) && x < dust_v_seg[b + 4u + s]) {
        uint key = dust_v_seg[b + s];
        dust_vw_unit_line(int(key >> 16u), int((key & 0xFFFFu) + x), lane % DUST_VW_COST);
    }
}
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
void dust_vw_units_eval(uint lane, uint par) {
    uint b = par * 8u;
    if (DUST_VW_ONE_PASS) {
        float tau = uintBitsToFloat(0x7FC00000u);
        uint s = lane / DUST_SEGMENT_TEXELS, x = lane % DUST_SEGMENT_TEXELS;
        if (x < dust_v_seg[b + 4u + s]) {
            uint key = dust_v_seg[b + s];
            tau = dust_vw_unit1(int(key >> 16u), int((key & 0xFFFFu) + x));
        }
        dust_t_row[lane] = floatBitsToUint(tau);
        return;
    }
    uint slot = dust_v_ctl[DUST_VC_CH] + lane / DUST_VW_COST;
    uint s = slot / DUST_SEGMENT_TEXELS, x = slot % DUST_SEGMENT_TEXELS;
    uint u = 0u;
    if (lane < DUST_VW_TP * DUST_VW_COST && slot < DUST_VW_SLOTS(par) && x < dust_v_seg[b + 4u + s]) {
        uint key = dust_v_seg[b + s];
        u = dust_vw_unit(int(key >> 16u), int((key & 0xFFFFu) + x), lane % DUST_VW_COST);
    }
    dust_v_unit[lane] = u;
}
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
void dust_vw_units_reduce(uint lane, uint par) {
    uint b = par * 8u;
    if (lane < DUST_VW_TP && dust_v_ctl[DUST_VC_CH] + lane < DUST_VW_SLOTS(par)) {
        uint slot = dust_v_ctl[DUST_VC_CH] + lane;
        uint s = slot / DUST_SEGMENT_TEXELS, x = slot % DUST_SEGMENT_TEXELS;
        float tau = uintBitsToFloat(0x7FC00000u);
        if (x < dust_v_seg[b + 4u + s]) {
            uint key = dust_v_seg[b + s];
            tau = dust_vw_reduce(int(key >> 16u), int((key & 0xFFFFu) + x), lane * DUST_VW_COST);
        }
        dust_t_row[slot] = floatBitsToUint(tau);
    }
}
#endif

// pass 2 of the group's segments: a lane a segment, the walk; its peak and written flag into
// dust_v_sp (dust_vw_take folds them)
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
void dust_vw_walk_group(uint lane, uint par) {
    if (lane >= DUST_T_SEGS) return;
    uint nt = dust_v_seg[par * 8u + 4u + lane];
#if defined(DUST_VW_STAGE) && defined(DUST_VW_WALK_SP)
    dust_v_sp[lane] = floatBitsToUint(0.0);
    dust_v_sp[4u + lane] = 0u;
    if (nt != 0u) {
        float peak = 0.0;
#else
    float peak = 0.0;
    bool wrote = false;
    if (nt != 0u) {
#endif
        float sz = float(1u << DUST_VG_LEV);
        bool noWrites = ((DUST_PW(DUST_PYR_SPLIT_DEPTH) >> 30u) & 3u) == 2u;
        float carry = dust_segment_carry(dust_hash32(DUST_VRW(18u) ^ dust_v_ctl[DUST_VC_GP]),
                                         dust_v_seg[par * 8u + lane]);
#if defined(DUST_VW_STAGE) && defined(DUST_VW_WALK_SP)
        bool wrote = dust_vw_walk(lane * DUST_SEGMENT_TEXELS, nt, carry, sz * sz, pc.invUnit, noWrites, peak);
        dust_v_sp[lane] = floatBitsToUint(peak);
        dust_v_sp[4u + lane] = wrote ? 1u : 0u;
    }
#else
        wrote = dust_vw_walk(lane * DUST_SEGMENT_TEXELS, nt, carry, sz * sz, pc.invUnit, noWrites, peak);
    }
    dust_v_sp[lane] = floatBitsToUint(peak);
    dust_v_sp[4u + lane] = wrote ? 1u : 0u;
#endif
}
#endif

// pass 3 of the walked group par (its piece still in dust_v_gx): a lane per texel, the deposit of
// its counts (dust_t_vdeposit; the accumulate mode's deposit into the scratch sums)
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
void dust_vw_deposit(uint lane, uint par) {
    uint s = lane / DUST_SEGMENT_TEXELS, x = lane % DUST_SEGMENT_TEXELS;
    if (x >= dust_v_seg[par * 8u + 4u + s]) return;
    float counts = uintBitsToFloat(dust_t_row[lane]);
    if (counts == 0.0) return;
    uint key = dust_v_seg[par * 8u + s];
    uint lev = DUST_VG_LEV;
    uint i = DUST_PW(DUST_PYR_TABLE + 3u * lev)
           + ((key >> 16u) * DUST_PW(DUST_PYR_TABLE + 3u * lev + 1u) + (key & 0xFFFFu) + x) * DUST_PYRAMID_TEXEL_WORDS;
    vec3 color = vec3(DUST_VRF(20u), DUST_VRF(21u), DUST_VRF(22u));
    uint depthBits = dust_v_st[DUST_VS_DEPTH];
#ifdef DUST_SUBGROUP
    // (perf9_O fix3) where the top-level path's and pass A's deposits share one cw: the accumulate
    // mode (dust_volume_deposit's acc branch, the sums in the scratch words) and dust_t_vdeposit's
    // writes in one loop, c and cw computed once before the branch, as dust_volume_deposit did (two
    // copies of the expression were contracted differently: the top texel's r, g off by a few)
    bool acc = ((dust_v_st[DUST_VS_CTL] >> 19u) & 1u) != 0u
        && i == DUST_PW(DUST_PYR_TABLE + 3u * (dust_v_st[DUST_VS_LEVELS] - 1u));
    uvec2 pyr = uvec2(pc.pyramid);
    float hf = float(dust_hash32(i) >> 8u) * (1.0 / 16777216.0);
    if (counts > DUST_COUNT_MAX_PER_ADD) {
        if (acc) atomicAdd(DUST_VA(DUST_VA_SPLITS), 1u);
        else atomicAdd(DUST_T_PV(pyr)[DUST_T_PV(pyr)[DUST_PYR_QUEUE_BASE] + DUST_QUEUE_WORDS + DUST_VQUEUE_SPLIT_ADDS], 1u);
    }
    for (float rest = counts; rest > 0.5; ) {
        uint c = uint(min(rest, DUST_COUNT_MAX_PER_ADD));
        uvec3 cw = uvec3(float(c) * color * DUST_COLOR_SCALE + hf);
        if (acc) {
            uint o0 = atomicAdd(DUST_VA(DUST_VA_SUM + 0u), c), o1 = atomicAdd(DUST_VA(DUST_VA_SUM + 1u), cw.r),
                 o2 = atomicAdd(DUST_VA(DUST_VA_SUM + 2u), cw.g), o3 = atomicAdd(DUST_VA(DUST_VA_SUM + 3u), cw.b);
            uint wr = (o0 + c < c ? 1u : 0u) | (o1 + cw.r < cw.r ? 2u : 0u) | (o2 + cw.g < cw.g ? 4u : 0u)
                    | (o3 + cw.b < cw.b ? 8u : 0u);
            if (wr != 0u) atomicOr(DUST_VA(DUST_VA_WRAP), wr);
        } else {
            uint split = DUST_T_PV(pyr)[DUST_PYR_SPLIT_DEPTH];
            dust_t_add_sat(pyr, i, c, split);
            dust_t_add_sat(pyr, i + 1u, cw.r, split);
            dust_t_add_sat(pyr, i + 2u, cw.g, split);
            dust_t_add_sat(pyr, i + 3u, cw.b, split);
        }
        rest -= float(c);
    }
    if (acc) {
        atomicMin(DUST_VA(DUST_VA_DEPTH), depthBits);
        return;
    }
    uint cur = DUST_T_PV(pyr)[i + 4u];
    if (cur == 0u || depthBits < cur) {
        uint seen = atomicCompSwap(DUST_T_PV(pyr)[i + 4u], 0u, depthBits);
        if (seen != 0u) atomicMin(DUST_T_PV(pyr)[i + 4u], depthBits);
    }
#else
    dust_t_vdeposit(uvec2(pc.pyramid), i, counts, color, depthBits);
#endif
}
#endif

// ---- the records: one loop of steps --------------------------------------------------------------
// The workgroup's records one after another, each a sequence of steps; an iteration of the
// driver's loop runs one step (every lane), a barrier, lane 0's choice of the next step (into
// dust_v_ctl[DUST_VC_PH]), a barrier. One loop: a loop around any step (the records, the cap
// loops, the texel groups, the unit passes) costs Mali-G52 the 32-register mode
const uint DUST_VW_PH_REC = 0u;     // the record's words, its state
const uint DUST_VW_PH_AXES = 1u;    // R1: the axes (lane 0)
const uint DUST_VW_PH_COV = 2u;     // R1: the covariance (lane 0) and the bound (lane 1)
const uint DUST_VW_PH_A1 = 3u;      // pass A's steps a1 .. a4b (DUST_VW_PH_A1 + 0 .. 4)
const uint DUST_VW_PH_TOPK = 8u;    // pass A, top known: the drawn flags
const uint DUST_VW_PH_A5A = 9u;     // pass A's a5a, a5b, a5c (+ 0 .. 2)
const uint DUST_VW_PH_DECIDE = 12u; // lane 0: the cap loops' decision
const uint DUST_VW_PH_COV4 = 13u;   // lane 0: the merged pieces' covariance
const uint DUST_VW_PH_PB = 14u;     // pass B
const uint DUST_VW_PH_FOLD = 15u;   // lane 0: the codes, the evaluations, the cursor
const uint DUST_VW_PH_TAKE = 16u;   // group g - 1's deposits, lane 0 takes group g
const uint DUST_VW_PH_CTX = 17u;    // the group's piece
const uint DUST_VW_PH_LINE = 18u;   // a pass's line closed forms
const uint DUST_VW_PH_EVAL = 19u;   // a pass's units
const uint DUST_VW_PH_RED = 20u;    // a pass's reductions
const uint DUST_VW_PH_WALK = 21u;   // the group's walk
const uint DUST_VW_PH_END = 22u;    // lane 0: the record's end words
const uint DUST_VW_PH_NEXT = 23u;   // (lane 0's: the next record)
const uint DUST_VW_PH_DONE = 24u;
const uint DUST_VW_PH_PBS = 25u;    // pass B's small pieces: + 0 .. 4 (dust_vw_small_a .. e)
const uint DUST_VW_PH_PB2 = 30u;    // pass B's ranges
const uint DUST_VW_PH_PLAN = 31u;   // lane 0: the pieces per axis, the renorm
const uint DUST_VW_PH_PB1 = 32u;    // pass B's order bound (the top / gated level's)
const uint DUST_VW_PH_A2B = 33u;    // pass A's a2 corners 4 .. 7
const uint DUST_VW_PH_LOAD = 34u;   // staged: the record's state from its scratch row
const uint DUST_VW_PH_SAVE = 35u;   // staged: the record's state into its scratch row
#ifdef DUST_VW_STAGE
#if DUST_VW_STAGE == 1
#define DUST_VW_IN_STAGE(p) ((p) == DUST_VW_PH_AXES || (p) == DUST_VW_PH_PLAN || (p) == DUST_VW_PH_COV)
#elif DUST_VW_STAGE == 2
#define DUST_VW_IN_STAGE(p) (((p) >= DUST_VW_PH_A1 && (p) <= DUST_VW_PH_COV4) || (p) == DUST_VW_PH_A2B)
#elif DUST_VW_STAGE == 3 && defined(DUST_VW_PART) && DUST_VW_PART == 1
#define DUST_VW_IN_STAGE(p) ((p) == DUST_VW_PH_PB || (p) == DUST_VW_PH_PB1 || (p) == DUST_VW_PH_PB2)
#elif DUST_VW_STAGE == 3 && defined(DUST_VW_PART) && DUST_VW_PART == 2
#define DUST_VW_IN_STAGE(p) ((p) == DUST_VW_PH_PBS || (p) == DUST_VW_PH_PBS + 1u)
#elif DUST_VW_STAGE == 3
#define DUST_VW_IN_STAGE(p) ((p) == DUST_VW_PH_PB || (p) == DUST_VW_PH_PB1 || (p) == DUST_VW_PH_PB2 \
                             || (p) == DUST_VW_PH_PBS || (p) == DUST_VW_PH_PBS + 1u)
#elif DUST_VW_STAGE == 4 && defined(DUST_VW_PART) && DUST_VW_PART == 1
#define DUST_VW_IN_STAGE(p) ((p) == DUST_VW_PH_PBS + 2u)
#elif DUST_VW_STAGE == 4 && defined(DUST_VW_PART) && DUST_VW_PART == 2
#define DUST_VW_IN_STAGE(p) ((p) >= DUST_VW_PH_PBS + 3u && (p) <= DUST_VW_PH_PBS + 4u)
#elif DUST_VW_STAGE == 4
#define DUST_VW_IN_STAGE(p) ((p) >= DUST_VW_PH_PBS + 2u && (p) <= DUST_VW_PH_PBS + 4u)
#elif DUST_VW_STAGE == 5
#define DUST_VW_IN_STAGE(p) ((p) >= DUST_VW_PH_TAKE && (p) <= DUST_VW_PH_NEXT)
#else
#error "DUST_VW_STAGE is 1 .. 5"
#endif
// LOAD (every lane): the state words, the axes (S2-S4), pass B's codes and evaluations (S4), the
// piece rows (S3; S2 writes them before reading), the resume phase; S5 the texel cursor (FOLD's)
void dust_vw_load(uint lane) {
    uint row = DUST_VW_ROW;
    uint res = pc.pyramid.v[row + DUST_VW_SX_RES];  // (every lane the same word)
    if (res == DUST_VW_PH_NEXT || !DUST_VW_IN_STAGE(res)) { if (lane == 0u) dust_v_ctl[DUST_VC_RES] = res; return; }
    if (lane < 32u) dust_v_st[lane] = pc.pyramid.v[row + DUST_VW_SX_ST + lane];
#if DUST_VW_STAGE <= 4 && !defined(DUST_VW_S3B)
    if (lane < 15u) dust_v_gx[lane] = pc.pyramid.v[row + DUST_VW_SX_GX + lane];
#endif
#if DUST_VW_STAGE == 4
    if (lane < 16u) {
        dust_v_unit[lane] = pc.pyramid.v[row + DUST_VW_SX_UNIT + lane];
        dust_t_row[lane] = pc.pyramid.v[row + DUST_VW_SX_TROW + lane];
    }
#endif
#if DUST_VW_STAGE == 3 && !defined(DUST_VW_S3B)
    {
        uint c = pc.pyramid.v[row + DUST_VW_SX_ST + DUST_VS_CTL];
        uint nw = (c & 31u) * ((c >> 5u) & 31u) * DUST_VW_PIECE_WORDS;
        [[unroll]] for (uint it = 0u; it < 7u; ++it) {
            uint i = lane + it * DUST_VW_LANES;
            if (i < nw) dust_v_pr[i] = pc.pyramid.v[row + DUST_VW_SX_PR + i];
        }
    }
#endif
    if (lane == 0u) {
        dust_v_ctl[DUST_VC_RES] = pc.pyramid.v[row + DUST_VW_SX_RES];
#if DUST_VW_STAGE == 5
        dust_v_ctl[DUST_VC_CP] = 0u;
        dust_v_ctl[DUST_VC_CY] = 0xFFFFFFFFu;
        dust_v_ctl[DUST_VC_G] = 0u;
#endif
    }
}
// SAVE (every lane): the state words, the axes, pass B's codes and evaluations (S3), the piece
// rows (S2, S3), the resume phase
void dust_vw_save(uint lane) {
    uint row = DUST_VW_ROW;
#ifndef DUST_VW_S3B
    if (lane < 32u) pc.pyramid.v[row + DUST_VW_SX_ST + lane] = dust_v_st[lane];
    if (lane < 15u) pc.pyramid.v[row + DUST_VW_SX_GX + lane] = dust_v_gx[lane];
#endif
#if DUST_VW_STAGE == 3 && !defined(DUST_VW_S3B)
    if (lane < 16u) {
        pc.pyramid.v[row + DUST_VW_SX_UNIT + lane] = dust_v_unit[lane];
        pc.pyramid.v[row + DUST_VW_SX_TROW + lane] = dust_t_row[lane];
    }
#endif
#if DUST_VW_STAGE == 2 || (DUST_VW_STAGE == 3 && !defined(DUST_VW_S3B))
    {
        uint nw = (dust_v_st[DUST_VS_CTL] & 31u) * ((dust_v_st[DUST_VS_CTL] >> 5u) & 31u) * DUST_VW_PIECE_WORDS;
        [[unroll]] for (uint it = 0u; it < 7u; ++it) {
            uint i = lane + it * DUST_VW_LANES;
            if (i < nw) pc.pyramid.v[row + DUST_VW_SX_PR + i] = dust_v_pr[i];
        }
    }
#endif
    if (lane == 0u) pc.pyramid.v[row + DUST_VW_SX_RES] = dust_v_ctl[DUST_VC_RES];
}
#endif
#define DUST_VW_J dust_v_ctl[DUST_VC_J]
#define DUST_VW_N ((dust_v_st[DUST_VS_CTL] & 31u) * ((dust_v_st[DUST_VS_CTL] >> 5u) & 31u))
#define DUST_VW_TOPK (((dust_v_st[DUST_VS_CTL] >> 19u) & 1u) != 0u)

// step REC
void dust_vw_rec(uint lane) {
    if (lane < DUST_VQUEUE_RECORD_WORDS)
        dust_v_rec[lane] = pc.pyramid.v[dust_v_ctl[DUST_VC_RA] + DUST_VW_J + lane * dust_v_ctl[DUST_VC_CAP]];
    if (lane == 0u) {
#ifdef DUST_PROFILE
        DUST_VA(DUST_VA_T0) = DUST_VW_CLOCK;
#endif
    }
}

// step FOLD (lane 0): the bad codes in k order, the top evaluations, the cursor
void dust_vw_fold() {
    uint n = DUST_VW_N;
    uint ev = 0u;
    [[unroll]] for (uint k = 0u; k < DUST_VOLUME_PIECES_MAX; ++k) {
      if (k < n) {
        uint bad = dust_v_unit[k];
        if (bad != 0u) {
            atomicAdd(DUST_PW(DUST_PYR_BAD_CODE), 1u);
            atomicCompSwap(DUST_PW(DUST_PYR_BAD_CODE_FIRST), 0u, ((bad >> 1u) & 0xFFFFu) | (k << 16u) | (n << 24u));
        }
        ev += dust_t_row[k];
      }
    }
    dust_v_st[DUST_VS_TOPEV] = ev;
    dust_v_ctl[DUST_VC_CP] = 0u;
    dust_v_ctl[DUST_VC_CY] = 0xFFFFFFFFu;
    dust_v_ctl[DUST_VC_G] = 0u;
#ifdef DUST_PROFILE
    dust_vw_prof_pass_b(n);
#endif
}

// step END (lane 0): the record's end words
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5  // the texel loop: stage 5 only
void dust_vw_end() {
    float peak = DUST_VSF(DUST_VS_PEAK);
    uint maskBits = dust_v_st[DUST_VS_MASK];
    bool top = DUST_VW_TOPK;
    uint ev = dust_v_st[DUST_VS_TOPEV];
#ifdef DUST_SUBGROUP
    if (top) {
        if (peak > 0.0) atomicMax(DUST_VA(DUST_VA_PEAK), floatBitsToUint(peak));
        if (maskBits != 0u) atomicOr(DUST_VA(DUST_VA_MASK), maskBits);
        if (ev != 0u) atomicAdd(DUST_VA(DUST_VA_EVALS), ev);
    } else
#endif
    {
        if (peak > 0.0) atomicMax(DUST_PW(DUST_PYR_TAU_MAX), floatBitsToUint(peak));
        if (maskBits != 0u) atomicOr(DUST_PW(DUST_PYR_LEVEL_MASK), maskBits);
        if (top && ev != 0u) atomicAdd(DUST_PW(DUST_PYR_VOLUME_EVALS), ev);
    }
#ifdef DUST_PROFILE
    dust_vw_prof_end();
#endif
}
#endif

// pass A's first step of a cap-loop iteration (a1 unless the stage keeps the levels)
uint dust_vw_pass_a_first() {
    if (dust_v_st[DUST_VS_STATE] != 2u) return DUST_VW_PH_A1;
    return DUST_VW_TOPK ? DUST_VW_PH_TOPK : DUST_VW_PH_A5A;
}

// lane 0, after step ph's barrier: the next step (and the step's lane-0 bookkeeping)
uint dust_vw_next(uint ph) {
#if defined(DUST_VW_STAGE) && DUST_VW_STAGE > 1
    if (ph == DUST_VW_PH_REC) return DUST_VW_PH_LOAD;
    if (ph == DUST_VW_PH_LOAD) {
        uint r = dust_v_ctl[DUST_VC_RES];
        if (r != DUST_VW_PH_NEXT && DUST_VW_IN_STAGE(r)) return r;  // (else LOAD skipped the record)
        DUST_VW_J = DUST_VW_J + 1u;
        return DUST_VW_J >= dust_v_ctl[DUST_VC_COUNT] ? DUST_VW_PH_DONE : DUST_VW_PH_REC;
    }
#else
    if (ph == DUST_VW_PH_REC) return DUST_VW_PH_AXES;
#endif
    if (ph == DUST_VW_PH_AXES) return dust_v_st[DUST_VS_STATE] == 3u ? DUST_VW_PH_NEXT : DUST_VW_PH_PLAN;
    if (ph == DUST_VW_PH_PLAN) return DUST_VW_PH_COV;
    if (ph == DUST_VW_PH_COV || ph == DUST_VW_PH_COV4) return dust_vw_pass_a_first();
    if (ph == DUST_VW_PH_A1 + 1u) return DUST_VW_PH_A2B;
    if (ph == DUST_VW_PH_A2B) return DUST_VW_PH_A1 + 2u;
    if (ph >= DUST_VW_PH_A1 && ph < DUST_VW_PH_A1 + 4u) return ph + 1u;
    if (ph == DUST_VW_PH_A1 + 4u) return DUST_VW_TOPK ? DUST_VW_PH_TOPK : DUST_VW_PH_A5A;
    if (ph == DUST_VW_PH_A5A || ph == DUST_VW_PH_A5A + 1u) return ph + 1u;
    if (ph == DUST_VW_PH_TOPK || ph == DUST_VW_PH_A5A + 2u) return DUST_VW_PH_DECIDE;
    if (ph == DUST_VW_PH_DECIDE) {
        uint s = dust_v_st[DUST_VS_STATE];
        if (s == 4u) return DUST_VW_PH_COV4;
        if (s != 3u && s != 5u) return dust_vw_pass_a_first();
#ifdef DUST_PROFILE
        uint profMode = (DUST_PW(DUST_PYR_FLAGS) >> DUST_FRAME_PROF_PAIR_SHIFT) & 3u;
        if (profMode == 0u && s == 3u) {
            atomicAdd(DUST_PW(DUST_PYR_PROF_PAIR), 1u);
            atomicAdd(DUST_PW(DUST_PYR_PROF_PAIR + 1u), (DUST_VW_CLOCK - DUST_VA(DUST_VA_T0)) >> 8u);
        }
        if (s == 3u) {
            uint dt = DUST_VW_CLOCK - DUST_VA(DUST_VA_T0);
            atomicAdd(DUST_VA(DUST_VA_VPLAN), dt);
            atomicAdd(DUST_VA(DUST_VA_THREAD), dt);
            atomicMax(DUST_VA(DUST_VA_MAX), dt);
        }
#endif
        return s == 3u ? DUST_VW_PH_NEXT : DUST_VW_PH_PB;
    }
    if (ph == DUST_VW_PH_PB) return DUST_VW_PH_PB1;
    if (ph == DUST_VW_PH_PB1) return DUST_VW_PH_PB2;
    if (ph == DUST_VW_PH_PB2) return DUST_VW_PH_PBS;
    if (ph >= DUST_VW_PH_PBS && ph < DUST_VW_PH_PBS + 4u) return ph + 1u;
    if (ph == DUST_VW_PH_PBS + 4u) {
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 4
        dust_vw_fold();
#endif
        return DUST_VW_PH_TAKE;
    }
    uint par = dust_v_ctl[DUST_VC_G] & 1u;
    if (ph == DUST_VW_PH_TAKE) {
        if (dust_v_ctl[DUST_VC_N0 + par] == 0u) return DUST_VW_PH_END;
        dust_v_ctl[DUST_VC_CH] = 0u;
        if (dust_v_ctl[DUST_VC_NEW] != 0u) return DUST_VW_PH_CTX;
        return DUST_VG_CLS == DUST_VOLUME_CLASS_LINE ? DUST_VW_PH_LINE : DUST_VW_PH_EVAL;
    }
    if (ph == DUST_VW_PH_CTX) return DUST_VG_CLS == DUST_VOLUME_CLASS_LINE ? DUST_VW_PH_LINE : DUST_VW_PH_EVAL;
    if (ph == DUST_VW_PH_LINE) return DUST_VW_PH_EVAL;
    if (ph == DUST_VW_PH_EVAL) return DUST_VW_ONE_PASS ? DUST_VW_PH_WALK : DUST_VW_PH_RED;
    if (ph == DUST_VW_PH_RED) {
        if (dust_v_ctl[DUST_VC_CH] + DUST_VW_TP >= DUST_VW_SLOTS(par)) return DUST_VW_PH_WALK;
        dust_v_ctl[DUST_VC_CH] = dust_v_ctl[DUST_VC_CH] + DUST_VW_TP;
        return DUST_VG_CLS == DUST_VOLUME_CLASS_LINE ? DUST_VW_PH_LINE : DUST_VW_PH_EVAL;
    }
    if (ph == DUST_VW_PH_WALK) {
        dust_v_ctl[DUST_VC_G] = dust_v_ctl[DUST_VC_G] + 1u;
        return DUST_VW_PH_TAKE;
    }
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5
    if (ph == DUST_VW_PH_END) dust_vw_end();
#endif
    // NEXT (or after END): the next record
    DUST_VW_J = DUST_VW_J + 1u;
    return DUST_VW_J >= dust_v_ctl[DUST_VC_COUNT] ? DUST_VW_PH_DONE : DUST_VW_PH_REC;
}

// step ph (every lane)
void dust_vw_step(uint lane, uint ph) {
    if (ph == DUST_VW_PH_REC) { dust_vw_rec(lane); return; }
#ifdef DUST_VW_STAGE
    if (ph == DUST_VW_PH_LOAD) { dust_vw_load(lane); return; }
    if (ph == DUST_VW_PH_SAVE) { dust_vw_save(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 1
    if (ph == DUST_VW_PH_PLAN) { if (lane == 0u) dust_vw_plan(); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 1
    if (ph == DUST_VW_PH_AXES) { if (lane == 0u) dust_vw_axes(); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 1
    if (ph == DUST_VW_PH_COV) {
        if (lane == 0u) dust_vw_cov();
        if (lane == 1u) dust_vw_bound();
        if (lane == 2u) dust_vw_renorm();
        return;
    }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_COV4) { if (lane == 0u) dust_vw_cov(); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_DECIDE) { if (lane == 0u) dust_vw_decide(); return; }
#endif
    uint n = DUST_VW_N;
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_A1 + 0u) { if (lane < n) dust_vw_a1(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_A1 + 1u) { if (lane < n) dust_vw_a2(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_A2B) { if (lane < n) dust_vw_a2b(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_A1 + 2u) { if (lane < n) dust_vw_a3(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_A1 + 3u) { if (lane < n) dust_vw_a4(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_A1 + 4u) { if (lane < n) dust_vw_a4b(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_TOPK) { if (lane < n) dust_v_unit[lane] = (DUST_VWP(lane, DUST_VP_FLAGS) >> 1u) & 1u; return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_A5A + 0u) {
        if (lane < n) dust_vw_a5a(lane, ((DUST_PW(DUST_PYR_SPLIT_DEPTH) >> 8u) & 0xFFu) + (dust_v_st[DUST_VS_STAGE] & 0xFFu));
        return;
    }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_A5A + 1u) { if (lane < n) dust_vw_a5b(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 2
    if (ph == DUST_VW_PH_A5A + 2u) { if (lane < n) dust_vw_a5c(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 3
    if (ph == DUST_VW_PH_PB) { if (lane < n) dust_vw_pass_b(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 3
    if (ph == DUST_VW_PH_PB1) { if (lane < n) dust_vw_pass_b1(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 3
    if (ph == DUST_VW_PH_PB2) { if (lane < n) dust_vw_pass_b2(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 3
    if (ph == DUST_VW_PH_PBS + 0u) { if (lane < n) dust_vw_small_a(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 3
    if (ph == DUST_VW_PH_PBS + 1u) { if (lane < n) dust_vw_small_b(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 4
    if (ph == DUST_VW_PH_PBS + 2u) { if (lane < 3u * n) dust_vw_small_c(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 4
    if (ph == DUST_VW_PH_PBS + 3u) { if (lane < n) dust_vw_small_d(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 4
    if (ph == DUST_VW_PH_PBS + 4u) { if (lane < n) dust_vw_small_e(lane); return; }
#endif
    uint par = dust_v_ctl[DUST_VC_G] & 1u;
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5
    if (ph == DUST_VW_PH_TAKE) {
        if (dust_v_ctl[DUST_VC_G] > 0u) dust_vw_deposit(lane, par ^ 1u);
        if (lane == 0u) dust_vw_take(par, dust_v_ctl[DUST_VC_G]);
        return;
    }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5
    if (ph == DUST_VW_PH_CTX) { dust_vw_context(lane); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5
    if (ph == DUST_VW_PH_LINE) { dust_vw_units_line(lane, par); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5
    if (ph == DUST_VW_PH_EVAL) { dust_vw_units_eval(lane, par); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5
    if (ph == DUST_VW_PH_RED) { dust_vw_units_reduce(lane, par); return; }
#endif
#if !defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5
    if (ph == DUST_VW_PH_WALK) { dust_vw_walk_group(lane, par); return; }
#endif
}

// ---- the workgroup -------------------------------------------------------------------------------
// one atomic add into the 64-bit counter at (w, w + 1) (dust_prof_add64's, from one lane)
void dust_vw_add64(uint w, uint v) {
    if (v == 0u) return;
    uint old = atomicAdd(pc.pyramid.v[w], v);
    if (old > 0xFFFFFFFFu - v) atomicAdd(pc.pyramid.v[w + 1u], 1u);
}
#ifdef DUST_SUBGROUP
// one word of the accumulate mode's sums (dust_volume_flush_word's semantics from the workgroup's
// sum mod 2^32 and its wrap bit: the word saturates when the total passed 2^32, in the sum or in
// the running word; the probe plain_add never saturates)
void dust_vw_flush_word(uint w, uint v, bool wrapped, bool plain) {
    bool over = wrapped;
    if (v != 0u) {
        uint old = atomicAdd(pc.pyramid.v[w], v);
        over = over || old + v < old;
    }
    if (over && !plain) atomicMax(pc.pyramid.v[w], 0xFFFFFFFFu);
}
#endif

// the workgroup's records of the chunk's volume queue, or of its volume spill with
// DUST_VIEW_VOLUME_SPILL (every lane)
void dust_volume_draw_workgroup(uint lane) {
    if (lane == 0u) {
        bool spill = (pc.flags & DUST_VIEW_VOLUME_SPILL) != 0u;
        uint base = spill ? dust_vw_vspill_base() : dust_vw_vqueue_base();
        uint cap = dust_vw_vqueue_capacity(base);
        uint n = min(pc.pyramid.v[base + 3u], cap);
        uint r0 = gl_WorkGroupID.x * DUST_VW_LANES;
        dust_v_ctl[DUST_VC_COUNT] = r0 < n ? min(n - r0, DUST_VW_LANES) : 0u;
        dust_v_ctl[DUST_VC_RA] = base + DUST_VQUEUE_HEADER_WORDS + r0;
        dust_v_ctl[DUST_VC_CAP] = cap;
        dust_v_ctl[DUST_VC_J] = 0u;
#ifdef DUST_VW_STAGE
        dust_v_ctl[DUST_VC_SR] = dust_vw_vscratch_base(spill) + r0 * DUST_VW_SX_WORDS;
#endif
        dust_v_gx[DUST_VG_ZERO] = 0u;
    }
    barrier();
    uint count = dust_v_ctl[DUST_VC_COUNT];
    if (count == 0u) return;
#if (defined(DUST_SUBGROUP) || defined(DUST_PROFILE)) && (!defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5)
    if (lane < DUST_VA_WORDS) DUST_VA(lane) = lane == DUST_VA_DEPTH ? 0xFFFFFFFFu : 0u;
#endif
    // the records' steps (dust_vw_step, dust_vw_next)
    if (lane == 0u) dust_v_ctl[DUST_VC_PH] = DUST_VW_PH_REC;
    for (;;) {
        barrier();
        uint ph = dust_v_ctl[DUST_VC_PH];
        if (ph == DUST_VW_PH_DONE) break;
#ifdef DUST_VW_STAGE
#ifndef DUST_VW_NO_OPQ
        // the lane through the shared zero, read each step: no lane compare or lane-derived address
        // hoisted out of the loop and live across every step (Adreno 619: predicates saved in packed
        // half registers, private memory)
        uint ln = lane + dust_v_gx[DUST_VG_ZERO];
#else
        uint ln = lane;
#endif
        dust_vw_step(ln, ph);
        barrier();
#ifndef DUST_VW_NO_PHR
        uint pn = dust_v_ctl[DUST_VC_PH];  // (the phase again: no step compare kept across the barrier)
#else
        uint pn = ph;
#endif
        if (ln == 0u) {
            uint nx = dust_vw_next(pn);
#else
        dust_vw_step(lane, ph);
        barrier();
        if (lane == 0u) {
            uint nx = dust_vw_next(ph);
#endif
#ifdef DUST_VW_STAGE
            // a phase of another stage: the record's state into its row, the next record
            if (nx != DUST_VW_PH_DONE && nx != DUST_VW_PH_REC && nx != DUST_VW_PH_LOAD && !DUST_VW_IN_STAGE(nx)) {
                dust_v_ctl[DUST_VC_RES] = nx;
                nx = DUST_VW_PH_SAVE;
            }
#endif
            dust_v_ctl[DUST_VC_PH] = nx;
        }
    }
#if defined(DUST_SUBGROUP) || defined(DUST_PROFILE)
    barrier();
    if (lane != 0u) return;
#endif
#if defined(DUST_SUBGROUP) && (!defined(DUST_VW_STAGE) || DUST_VW_STAGE == 5)
    {
        uint levels = DUST_PW(DUST_PYR_LEVELS);
        uint off = DUST_PW(DUST_PYR_TABLE + 3u * (levels - 1u));
        bool plain = ((DUST_PW(DUST_PYR_SPLIT_DEPTH) >> 30u) & 3u) == 1u;
        uint wrap = DUST_VA(DUST_VA_WRAP);
        dust_vw_flush_word(off,      DUST_VA(DUST_VA_SUM + 0u), (wrap & 1u) != 0u, plain);
        dust_vw_flush_word(off + 1u, DUST_VA(DUST_VA_SUM + 1u), (wrap & 2u) != 0u, plain);
        dust_vw_flush_word(off + 2u, DUST_VA(DUST_VA_SUM + 2u), (wrap & 4u) != 0u, plain);
        dust_vw_flush_word(off + 3u, DUST_VA(DUST_VA_SUM + 3u), (wrap & 8u) != 0u, plain);
        uint d = DUST_VA(DUST_VA_DEPTH);
        if (d != 0xFFFFFFFFu) {
            uint cur = pc.pyramid.v[off + 4u];
            if (cur == 0u || d < cur) {
                uint seen = atomicCompSwap(pc.pyramid.v[off + 4u], 0u, d);
                if (seen != 0u) atomicMin(pc.pyramid.v[off + 4u], d);
            }
        }
        uint splits = DUST_VA(DUST_VA_SPLITS);
        if (splits != 0u) atomicAdd(pc.pyramid.v[dust_vw_vqueue_base() + DUST_VQUEUE_SPLIT_ADDS], splits);
        uint peakBits = DUST_VA(DUST_VA_PEAK);
        if (peakBits != 0u) atomicMax(DUST_PW(DUST_PYR_TAU_MAX), peakBits);
        uint mask = DUST_VA(DUST_VA_MASK);
        if (mask != 0u) atomicOr(DUST_PW(DUST_PYR_LEVEL_MASK), mask);
        uint evals = DUST_VA(DUST_VA_EVALS);
        if (evals != 0u) atomicAdd(DUST_PW(DUST_PYR_VOLUME_EVALS), evals);
    }
#endif
#ifdef DUST_PROFILE
    {
        uint base = pc.pyramid.v[DUST_PYR_QUEUE_BASE];
        dust_vw_add64(base + DUST_PROF_CTHREAD, DUST_VA(DUST_VA_THREAD));
        dust_vw_add64(base + DUST_PROF_CVPLAN, DUST_VA(DUST_VA_VPLAN));
        dust_vw_add64(base + DUST_PROF_CVDRAW, DUST_VA(DUST_VA_VDRAW));
        atomicAdd(pc.pyramid.v[base + DUST_PROF_CPACKETS], count);
        dust_vw_add64(base + DUST_PROF_CTEXELS, DUST_VA(DUST_VA_TEXELS));
        uint sf = DUST_VA(DUST_VA_SMALL_FAST);
        if (sf != 0u) atomicAdd(pc.pyramid.v[base + DUST_PROF_SMALL_PATHS + 0u], sf);
        uint mx = DUST_VA(DUST_VA_MAX);
        if (mx != 0u) atomicMax(pc.pyramid.v[base + DUST_PROF_CMAX], mx);
    }
#endif
}
