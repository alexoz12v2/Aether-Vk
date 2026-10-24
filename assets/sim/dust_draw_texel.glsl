// @assets/sim/dust_draw_texel.glsl
//
// perf9_O (the tight occupancy budget: <= 32 registers a lane, <= 2 KB shared a 64-lane
// workgroup): the texel-lane drawing of the workgroup consumers. Include after dust_scatter.glsl.
//
// A row segment of DUST_SEGMENT_TEXELS texels (dust_scatter_segment) is drawn by 16 lanes in three
// passes over a 64-entry shared row (four segments a pass):
//   1. a lane per texel: the texel's tau, the segment loop's expression (capsule x area x invUnit,
//      x the weight when lit), into dust_t_row[lane];
//   2. a lane per segment (4 lanes): the segment's carry walk over its 16 taus in column order,
//      the loop's rounding verbatim (the skip test, min, + carry, + 0.5, the new carry), the counts
//      of each texel into the same slot (0: no deposit);
//   3. a lane per texel: the deposit of its counts (the saturating adds, the colour dither, the
//      nearest depth), the loop body after `counts == 0u` verbatim.
// The deposits are order-free atomics (dust_atomic_add_sat's proof; the colour dither is hashed per
// texel word, the depth is a minimum), so the pyramid is word for word the one-lane segment's.
// Two barriers a group (1 -> 2, 2 -> 3); pass 3 of a group and pass 1 of the next run in one phase,
// each lane reading its slot's counts before it writes its next tau.

// No helper takes a buffer reference: the Adreno 619 compiler puts a DustUintBuffer parameter in
// private memory (160 B in dust_draw_queue.comp). The helpers take the address (uvec2(pc.pyramid))
// and read and write through DustUintBuffer(pyr).v[...] at each access; dust_atomic_add_sat,
// dust_phase_function and dust_radiance_weight have pointer-free copies here (the same
// statements), and the probe bits are a parameter instead of the global dust_pyr_split_word.
#define DUST_T_PV(pyr) DustUintBuffer(pyr).v
// dust_atomic_add_sat with the probe bits as a parameter (split = DUST_PYR_SPLIT_DEPTH's word)
void dust_t_add_sat(uvec2 pyr, uint i, uint b, uint split) {
    if (b == 0u) return;
    uint old = atomicAdd(DUST_T_PV(pyr)[i], b);
    if (((split >> 30u) & 3u) == 1u) return;
    if (old + b < old) atomicMax(DUST_T_PV(pyr)[i], 0xFFFFFFFFu);
}
// dust_phase_function, dust_radiance_weight (the phase table in the pyramid's header)
float dust_t_phase_function(uvec2 pyr, float alphaRad) {
    float deg = clamp(alphaRad * (180.0 / DUST_PI_F32), 0.0, 180.0);
    uint i = min(uint(deg), 179u);
    float f = deg - float(i);
    float a = uintBitsToFloat(DUST_T_PV(pyr)[DUST_PYR_PHASE_LUT + i]);
    float b = uintBitsToFloat(DUST_T_PV(pyr)[DUST_PYR_PHASE_LUT + i + 1u]);
    return a + (b - a) * f;
}
float dust_t_radiance_weight(uvec2 pyr, float albedo, vec3 sunDir, vec3 toEye, float rhAu) {
    float n = dot(toEye, toEye);
    float cosA = n > 0.0 ? dot(sunDir, toEye) / sqrt(n) : 1.0;
    float alpha = acos(clamp(cosA, -1.0, 1.0));
    float r = max(rhAu, 1e-6);
    return albedo * dust_t_phase_function(pyr, alpha) / (r * r);
}

const uint DUST_T_LANES = 64u;                                   // the workgroup (local_size_x)
const uint DUST_T_SEGS = DUST_T_LANES / DUST_SEGMENT_TEXELS;     // segments a pass (4)
// the lane's texel tau (pass 1), replaced in place by its counts (pass 2: the walk reads a texel's
// tau before it writes its counts, a lane at a time in column order), read by pass 3
shared uint  dust_t_row[DUST_T_LANES];

// pass 2 of a segment whose 16 taus sit at dust_t_tau[row .. row + n - 1]: the carry walk of
// dust_scatter_segment (and dust_volume_deposit's identical prefix), counts into dust_t_cnt
void dust_t_carry_walk(uint row, uint n, float carry) {
    // not unrolled: an unrolled walk lets the scheduler hoist its 16 shared loads (GCN: 20 -> 48 VGPRs)
    [[dont_unroll]] for (uint j = 0u; j < n; ++j) {
        float tau = uintBitsToFloat(dust_t_row[row + j]);
        uint counts = 0u;
        if ((tau >= 0.0) && !isinf(tau)) {
            float tauC = min(tau, DUST_COUNT_MAX_PER_ADD) + carry;
            counts = uint(tauC + 0.5);
            carry = tauC - float(counts);
        }
        dust_t_row[row + j] = counts;
    }
}

// pass 3: dust_scatter_segment's deposit of `counts` at texel word i (full: colour and depth)
void dust_t_deposit_full(uvec2 pyr, uint i, uint counts, vec3 color, uint depthBits) {
    uint split = DUST_T_PV(pyr)[DUST_PYR_SPLIT_DEPTH];
    dust_t_add_sat(pyr, i, counts, split);
    // colour words floor(counts * c + h), h a hashed threshold per texel, an ordered
    // dither (dust::scatter_into): rounding counts * c + 0.5 per deposit turned every
    // faint texel into (1, 1, 0)
    float hf = float(dust_hash32(i) >> 8u) * (1.0 / 16777216.0);
    uvec3 n = uvec3(float(counts) * color * DUST_COLOR_SCALE + hf);
    dust_t_add_sat(pyr, i + 1u, n.r, split);
    dust_t_add_sat(pyr, i + 2u, n.g, split);
    dust_t_add_sat(pyr, i + 3u, n.b, split);
    // nearest dust: 0 means "no dust yet" (positive f32 bits order as uints)
    uint cur = DUST_T_PV(pyr)[i + 4u];
    if (cur == 0u || depthBits < cur) {
        uint seen = atomicCompSwap(DUST_T_PV(pyr)[i + 4u], 0u, depthBits);
        if (seen != 0u) atomicMin(DUST_T_PV(pyr)[i + 4u], depthBits);
    }
}

// ---- the volumetric deposit (dust_volume_deposit) split for texel lanes -----------------------
// pass 2 of a volumetric row segment: dust_volume_deposit's prefix in column order (the skip test,
// the peak, tau, + carry, the counts and the new carry), the counts (float bits, 0: no deposit)
// into the same slot; the segment's peak into peak
void dust_t_vwalk(uint row, uint n, float carry, float area, float invUnit, bool noWrites, inout float peak) {
    [[dont_unroll]] for (uint j = 0u; j < n; ++j) {
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
        dust_t_row[row + j] = floatBitsToUint(out_);
    }
}
// pass 3: dust_volume_deposit's writes of `counts` (>= 1) at texel word i, verbatim with its
// accumulate mode off (dust_vacc false: the edge pass, its only caller, never accumulates; the
// branch is gone so that no private global (dust_vacc_*) is read on the path)
void dust_t_vdeposit(uvec2 pyr, uint i, float counts, vec3 color, uint depthBits) {
    float hf = float(dust_hash32(i) >> 8u) * (1.0 / 16777216.0);
    if (counts > DUST_COUNT_MAX_PER_ADD) {
        atomicAdd(DUST_T_PV(pyr)[DUST_T_PV(pyr)[DUST_PYR_QUEUE_BASE] + DUST_QUEUE_WORDS + DUST_VQUEUE_SPLIT_ADDS], 1u);
    }
    for (float rest = counts; rest > 0.5; ) {
        uint c = uint(min(rest, DUST_COUNT_MAX_PER_ADD));
        uvec3 cw = uvec3(float(c) * color * DUST_COLOR_SCALE + hf);
        uint split = DUST_T_PV(pyr)[DUST_PYR_SPLIT_DEPTH];
        dust_t_add_sat(pyr, i, c, split);
        dust_t_add_sat(pyr, i + 1u, cw.r, split);
        dust_t_add_sat(pyr, i + 2u, cw.g, split);
        dust_t_add_sat(pyr, i + 3u, cw.b, split);
        rest -= float(c);
    }
    uint cur = DUST_T_PV(pyr)[i + 4u];
    if (cur == 0u || depthBits < cur) {
        uint seen = atomicCompSwap(DUST_T_PV(pyr)[i + 4u], 0u, depthBits);
        if (seen != 0u) atomicMin(DUST_T_PV(pyr)[i + 4u], depthBits);
    }
}

#ifdef DUST_QUEUE_TEXEL
// ---- dust_draw_queue.comp: the EWA record of the queue drawn by texel lanes ------------------
// A loop iteration draws a group of up to DUST_T_SEGS segments of the record's band in four
// barrier-separated phases:
//   1a lane 0 takes the next segments in dust_scatter_rows' order (row by row, a row's segments
//      left to right), the row's column range from dust_band_columns once per row: each
//      segment's key row << 16 | first column and its texel count (0: none);
//   1b a lane per texel (segment lane / 16, column lane % 16): the texel's tau;
//   2  a lane per segment: the carry walk (the segment's key seeds it, as in dust_scatter_segment);
//   3  a lane per texel: the deposit.
// The segments, their keys and their texels are dust_scatter_rows' (stride 1); only which lanes
// evaluate them changes, and the deposits are order-free (see the header).
// Workgroup-uniform state is in shared words, read at use (Mali-G52 holds a workgroup-uniform
// register in every lane): the record's words, its band, the row cursor. Every branch around a
// barrier tests words that no lane writes between the barriers enclosing the test.
shared uint dust_t_key[DUST_T_SEGS];    // the group's segments: row << 16 | first column
shared uint dust_t_n[DUST_T_SEGS];      // their texel counts (0: none)
// 0 the record's word address, 1 the band's last row, 2 a texel was written, 3 the cursor's row,
// 4 its next segment in that row (0: the row's columns are not computed yet), 5/6 the row's
// columns x0/x1, 7 the group's segment count (0: the band is done)
shared uint dust_qt_ctl[8];
// the record's words 0 .. DUST_QT_REC_WORDS - 1, copied into shared memory once (lane w copies
// word w): a buffer word read in two phases is a load the compiler may merge across the barrier
// between them (barrier() orders shared memory only), keeping it live through every phase
const uint DUST_QT_REC_WORDS = 11u + 33u;
shared uint dust_qt_rec[DUST_QT_REC_WORDS];
#define DUST_QT_W(k) dust_qt_rec[k]
#define DUST_QT_F(k) uintBitsToFloat(DUST_QT_W(k))
#define DUST_QT_S float(1u << (DUST_QT_W(0u) & 0xFFu))
#define DUST_SP_U(k) DUST_QT_W(11u + (k))
#define DUST_SP_F(k) uintBitsToFloat(DUST_SP_U(k))
// the record's band (dust_band_new of its packet, lane 0 once): n, dd, w, uLo, uHi, ext (mu and d
// are the packet's words, point is !(dd > 1e-12) again)
shared float dust_qt_band[8];
#include "dust_texel_packet.glsl"

#ifdef DUST_PROFILE
// the profiled build's slots (see dust_qt_prof_flush)
shared uint dust_qt_t0[DUST_T_LANES];
shared uint dust_qt_pf[11];
// lane 0 only: the 64-bit sum at words w, w + 1 plus v
void dust_qt_pf_fold(uint w, uint v) {
    uint lo = dust_qt_pf[w] + v;
    dust_qt_pf[w + 1u] += lo < v ? 1u : 0u;
    dust_qt_pf[w] = lo;
}
#endif

// lane 0, phase 1a: the next up-to-DUST_T_SEGS segments from the cursor
void dust_qt_take_segments() {
    uint cnt = 0u;
    int y = int(dust_qt_ctl[3]);
    uint j = dust_qt_ctl[4];
    int x0 = int(dust_qt_ctl[5]), x1 = int(dust_qt_ctl[6]);
    int y1 = int(dust_qt_ctl[1]);
    while (cnt < DUST_T_SEGS && y <= y1) {
        if (j == 0u) {
            DustBand B;
            B.mu = vec2(DUST_SP_F(0u), DUST_SP_F(1u)); B.d = vec2(DUST_SP_F(5u), DUST_SP_F(6u));
            B.n = vec2(dust_qt_band[0], dust_qt_band[1]); B.dd = dust_qt_band[2]; B.w = dust_qt_band[3];
            B.uLo = dust_qt_band[4]; B.uHi = dust_qt_band[5]; B.point = !(B.dd > 1e-12);
            B.ext = vec2(dust_qt_band[6], dust_qt_band[7]);
            if (!dust_band_columns(B, (float(y) + 0.5) * DUST_QT_S, DUST_QT_S, DUST_QT_W(3u), x0, x1)) {
                ++y;
                continue;
            }
        }
        int xa = x0 + int(j * DUST_SEGMENT_TEXELS);
        int xb = min(xa + int(DUST_SEGMENT_TEXELS) - 1, x1);
        dust_t_key[cnt] = (uint(y) << 16u) | uint(xa);
        dust_t_n[cnt] = uint(xb - xa + 1);
        ++cnt;
        ++j;
        if (j >= uint(x1 - x0) / DUST_SEGMENT_TEXELS + 1u) { j = 0u; ++y; }
    }
    dust_qt_ctl[7] = cnt;
#ifdef DUST_PROFILE
    // the previous pass's qseg times (each lane's, in its row slot after its deposit read; 0 before
    // the first pass) and this pass's texels (the walk's lanes added them before)
    for (uint l = 0u; l < DUST_T_LANES; ++l) dust_qt_pf_fold(4u, dust_t_row[l]);
    for (uint c = 0u; c < cnt; ++c) dust_qt_pf_fold(8u, dust_t_n[c]);
#endif
    for (uint c = cnt; c < DUST_T_SEGS; ++c) dust_t_n[c] = 0u;
    dust_qt_ctl[3] = uint(y);
    dust_qt_ctl[4] = j;
    dust_qt_ctl[5] = uint(x0);
    dust_qt_ctl[6] = uint(x1);
}

#ifdef DUST_PROFILE
// The profiled build (dust_prof_flush_queue's counters, the same words and meaning: per lane the
// cycles of qload (thread start -> the rows), qrows, qseg (the texel's tau), qthread, the texels,
// the slowest lane's qthread). No profile value is held in a register across the row loop and no
// lane adds into a word another lane adds into: a lane's values are in its own slots (the start
// clock in dust_qt_t0; its qload, then each pass's qseg time, then its rows-end time in its
// dust_t_row slot whenever the row is free: before the loop, after the lane's deposit read, after
// the loop), and lane 0 folds the 64 slots into the (lo, hi) sums of dust_qt_pf with plain adds
// (qload at the rows' start, qseg at the next pass's segment fetch, the rest in the flush), then
// adds those to the pyramid. LDS stays at 832 B: above 1024 B RADV's scheduler spends VGPRs up to
// the LDS-limited wave count (a second 64-slot array: 1088 B, GCN 36 VGPRs). (With the probes' registers live across the loop and
// dust_prof_flush_queue's subgroup sums after it, Mali-G52 used 60-63 work registers; with every
// lane's atomicAdd / atomicMax at one shared word, RADV made them subgroup reductions / scans whose
// two linear VGPRs are live from the program's top: GCN 36 VGPRs.) Words: qload 0, (rows end -
// thread start) 2, qseg 4, qthread 6, texels 8, max qthread 10; qrows = word 2 - word 0 (sum of
// (rows end - rows start)).
// (the slots and dust_qt_pf_fold are declared above dust_qt_take_segments)
void dust_qt_pf_add64(uvec2 pyr, uint w, uint lo, uint hi) {
    uint old = atomicAdd(DUST_T_PV(pyr)[w], lo);
    hi += (old > 0xFFFFFFFFu - lo) ? 1u : 0u;
    if (hi > 0u) atomicAdd(DUST_T_PV(pyr)[w + 1u], hi);
}
// every lane of the workgroup calls it (a barrier) after dust_queue_rows_texel
void dust_qt_prof_flush(uvec2 pyr, uint base, uint lane) {
    dust_qt_t0[lane] = clock2x32ARB().x - dust_qt_t0[lane];
    barrier();
    if (lane == 0u) {
        for (uint l = 0u; l < DUST_T_LANES; ++l) {
            dust_qt_pf_fold(2u, dust_t_row[l]);
            dust_qt_pf_fold(6u, dust_qt_t0[l]);
            dust_qt_pf[10] = max(dust_qt_pf[10], dust_qt_t0[l]);
        }
        uint s0 = base + DUST_PROF_QSTAGE0;
        dust_qt_pf_add64(pyr, s0 + 0u, dust_qt_pf[0], dust_qt_pf[1]);
        uint rLo = dust_qt_pf[2] - dust_qt_pf[0];
        uint rHi = dust_qt_pf[3] - dust_qt_pf[1] - (dust_qt_pf[2] < dust_qt_pf[0] ? 1u : 0u);
        dust_qt_pf_add64(pyr, s0 + 2u, rLo, rHi);
        dust_qt_pf_add64(pyr, s0 + 4u, dust_qt_pf[4], dust_qt_pf[5]);
        dust_qt_pf_add64(pyr, s0 + 6u, dust_qt_pf[6], dust_qt_pf[7]);
        atomicAdd(DUST_T_PV(pyr)[base + DUST_PROF_QRECORDS], 1u);
        dust_qt_pf_add64(pyr, base + DUST_PROF_QTEXELS, dust_qt_pf[8], dust_qt_pf[9]);
        if (dust_qt_pf[10] > 0u) atomicMax(DUST_T_PV(pyr)[base + DUST_PROF_QMAX_WG], dust_qt_pf[10]);
    }
}
#endif

// the record at `at` drawn by the workgroup; every lane calls it (the barriers); whether a texel
// was written (lane 0 answers, false on the others)
bool dust_queue_rows_texel(uvec2 pyr, uint at, uint lane) {
#ifdef DUST_PROFILE
    dust_t_row[lane] = clock2x32ARB().x - dust_qt_t0[lane];
    if (lane < 11u) dust_qt_pf[lane] = 0u;
#endif
    if (lane < DUST_QT_REC_WORDS) dust_qt_rec[lane] = DUST_T_PV(pyr)[at + lane];
    barrier();
    if (lane == 0u) {
#ifdef DUST_PROFILE
        // qload: the lanes' row slots, then 0 there (the first pass's qseg fold reads them)
        for (uint l = 0u; l < DUST_T_LANES; ++l) {
            dust_qt_pf_fold(0u, dust_t_row[l]);
            dust_t_row[l] = 0u;
        }
#endif
        dust_qt_ctl[0] = at;
        dust_qt_ctl[2] = 0u;
        int y0, y1;
        // the packet from the record's shared words (dust_queue_get_screen_packet's fields)
        DustScreenPacket P;
        P.mu = vec2(DUST_SP_F(0u), DUST_SP_F(1u));
        P.cov = vec3(DUST_SP_F(2u), DUST_SP_F(3u), DUST_SP_F(4u));
        P.seg = vec2(DUST_SP_F(5u), DUST_SP_F(6u));
        P.amp = DUST_SP_F(7u);
        P.sigmaMin = DUST_SP_F(8u);
        P.det = DUST_SP_F(9u);
        P.depthAu = DUST_SP_F(10u);
        P.openStart = (DUST_SP_U(11u) & 1u) != 0u;
        P.openEnd = (DUST_SP_U(11u) & 2u) != 0u;
        P.rho = vec2(DUST_SP_F(12u), DUST_SP_F(13u));
        P.mean = vec3(DUST_SP_F(14u), DUST_SP_F(15u), DUST_SP_F(16u));
        P.kx = vec3(DUST_SP_F(17u), DUST_SP_F(18u), DUST_SP_F(19u));
        P.ky = vec3(DUST_SP_F(20u), DUST_SP_F(21u), DUST_SP_F(22u));
        P.sun = vec3(DUST_SP_F(23u), DUST_SP_F(24u), DUST_SP_F(25u));
        P.rhAu = DUST_SP_F(26u);
        P.eye = vec4(DUST_SP_F(27u), DUST_SP_F(28u), DUST_SP_F(29u), DUST_SP_F(30u));
        P.albedo = DUST_SP_F(31u);
        P.wMean = DUST_SP_F(32u);
        DustBand B = dust_band_new(P);
        dust_qt_band[0] = B.n.x; dust_qt_band[1] = B.n.y; dust_qt_band[2] = B.dd; dust_qt_band[3] = B.w;
        dust_qt_band[4] = B.uLo; dust_qt_band[5] = B.uHi; dust_qt_band[6] = B.ext.x; dust_qt_band[7] = B.ext.y;
        if (!dust_band_rows(B, DUST_QT_S, DUST_QT_W(4u), y0, y1)) {
            y0 = 1; y1 = 0;
        }
        y0 = max(y0, int(DUST_QT_W(5u)));
        dust_qt_ctl[1] = uint(min(y1, int(DUST_QT_W(6u))));
        dust_qt_ctl[3] = uint(y0);
        dust_qt_ctl[4] = 0u;
    }
    for (;;) {
        // phase 1a (the barrier before it: lane 0's previous deposit-phase reads are done)
        barrier();
        if (lane == 0u) dust_qt_take_segments();
        barrier();
        if (dust_qt_ctl[7] == 0u) break;
        // phase 1b: the texel's tau (dust_scatter_segment's expression and order); a lane past its
        // segment's end writes NaN, which the carry walk skips as the segment loop's bound did
#ifdef DUST_PROFILE
        uint dSeg;
#endif
        {
            uint sg = lane / DUST_SEGMENT_TEXELS;
            // (perf9_O fix) a lane of a segment not taken (dust_t_n[sg] == 0: its key is stale or never
            // written) holds 0, no deposit (the walk visits taken segments only); past a taken
            // segment's end NaN, which the walk skips
            float tau = uintBitsToFloat(dust_t_n[sg] != 0u ? 0x7FC00000u : 0u);
#ifdef DUST_PROFILE
            // qseg: the phase's cycles of the lane, its start clock in the lane's free row slot (a
            // register across the tau: Mali-G52 62 work registers)
            dust_t_row[lane] = clock2x32ARB().x;
#endif
            if (lane % DUST_SEGMENT_TEXELS < dust_t_n[sg]) {
                uint key = dust_t_key[sg];
                float s = DUST_QT_S;
                float yc = (float(key >> 16u) + 0.5) * s;
                float area = s * s;
                vec2 x = vec2((float((key & 0xFFFFu) + lane % DUST_SEGMENT_TEXELS) + 0.5) * s, yc);
                tau = dust_capsule_tau_at(pyr, x) * area * DUST_QT_F(10u);
                if (DUST_QT_F(11u + 31u) > 0.0) tau *= dust_texel_weight_at(pyr, x);
            }
#ifdef DUST_PROFILE
            dSeg = clock2x32ARB().x - dust_t_row[lane];
#endif
            dust_t_row[lane] = floatBitsToUint(tau);

        }
        barrier();
        // phase 2: the segment's carry walk, a lane per segment
        if (lane < DUST_T_SEGS && dust_t_n[lane] != 0u) {
            dust_t_carry_walk(lane * DUST_SEGMENT_TEXELS, DUST_SEGMENT_TEXELS, dust_segment_carry(DUST_QT_W(1u), dust_t_key[lane]));
        }
        barrier();
        // phase 3: the deposit (a lane past its segment's end has counts 0 from its NaN)
        {
            uint counts = dust_t_row[lane];
#ifdef DUST_PROFILE
            // the lane's qseg time into its own slot once read (lane 0 folds it at the next pass)
            dust_t_row[lane] = dSeg;
#endif
            if (counts != 0u) {
                uint key = dust_t_key[lane / DUST_SEGMENT_TEXELS];
                uint i = DUST_QT_W(2u) + ((key >> 16u) * DUST_QT_W(3u) + (key & 0xFFFFu) + lane % DUST_SEGMENT_TEXELS) * DUST_PYRAMID_TEXEL_WORDS;
                dust_t_deposit_full(pyr, i, counts, vec3(DUST_QT_F(7u), DUST_QT_F(8u), DUST_QT_F(9u)),
                                    floatBitsToUint(max(DUST_QT_F(11u + 10u), 0.0)));
                atomicOr(dust_qt_ctl[2], 1u);
            }
        }
    }
#ifdef DUST_PROFILE
    // the row is free after the loop (its last reads, the deposits, precede the loop's barrier)
    dust_t_row[lane] = clock2x32ARB().x - dust_qt_t0[lane];
#endif
    // lane 0 answers for the workgroup (the level mask's OR is the same once or 64 times)
    return lane == 0u && dust_qt_ctl[2] != 0u;
}
#undef DUST_QT_W
#undef DUST_QT_F
#undef DUST_QT_S
#undef DUST_SP_U
#undef DUST_SP_F
#endif
