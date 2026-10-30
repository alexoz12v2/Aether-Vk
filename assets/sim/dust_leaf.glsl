// @assets/sim/dust_leaf.glsl
//
// The leaf path of the dust splat (dust_split.comp: a workgroup per cluster, its sub-packet
// records from dust_splat.comp's queue): the perspective split of a packet, its EWA leaves
// (scatter: in-thread rows and the EWA queue) and its volumetric packets (dust_volume_emit: the
// volume queue, or the volume spill; drawn by dust_draw_volume.comp). The rounding seeds
// (dust_next_packet_seed) are drawn in the order the per-cluster thread drew them: the splitter's counting pass
// (dust_counting) and its prefix give every record its ordinal base. Include after the
// push-constant block `pc` (dust::DustSplatPushConstants) and dust_frame_gates.glsl.

// the volume packet queue is on this frame (PYR_FLAGS bit DUST_FRAME_VQUEUE_OFF clear; set in main)
bool dust_vqueue_on = false;
// the sub-packet record (recs + r) of the leaf being drawn (the leaf record's
// index to the exposure, colour and photometry; set per record by dust_project_split_scatter)
uint dust_leaf_rec = 0u;

// ── The prep stage (dust_split.comp -DDUST_SPLIT_PREP, a lane per record) ─────────────────────
// A perspective packet its support cuts (dust::project_split_scatter; not with a speck shell, the
// split's hand-over): drawn whole by the exact ray integral over the half-space when the eye is
// near it (DUST_SQUEUE_KIND_VOLUME), else as its cone of grains when its cell has an origin
// (dust_conic_piece, the cut baked in; DUST_SQUEUE_KIND_CONIC), else as its truncation's moments
// (dust_support_trunc_piece; DUST_SQUEUE_KIND_TRUNC); any other packet whole (DUST_SQUEUE_KIND_PLAIN).
// The same tests in the same order as the walk's former preamble; the kind and the pieces'
// derived state go to the record's cone words (DUST_SQUEUE_PUT_CONE) for the walk stage, whose
// lanes then carry no root and no cone state (the walk's spill on Mali).
// The prep stage on the record (a lane per record): the root packet's words and the cone
// words read from the record at each use through one per-lane pointer (volatile: no load merged
// with an earlier one or hoisted, nothing of the packet kept in registers across the pieces), the
// cone words written as they are decided (DUST_SQUEUE_PUT_CONE's words and values) and re-read by
// the pieces. The pieces compute their kept flux w only (q is not used by the prep: the same
// expressions as dust_conic_piece / dust_support_trunc_piece up to w, the rest was dead).
layout(buffer_reference, std430, buffer_reference_align = 4) volatile buffer DustPRow { uint v[]; };
uvec2 dust_p_row;
// the pieces read the record's words through the plain (non-volatile) view: within a piece the
// compiler may merge a word's loads (no store between), and the pieces hold no value across calls
layout(buffer_reference, std430, buffer_reference_align = 4) buffer DustPRowN { uint v[]; };
#define PN(k) uintBitsToFloat(DustPRowN(dust_p_row).v[(k) * DUST_SQUEUE_CAPACITY])
#ifdef DUST_SPLIT_PREP
shared uint dust_p_lrs[64];
#define DUST_O_PRL() { memoryBarrierShared(); dust_p_row = dust_p_addr(dust_p_lrs[gl_LocalInvocationID.x]); }
#else
#define DUST_O_PRL()
#endif
uvec2 dust_p_addr(uint word) { uvec2 a = uvec2(pc.pyramid); uint o = word * 4u; uint lo = a.x + o; return uvec2(lo, a.y + (lo < o ? 1u : 0u)); }
#define PR(k) DustPRow(dust_p_row).v[(k) * DUST_SQUEUE_CAPACITY]
#define PW(k) PR(k)
#define PRF(k) uintBitsToFloat(PR(k))
#define P_MEAN vec3(PRF(0u), PRF(1u), PRF(2u))
#define P_SEG vec3(PRF(9u), PRF(10u), PRF(11u))
#define P_N vec3(PRF(23u), PRF(24u), PRF(25u))
#define P_SN vec3(PRF(34u), PRF(35u), PRF(36u))
// the frame values the prep reads per record, re-derived at their uses (nothing of them
// kept in registers over the pieces): the view's perspective flag (dust_view_forward's test on the
// push constants), the speck shell's radius for this record (the main's former recShell), and the
// end of the cluster's records (main's recs + recBase + n, from the queue words, volatile)
bool dust_p_persp() { vec3 r3 = vec3(pc.mvp[0][3], pc.mvp[1][3], pc.mvp[2][3]); return dot(r3, r3) > 0.0; }
float dust_p_shell() {
    return (PR(14u) & 8u) != 0u ? uintBitsToFloat(DustPRow(uvec2(pc.pyramid)).v[DUST_PYR_SPECKS_DMAX]) : 0.0;
}
// dust_legacy_kernel() on the frame word itself (not the per-thread copy dust_pyr_split_word)
bool dust_p_legacy() { return ((DustPRow(uvec2(pc.pyramid)).v[DUST_PYR_SPLIT_DEPTH] >> 29u) & 1u) != 0u; }
uint dust_p_end() {
    uint vbase = DustPRow(uvec2(pc.pyramid)).v[DUST_PYR_QUEUE_BASE] + DUST_QUEUE_WORDS;
    uint nv = DustPRow(uvec2(pc.pyramid)).v[vbase + DUST_VQUEUE_RECORDS];
    nv = nv == 0u ? DUST_VQUEUE_CAPACITY : min(nv, DUST_VQUEUE_CAPACITY);
    uint sbase = vbase + DUST_VQUEUE_HEADER_WORDS + nv * DUST_VQUEUE_RECORD_WORDS;
    uint t = dust_squeue_table(sbase) + gl_WorkGroupID.x * DUST_SQUEUE_TABLE_WORDS;
    return dust_squeue_records(sbase) + DustPRow(uvec2(pc.pyramid)).v[t] + min(DustPRow(uvec2(pc.pyramid)).v[t + 1u], DUST_SQUEUE_PER_CLUSTER);
}
// dust_is_volumetric(p0, eye) on the record's words
bool dust_p_is_volumetric(vec3 eye) {
    DUST_O_PRL()
    vec3 q = eye - vec3(PN(0u), PN(1u), PN(2u));
    vec3 r = q - vec3(PN(9u), PN(10u), PN(11u)) * dust_segment_param(q, vec3(PN(9u), PN(10u), PN(11u)));
    float tr = PN(3u) + PN(6u) + PN(8u);
    return tr > DUST_VOLUME_RATIO * DUST_VOLUME_RATIO * dot(r, r);
}
// dust_support_mass(p0, sup) on the record's words (the support is the record's: has_support known)
float dust_p_support_mass() {
    DUST_O_PRL()
    float sig = sqrt(max(dust_var_along(float[6](PN(3u), PN(4u), PN(5u), PN(6u), PN(7u), PN(8u)), vec3(PN(23u), PN(24u), PN(25u))), 0.0));
    float m0 = dot(vec3(PN(23u), PN(24u), PN(25u)), vec3(PN(0u), PN(1u), PN(2u))) - PN(26u), ds = dot(vec3(PN(23u), PN(24u), PN(25u)), vec3(PN(9u), PN(10u), PN(11u)));
    float acc = 0.0, wsum = 0.0;
    for (int k = 0; k < 16; ++k) {
        float u = (float(k) + 0.5) / 16.0;
        float w = PN(15u) + (PN(16u) - PN(15u)) * u;
        float x = m0 + ds * u;
        float f = sig > 0.0 ? 0.5 * (1.0 + dust_erf(x / (sig * 1.4142135623730951))) : (x >= 0.0 ? 1.0 : 0.0);
        acc += w * f;
        wsum += w;
    }
    return wsum > 0.0 ? acc / wsum : 1.0;
}
// sn = S n and v0 = max(n.sn, 0) to the cone words 34-37
void dust_p_put_sn() {
    DUST_O_PRL()
    vec3 sup = P_N;
    PW(34u) = floatBitsToUint(PRF(3u) * sup.x + PRF(4u) * sup.y + PRF(5u) * sup.z);
    PW(35u) = floatBitsToUint(PRF(4u) * sup.x + PRF(6u) * sup.y + PRF(7u) * sup.z);
    PW(36u) = floatBitsToUint(PRF(5u) * sup.x + PRF(7u) * sup.y + PRF(8u) * sup.z);
    PW(37u) = floatBitsToUint(max(dot(P_N, P_SN), 0.0));
}
// dust_conic_piece's w (cut: cone word 39; sn, v0: 34-37)
bool dust_p_conic_piece(uint j, uint e, out float w) {
    DUST_O_PRL()
    w = 0.0;
    float k = float(DUST_SUPPORT_TRUNC_PIECES);
    float u0 = float(j) / k, u1 = float(j + 1u) / k;
    float ra = PN(15u) + (PN(16u) - PN(15u)) * u0;
    float rb = PN(15u) + (PN(16u) - PN(15u)) * u1;
    float rm = max(0.5 * (ra + rb), 1e-6);
    float qa = ra / rm, qb = rb / rm;
    float uc = qa + qb > 0.0 ? (qa + 2.0 * qb) / (3.0 * (qa + qb)) : 0.5;
    vec3 qc = (vec3(PN(0u), PN(1u), PN(2u)) + vec3(PN(9u), PN(10u), PN(11u)) * u0) + (vec3(PN(9u), PN(10u), PN(11u)) * (u1 - u0)) * uc;
    float ds = dot(vec3(PN(23u), PN(24u), PN(25u)), vec3(PN(9u), PN(10u), PN(11u)) * (u1 - u0));
    float v = PN(37u) + ds * ds / 12.0;
    float sig = sqrt(max(v, 1e-30));
#define DUST_O_W39 DustPRowN(dust_p_row).v[39u * DUST_SQUEUE_CAPACITY]
    float alpha = DUST_O_W39 != 0u ? max((PN(26u) - dot(vec3(PN(23u), PN(24u), PN(25u)), qc)) / sig, -DUST_CONIC_FAR) : -DUST_CONIC_FAR;
    uint c = 0u;
    for (uint i = 0u; i < DUST_SUPPORT_TRUNC_SLABS - 1u; ++i) if (dust_conic_slab_edge(i) > alpha) c++;
    uint ne = c + 1u;
    if (e >= ne) return false;
    uint first = DUST_SUPPORT_TRUNC_SLABS - 1u - c;
    float za = e == 0u ? alpha : dust_conic_slab_edge(first + e - 1u);
    float zb = e + 1u == ne ? DUST_CONIC_FAR : dust_conic_slab_edge(first + e);
    float qa_ = dust_nupper(za);
    float qb_ = dust_nupper(zb);
    float ps = qa_ - qb_;
    if (!(ps > 1e-9)) return false;
    w = (PN(12u) * (u1 - u0) * rm) * ps;
    return true;
}
// the bounds from the cone words 41-44
DustTruncBounds dust_p_bounds() {
    DustTruncBounds tb;
    tb.za = PRF(41u); tb.zb = PRF(42u); tb.before = PR(43u); tb.inner = PR(44u);
    return tb;
}
// dust_support_trunc_piece's w (sn, v0: cone words 34-37)
bool dust_p_trunc_piece(float u0, float u1, uint j, uint e, out float w) {
    DUST_O_PRL()
    w = 0.0;
    if (!(u1 > u0)) return false;
    float ra0 = PN(15u) + (PN(16u) - PN(15u)) * u0;
    float rb0 = PN(15u) + (PN(16u) - PN(15u)) * u1;
    float rm = max(0.5 * (ra0 + rb0), 1e-6);
    float ra = ra0 / rm, rb = rb0 / rm;
    float uc = ra + rb > 0.0 ? (ra + 2.0 * rb) / (3.0 * (ra + rb)) : 0.5;
    float ds = dot(vec3(PN(23u), PN(24u), PN(25u)), vec3(PN(9u), PN(10u), PN(11u)) * (u1 - u0));
    float m = dot(vec3(PN(23u), PN(24u), PN(25u)), (vec3(PN(0u), PN(1u), PN(2u)) + vec3(PN(9u), PN(10u), PN(11u)) * u0) + (vec3(PN(9u), PN(10u), PN(11u)) * (u1 - u0)) * uc) - PN(26u);
    float v = PN(37u) + ds * ds / 12.0;
    if (!(v > 0.0)) {
        if (e != 0u || !(m >= 0.0)) return false;
        w = PN(12u) * (u1 - u0) * rm;
        return true;
    }
    float sig = sqrt(v);
    float alpha = -m / sig;
    float pk = dust_nupper(alpha);
    if (!(pk > 1e-6)) return false;
    float lam = dust_npdf(alpha) / pk;
    float t = sqrt(max(1.0 + alpha * lam - lam * lam, 1e-4));
    float za = e == 0u ? alpha : alpha + dust_support_slab_edge(e - 1u) * t;
    bool last = e + 1u == DUST_SUPPORT_TRUNC_SLABS;
    float qa = dust_nupper(za);
    float qb = 0.0;
    if (!last) {
        float zb = alpha + dust_support_slab_edge(e) * t;
        qb = dust_nupper(zb);
    }
    float ps = qa - qb;
    if (!(ps > 1e-9)) return false;
    w = (PN(12u) * (u1 - u0) * rm) * ps;
    return true;
}
// dust_leaf_prep on record word index `rec` (recs + r): the same tests in the same order, the cone
// words written as decided (defaults first: DUST_SQUEUE_PUT_CONE's values for a plain record)
void dust_leaf_prep_p(uint rec) {
    dust_p_row = dust_p_addr(rec);
#ifdef DUST_SPLIT_PREP
    dust_p_lrs[gl_LocalInvocationID.x] = rec;
#endif
    PW(33u) = DUST_SQUEUE_KIND_PLAIN;
    PW(34u) = 0u; PW(35u) = 0u; PW(36u) = 0u; PW(37u) = 0u; PW(38u) = 0u; PW(39u) = 0u;
    PW(40u) = floatBitsToUint(1.0);
    PW(41u) = floatBitsToUint(0.0); PW(42u) = floatBitsToUint(1.0); PW(43u) = 0u; PW(44u) = 0u;
    DUST_PROF_BEGIN_AS(cone1)
    if (dust_p_persp() && !(dust_p_shell() > 0.0) && dust_has_support(vec4(PRF(23u), PRF(24u), PRF(25u), PRF(26u))) && (PR(14u) & 4u) != 0u && !dust_p_is_volumetric(pc.eyeLocal.xyz) && !dust_p_legacy()) {
        float ra = PRF(15u), rb = PRF(16u);
        float ucp = ra + rb > 0.0 ? (ra + 2.0 * rb) / (3.0 * (ra + rb)) : 0.5;
        float cDC = dot(P_N, P_MEAN + P_SEG * ucp) - PRF(27u);
        PW(38u) = floatBitsToUint(cDC);
        if (cDC > 0.0) {
            dust_p_put_sn();
            PW(39u) = dust_p_support_mass() < 1.0 - DUST_SUPPORT_CUT_MIN ? 1u : 0u;
            float total = 0.0;
            uint cnt = 0u;
            for (uint it = 0u; it < DUST_SUPPORT_TRUNC_PIECES * DUST_SUPPORT_TRUNC_SLABS; ++it) {
                float w;
                if (dust_p_conic_piece(it / DUST_SUPPORT_TRUNC_SLABS, it % DUST_SUPPORT_TRUNC_SLABS, w)) {
                    total += w;
                    cnt++;
                }
            }
            if (cnt > 0u) {
                PW(33u) = DUST_SQUEUE_KIND_CONIC;
                PW(40u) = floatBitsToUint(PRF(12u) > 0.0 ? 1.0 / max(total / PRF(12u), DUST_SUPPORT_MASS_MIN) : 0.0);
            }
        }
    }
    if (PR(33u) != DUST_SQUEUE_KIND_CONIC && dust_p_persp() && !(dust_p_shell() > 0.0) && dust_has_support(vec4(PRF(23u), PRF(24u), PRF(25u), PRF(26u)))
        && dust_p_support_mass() < 1.0 - DUST_SUPPORT_CUT_MIN) {
        if (dust_p_is_volumetric(pc.eyeLocal.xyz)) {
            PW(33u) = DUST_SQUEUE_KIND_VOLUME;
        } else {
            PW(33u) = DUST_SQUEUE_KIND_TRUNC;
            dust_p_put_sn();
            // dust_support_trunc_bounds(p0, sup, v0) to the cone words 41-44
            DUST_O_PRL()
            float dn = dot(P_N, P_SEG);
            if (abs(dn) > 0.0) {
                float half_ = 3.0 * sqrt(PRF(37u)) / abs(dn);
                float ux = (PRF(26u) - dot(P_N, P_MEAN)) / dn;
                float za = max(ux - half_, 0.0), zb = min(ux + half_, 1.0);
                if (zb > za && (za > 0.0 || zb < 1.0)) {
                    uint before = za > 0.0 ? 1u : 0u, after = zb < 1.0 ? 1u : 0u;
                    PW(41u) = floatBitsToUint(za); PW(42u) = floatBitsToUint(zb); PW(43u) = before;
                    PW(44u) = DUST_SUPPORT_TRUNC_PIECES - before - after;
                }
            }
            float total = 0.0;
            for (uint it = 0u; it < DUST_SUPPORT_TRUNC_PIECES * DUST_SUPPORT_TRUNC_SLABS; ++it) {
                uint j = it / DUST_SUPPORT_TRUNC_SLABS, e = it % DUST_SUPPORT_TRUNC_SLABS;
                float w;
                if (dust_p_trunc_piece(dust_trunc_bound(dust_p_bounds(), j), dust_trunc_bound(dust_p_bounds(), j + 1u), j, e, w)) total += w;
            }
            PW(40u) = floatBitsToUint(PRF(12u) > 0.0 ? 1.0 / max(total / PRF(12u), DUST_SUPPORT_MASS_MIN) : 0.0);
        }
    }
    DUST_PROF_END_AS(cone, cone1)
    DUST_PROF_COUNT(cone, (PR(33u) == DUST_SQUEUE_KIND_TRUNC || PR(33u) == DUST_SQUEUE_KIND_CONIC) ? 1u : 0u)
}

#ifndef DUST_SPLIT_PREP
#ifdef DUST_SPLIT_SEQ

// A volumetric packet of this cluster: its rounding seed drawn first (in the thread's packet
// order, so every seed is what it was: dust::volume_scatter), then emitted with it to the chunk's
// volume queue (dust_vqueue_append) or, when that refuses it (full, or off: DUST_FRAME_VQUEUE_OFF),
// to the volume spill (dust_vspill_base); dust_draw_volume.comp draws both, so the pyramid is the
// same whichever queue held it. Never drawn here. A record the spill refuses too is dropped: the
// spill drain counts it (DUST_VQUEUE_LOST, the spill's raw appends past its capacity) and the host
// logs it. (levels, W, H, bias, axis: the consumer reads them from the frame words.)
void dust_volume_emit(DustPacket p, float exposure, vec3 color, uint levels, float W, float H, uint bias, uint axis) {
    if (dust_counting) { dust_seed_count++; return; }
    DUST_PROF_COUNT(volume, 1u)
    uint seed = dust_next_packet_seed();
    if (dust_probe() == 3u) return;
    dust_phot_sun = vec3(uintBitsToFloat(pc.pyramid.v[dust_leaf_rec + 28u * DUST_SQUEUE_CAPACITY]), uintBitsToFloat(pc.pyramid.v[dust_leaf_rec + 29u * DUST_SQUEUE_CAPACITY]), uintBitsToFloat(pc.pyramid.v[dust_leaf_rec + 30u * DUST_SQUEUE_CAPACITY]));
    dust_phot_rh = uintBitsToFloat(pc.pyramid.v[dust_leaf_rec + 31u * DUST_SQUEUE_CAPACITY]);
    dust_phot_support = dust_squeue_kind_o(uvec2(pc.pyramid), dust_leaf_rec, 0u) != DUST_SQUEUE_KIND_PLAIN ? vec4(0.0) : dust_squeue_support(pc.pyramid, dust_leaf_rec, 0u);
    DUST_PROF_BEGIN(vemit)
    if (dust_vqueue_on) {
        uint vbase = dust_vqueue_base_o(uvec2(pc.pyramid));
        uint r = dust_vqueue_append_o(uvec2(pc.pyramid), vbase);
        if (r != 0xFFFFFFFFu) {
            uint cap = dust_vqueue_capacity_o(uvec2(pc.pyramid), vbase);
            DUST_VQUEUE_PUT(pc.pyramid, vbase, cap, r, p, seed, exposure, color, dust_phot_support, dust_phot_sun, dust_phot_rh)
            DUST_PROF_END(vemit)
            return;
        }
    }
    uint sbase = dust_vspill_base_o(uvec2(pc.pyramid));
    uint rs = dust_vqueue_append_o(uvec2(pc.pyramid), sbase);
    if (rs != 0xFFFFFFFFu) {
        uint scap = dust_vqueue_capacity_o(uvec2(pc.pyramid), sbase);
        DUST_VQUEUE_PUT(pc.pyramid, sbase, scap, rs, p, seed, exposure, color, dust_phot_support, dust_phot_sun, dust_phot_rh)
    }
    DUST_PROF_END(vemit)
}


// dust_volume_emit for a candidate slot: the seed first, then the volume record
// written word by word: each word loaded from the candidate slot or the sub-packet record
// and stored at once (DUST_VQUEUE_PUT of DUST_CQUEUE_GET's packet: the packet's fields are bit
// copies, segT the get's 0; the photometry words the record's, as this function set
// dust_phot_sun / _rh / _support, which nothing else in the sequencer reads). No packet held.
layout(buffer_reference, std430, buffer_reference_align = 4) buffer DustSRow { uint v[]; };
uvec2 dust_s_slot;
#if defined(DUST_SEQ_DRAW)
#define DUST_O_SQ(k) pc.pyramid.v[q_w + (k)]
#else
#define DUST_O_SQ(k) DustSRow(q_).v[k]
#endif
uvec2 dust_s_addr(uint word) { uvec2 a = uvec2(pc.pyramid); uint o = word * 4u; uint lo = a.x + o; return uvec2(lo, a.y + (lo < o ? 1u : 0u)); }
void dust_vqueue_put_slot(uint base, uint cap, uint r, uint cslot, uint seed) {
    // one loop over the volume record's 32 words in word order, each word's source picked
    // by its index: the candidate slot and the sub-packet record each through one pointer, the
    // destination word's address formed per iteration (one address live, not one per word)
#if defined(DUST_SEQ_DRAW)
    uint q_w = dust_cqueue_base_o(uvec2(pc.pyramid)) + DUST_VQUEUE_HEADER_WORDS + cslot * DUST_CQUEUE_RECORD_WORDS;
#endif
    uvec2 q_ = dust_s_addr(dust_cqueue_base_o(uvec2(pc.pyramid)) + DUST_VQUEUE_HEADER_WORDS + cslot * DUST_CQUEUE_RECORD_WORDS);
    uvec2 e_ = dust_s_addr(dust_leaf_rec);
#if defined(DUST_SEQ_DRAW)
#define RW(k) pc.pyramid.v[dust_leaf_rec + (k) * DUST_SQUEUE_CAPACITY]
#else
#define RW(k) DustSRow(e_).v[(k) * DUST_SQUEUE_CAPACITY]
#endif
    [[dont_unroll]]
    for (uint k = 0u; k < 32u; ++k) {
        uint w;
        if (k < 15u) w = DUST_O_SQ(k);
        else if (k < 18u) w = floatBitsToUint(0.0);
        else if (k == 18u) w = seed;
        else if (k == 19u) w = RW(13u);
        else if (k < 23u) w = RW(k);
        else if (k < 27u) w = RW(33u) == DUST_SQUEUE_KIND_PLAIN ? RW(k) : floatBitsToUint(0.0);
        else if (k < 31u) w = RW(k + 1u);
        else { uint f_ = DUST_O_SQ(15u); w = ((f_ & 1u) != 0u ? 1u : 0u) | ((f_ & 2u) != 0u ? 2u : 0u); }
#if defined(DUST_SEQ_DRAW)
        pc.pyramid.v[base + DUST_VQUEUE_HEADER_WORDS + r + k * cap] = w;
#else
        DustSRow(dust_s_addr(base + DUST_VQUEUE_HEADER_WORDS + r + k * cap)).v[0] = w;
#endif
    }
#undef RW
}
void dust_volume_emit_slot(uint cslot) {
    if (dust_counting) { dust_seed_count++; return; }
    DUST_PROF_COUNT(volume, 1u)
    uint seed = dust_next_packet_seed();
    dust_pyr_split_word = pc.pyramid.v[DUST_PYR_SPLIT_DEPTH];
    if (dust_probe() == 3u) return;
    DUST_PROF_BEGIN(vemit)
    dust_vqueue_on = (pc.pyramid.v[DUST_PYR_FLAGS] & DUST_FRAME_VQUEUE_OFF) == 0u;
    if (dust_vqueue_on) {
        uint vbase = dust_vqueue_base_o(uvec2(pc.pyramid));
        uint r = dust_vqueue_append_o(uvec2(pc.pyramid), vbase);
        if (r != 0xFFFFFFFFu) {
            dust_vqueue_put_slot(vbase, dust_vqueue_capacity_o(uvec2(pc.pyramid), vbase), r, cslot, seed);
            DUST_PROF_END(vemit)
            return;
        }
    }
    uint sbase = dust_vspill_base_o(uvec2(pc.pyramid));
    uint rs = dust_vqueue_append_o(uvec2(pc.pyramid), sbase);
    if (rs != 0xFFFFFFFFu) dust_vqueue_put_slot(sbase, dust_vqueue_capacity_o(uvec2(pc.pyramid), sbase), rs, cslot, seed);
    DUST_PROF_END(vemit)
}

// the capsule band's texels at a level (dust::splat_scatter)
float dust_band_at(float segLen, float ex, float ey, uint level) {
    float s = float(1u << level);
    return (segLen / s + 2.0 * ex / s + 2.0) * (2.0 * ey / s + 2.0);
}

// v px^2 added to both axes of the screen covariance, the integral kept (dust::widen_screen_packet;
// v = s^2/4 for texels of s px: their centres sum to the integral within 1.4 %)
void dust_widen_screen_packet(inout DustScreenPacket P, float v) {
    P.cov.x += v;
    P.cov.z += v;
    precise float half_ = 0.5 * (P.cov.x + P.cov.z);
    precise float dd = sqrt((0.5 * (P.cov.x - P.cov.z)) * (0.5 * (P.cov.x - P.cov.z)) + P.cov.y * P.cov.y);
    float lamMax = max(half_ + dd, DUST_PIXEL_FILTER_VAR);
    precise float detRaw = P.cov.x * P.cov.z - P.cov.y * P.cov.y;
    float det = max(max(detRaw, DUST_DET_ANISO_FLOOR * lamMax * lamMax), 1e-30);
    P.amp *= sqrt(P.det / det);
    P.det = det;
    P.sigmaMin = sqrt(max(det / lamMax, DUST_PIXEL_FILTER_VAR * 0.5));
}

// (the leaf packet q rides along for the leaf record)
// (the leaf's screen packet is in its candidate slot, dust_project_leaf.comp's;
// every statement reads the fields it needs from there — dust_pq_get, the rest dead loads — so no
// packet is live across the band; the widening and the closing of short open ends are written back
// for the consumer, which draws the slot as it is)
// the candidate's screen packet slot through one per-lane pointer (consecutive words at
// constant offsets: one address pair, not one per word), as dust_pq_get reads it
#define SQF(k) uintBitsToFloat(DUST_O_SL(k))
#if defined(DUST_SEQ_DRAW)
uint dust_s_w;
#define DUST_O_SL(k) pc.pyramid.v[dust_s_w + (k)]
#else
#define DUST_O_SL(k) DustSRow(dust_s_slot).v[k]
#endif
#if defined(DUST_SEQ_DRAW)
#define DUST_O_SS(k, v_) pc.pyramid.v[dust_s_w + (k)] = (v_)
#else
#define DUST_O_SS(k, v_) DustSRow(dust_s_slot).v[k] = (v_)
#endif
DustScreenPacket dust_s_get() {
    DustScreenPacket P;
    P.mu = vec2(SQF(0u), SQF(1u));
    P.cov = vec3(SQF(2u), SQF(3u), SQF(4u));
    P.seg = vec2(SQF(5u), SQF(6u));
    P.amp = SQF(7u);
    P.sigmaMin = SQF(8u);
    P.det = SQF(9u);
    P.depthAu = SQF(10u);
    uint open = DUST_O_SL(11u);
    P.openStart = (open & 1u) != 0u;
    P.openEnd = (open & 2u) != 0u;
    P.rho = vec2(SQF(12u), SQF(13u));
    P.mean = vec3(SQF(14u), SQF(15u), SQF(16u));
    P.kx = vec3(SQF(17u), SQF(18u), SQF(19u));
    P.ky = vec3(SQF(20u), SQF(21u), SQF(22u));
    P.sun = vec3(SQF(23u), SQF(24u), SQF(25u));
    P.rhAu = SQF(26u);
    P.eye = vec4(SQF(27u), SQF(28u), SQF(29u), SQF(30u));
    P.albedo = SQF(31u);
    P.wMean = SQF(32u);
    return P;
}
void scatter(uint cslot, vec3 color, uint levels) {
    DUST_PROF_BEGIN(scprep)
    // the slot through one pointer to its word 0, the fields at constant offsets
    dust_s_slot = dust_s_addr(dust_pqueue_base_o(uvec2(pc.pyramid)) + cslot * DUST_PQUEUE_RECORD_WORDS);
#if defined(DUST_SEQ_DRAW)
    dust_s_w = dust_pqueue_base_o(uvec2(pc.pyramid)) + cslot * DUST_PQUEUE_RECORD_WORDS;
#endif
    // the peak is measured whatever the unit (a view fainter than the unit must still move it:
    // not_flow.rdc), before any texel test
    if (!dust_counting) atomicMax(pc.pyramid.v[DUST_PYR_TAU_MAX], floatBitsToUint(dust_s_get().amp * dust_s_get().wMean));
    uint level = dust_splat_level(dust_s_get().sigmaMin, levels);
    float ex = DUST_SPLAT_SIGMAS * sqrt(dust_s_get().cov.x), ey = DUST_SPLAT_SIGMAS * sqrt(dust_s_get().cov.z);
    // the part of the segment the band can draw: inside the frame expanded by the band's margin
    // (dust::splat_scatter: the rows and columns clip the drawn texels; the estimate must too)
    float W = float(pc.pyramid.v[DUST_PYR_WIDTH]), H = float(pc.pyramid.v[DUST_PYR_HEIGHT]);
    float segLen;
    {
        DustScreenPacket P = dust_s_get();
        segLen = sqrt(P.seg.x * P.seg.x + P.seg.y * P.seg.y)
               * dust_visible_fraction(P.mu, P.seg, ex, ey, W, H);
    }
    // safety valve against runaway threads: a packet whose capsule band would visit more than
    // its cap goes to a coarser level (blur, never a stall). The cap is DUST_SPLAT_MAX_TEXELS
    // quartered per level of the frame's EWA bias (bits 18-21 of DUST_PYR_SPLIT_DEPTH, host-fed
    // from DUST_PYR_EWA_TEXELS: inside the coma ~10^6 packets cross the whole screen)
    uint ewaBias = min((pc.pyramid.v[DUST_PYR_SPLIT_DEPTH] >> 18u) & 0xFu, DUST_EWA_BIAS_MAX);
    float cap = DUST_SPLAT_MAX_TEXELS / float(1u << (2u * ewaBias));
    float band = dust_band_at(segLen, ex, ey, level);
    while (band > DUST_SPLAT_MAX_TEXELS && level + 1u < levels) {
        level++;
        band = dust_band_at(segLen, ex, ey, level);
    }
    // the level the per-packet valve alone gives: beyond it (the bias, the gate) the packet is
    // widened to half its texels below, so it blurs instead of missing every texel centre
    uint levelValve = level;
    // the bias stops at the gate level (<= 16 texels a packet there; the 1 x 1 level loses a
    // widened packet's light off its single texel; dust::splat_scatter)
    while (band > cap && level + 1u < levels && level < dust_gate_level(levels)) {
        level++;
        band = dust_band_at(segLen, ex, ey, level);
    }
    // the band's texels inside the level (the rows and columns clip to it): the work counted, and
    // the in-frame gate (dust::splat_scatter): past DUST_FRAME_HARD_FACTOR budgets this frame, at
    // least dust_gate_level (a frame the host feedback has not caught up with is bounded at once)
    uint lw0 = pc.pyramid.v[DUST_PYR_TABLE + 3u * level + 1u];
    uint lh0 = pc.pyramid.v[DUST_PYR_TABLE + 3u * level + 2u];
    float work = min(band, float(lw0) * float(lh0));
    // (the counting pass neither counts the work nor sees the gate: a frame the gate fires in is
    // order-dependent anyway, dust::splat_scatter)
    uint before = dust_counting ? 0u : atomicAdd(pc.pyramid.v[DUST_PYR_EWA_TEXELS], uint(work / DUST_EWA_TEXEL_UNIT) + 1u);
    if (!dust_counting && dust_frame_gates_on() && before > DUST_FRAME_HARD_FACTOR * clamp(pc.pyramid.v[DUST_PYR_EWA_GATE_SCALE], 1u, DUST_EWA_GATE_SCALE_MAX) * (DUST_EWA_FRAME_BUDGET / uint(DUST_EWA_TEXEL_UNIT))) level = max(level, dust_gate_level(levels));
    // (the counting pass widens and closes nothing: the drawing pass writes the slot once, and the
    // band test below reads the same fields either way — the widening reads them from the slot)
    if (level > levelValve) {
        float sz = float(1u << level);
        DustScreenPacket P = dust_s_get();
        dust_widen_screen_packet(P, 0.25 * sz * sz);
        if (!dust_counting) { DUST_O_SS(2u, floatBitsToUint(P.cov.x)); DUST_O_SS(4u, floatBitsToUint(P.cov.z)); DUST_O_SS(7u, floatBitsToUint(P.amp)); DUST_O_SS(8u, floatBitsToUint(P.sigmaMin)); DUST_O_SS(9u, floatBitsToUint(P.det)); }
    }
    uint h = pc.pyramid.v[DUST_PYR_TABLE + 3u * level + 2u];
    float s = float(1u << level);
    // a segment shorter than two texels at this level may miss every texel centre with open
    // ends: drawn closed (Gaussian caps), so its flux lands on the texels around it
    bool closed = dot(dust_s_get().seg, dust_s_get().seg) < 4.0 * s * s;
    if (closed && !dust_counting) DUST_O_SS(11u, DUST_O_SL(11u) & ~3u);
    DustBand B;
    {
        DustScreenPacket P = dust_s_get();
        // (the counting pass's packet closed the same way, unwritten)
        if (closed) { P.openStart = false; P.openEnd = false; }
        B = dust_band_new(P);
    }
    int y0, y1;
    if (!dust_band_rows(B, s, h, y0, y1)) { DUST_PROF_END(scprep) return; }
    if (dust_counting) { dust_seed_count++; DUST_PROF_END(scprep) return; }
    uint seed = dust_next_packet_seed();
    // the splitter draws no texel. The leaf, found on its band, goes to the leaf
    // queue with its seed and levels (dust_scatter.glsl DUST_LQUEUE_*; a large one, its band over
    // DUST_QUEUE_MIN_TEXELS texels, listed for dust_draw_leaf.comp's workgroup blob, a small one
    // drawn by its lane blob). DUST_FRAME_QUEUE_ALL: every leaf large; DUST_FRAME_QUEUE_OFF: every
    // leaf small (the test hooks: the same pyramid either way)
    uint qflags = pc.pyramid.v[DUST_PYR_FLAGS];
    float bandAt = dust_band_at(segLen, ex, ey, level);
    bool large = (qflags & DUST_FRAME_QUEUE_OFF) == 0u
        && ((qflags & DUST_FRAME_QUEUE_ALL) != 0u || bandAt > DUST_QUEUE_MIN_TEXELS);
    DUST_PROF_BEGIN(qput)
    DUST_PROF_COUNT(records, 1u)
    dust_leaf_emit_o(uvec2(pc.pyramid), cslot, dust_leaf_rec, seed, level, levelValve, large);
    DUST_PROF_END(qput)
    DUST_PROF_END(scprep)
    // (the level mask: the consumer sets it for the texels it writes)
}

// ── The sequencer stage (dust_split.comp -DDUST_SPLIT_SEQ) ───────────────────────────────────
// Record `recs + r`'s run of candidates (the walk stage's, DUST_CQUEUE_*) in its walk order
// through the leaf path as the walk drew it: a volumetric packet to dust_volume_emit, an EWA leaf
// projected (dust_project_packet) and, on screen, through scatter — the same calls with the same
// arguments in the same order, so the seeds (counted, prefixed and drawn here) and every atomic
// are the per-cluster thread's
float dust_seq_exposure() { return uintBitsToFloat(pc.pyramid.v[dust_leaf_rec + 13u * DUST_SQUEUE_CAPACITY]); }
vec3 dust_seq_color() { return vec3(uintBitsToFloat(pc.pyramid.v[dust_leaf_rec + 20u * DUST_SQUEUE_CAPACITY]), uintBitsToFloat(pc.pyramid.v[dust_leaf_rec + 21u * DUST_SQUEUE_CAPACITY]), uintBitsToFloat(pc.pyramid.v[dust_leaf_rec + 22u * DUST_SQUEUE_CAPACITY])); }
// ── The sequencer in two stages ──────────────────────────────────────────────────────────────────
// The order stage (dust_split.comp -DDUST_SPLIT_SEQ, a workgroup per cluster as before) decides for
// every candidate of the cluster's run whether it draws a seed (the counting pass: dust_s_count) and
// writes each candidate's seed ordinal — the cluster-wide exclusive prefix of those counts in
// candidate order, which is record order and the walk's order inside a record — into the candidate's
// word 15 above its three flag bits. The draw stage (-DDUST_SPLIT_SEQ -DDUST_SEQ_DRAW, a lane per
// candidate, indirect from the candidate buffer's header word 0 as dust_project_leaf.comp) runs the
// leaf path of one candidate straight through from that ordinal: no loop around the leaf path, which
// is what kept its values in registers across every candidate (Mali-G52: 32 registers straight, 50
// around a loop of the same body).

// the counting pass of `scatter` for a screen candidate (the seeds it draws: 0 or 1). The
// counting pass of today's leaf path (dust::splat_scatter / the tree's scatter): the level, the
// valve, no work counted and no gate, the packet widened and closed in registers, its band's rows.
// (The same statements as scatter's in the same order; the widening on the packet as the tree's
// counting pass did — the slot is not written by the counting pass, so its band is the widened
// packet's, not the slot's.)
uint dust_s_count_screen(uint cslot, uint levels) {
    dust_s_slot = dust_s_addr(dust_pqueue_base_o(uvec2(pc.pyramid)) + cslot * DUST_PQUEUE_RECORD_WORDS);
#if defined(DUST_SEQ_DRAW)
    dust_s_w = dust_pqueue_base_o(uvec2(pc.pyramid)) + cslot * DUST_PQUEUE_RECORD_WORDS;
#endif
    uint level = dust_splat_level(dust_s_get().sigmaMin, levels);
    float ex = DUST_SPLAT_SIGMAS * sqrt(dust_s_get().cov.x), ey = DUST_SPLAT_SIGMAS * sqrt(dust_s_get().cov.z);
    float W = float(pc.pyramid.v[DUST_PYR_WIDTH]), H = float(pc.pyramid.v[DUST_PYR_HEIGHT]);
    float segLen;
    {
        DustScreenPacket P = dust_s_get();
        segLen = sqrt(P.seg.x * P.seg.x + P.seg.y * P.seg.y)
               * dust_visible_fraction(P.mu, P.seg, ex, ey, W, H);
    }
    uint ewaBias = min((pc.pyramid.v[DUST_PYR_SPLIT_DEPTH] >> 18u) & 0xFu, DUST_EWA_BIAS_MAX);
    float cap = DUST_SPLAT_MAX_TEXELS / float(1u << (2u * ewaBias));
    float band = dust_band_at(segLen, ex, ey, level);
    while (band > DUST_SPLAT_MAX_TEXELS && level + 1u < levels) {
        level++;
        band = dust_band_at(segLen, ex, ey, level);
    }
    uint levelValve = level;
    while (band > cap && level + 1u < levels && level < dust_gate_level(levels)) {
        level++;
        band = dust_band_at(segLen, ex, ey, level);
    }
    DustScreenPacket P = dust_s_get();
    if (level > levelValve) {
        float sz = float(1u << level);
        dust_widen_screen_packet(P, 0.25 * sz * sz);
    }
    uint h = pc.pyramid.v[DUST_PYR_TABLE + 3u * level + 2u];
    float s = float(1u << level);
    if (dot(P.seg, P.seg) < 4.0 * s * s) { P.openStart = false; P.openEnd = false; }
    DustBand B = dust_band_new(P);
    int y0, y1;
    return dust_band_rows(B, s, h, y0, y1) ? 1u : 0u;
}
// the seeds candidate `c` draws (dust_volume_emit_slot's counting pass: 1; a screen candidate off
// screen: 0)
uint dust_s_count(uint c) {
    uint cbase = dust_cqueue_base_o(uvec2(pc.pyramid));
    if ((pc.pyramid.v[cbase + DUST_VQUEUE_HEADER_WORDS + c * DUST_CQUEUE_RECORD_WORDS + 15u] & 4u) != 0u) return 1u;
    if (!dust_pq_on_o(uvec2(pc.pyramid), dust_pqueue_base_o(uvec2(pc.pyramid)), dust_cqueue_capacity_o(uvec2(pc.pyramid), cbase), c)) return 0u;
    return dust_s_count_screen(c, pc.pyramid.v[DUST_PYR_LEVELS]);
}
// the draw stage's candidate `c`: the leaf path from its ordinal (word 15 >> 3) and its record
// (word 16); exactly the candidates of the records' runs [cb, cb + cn) (the slots the walk wrote)
void dust_s_draw(uint c) {
    uint cbase = dust_cqueue_base_o(uvec2(pc.pyramid));
#if defined(DUST_SEQ_DRAW)
    uint q_w = cbase + DUST_VQUEUE_HEADER_WORDS + c * DUST_CQUEUE_RECORD_WORDS;
#endif
    uvec2 q_ = dust_s_addr(cbase + DUST_VQUEUE_HEADER_WORDS + c * DUST_CQUEUE_RECORD_WORDS);
    dust_leaf_rec = DUST_O_SQ(16u);
    if (c - pc.pyramid.v[dust_leaf_rec + DUST_SQUEUE_CAND_BASE * DUST_SQUEUE_CAPACITY] >= pc.pyramid.v[dust_leaf_rec + DUST_SQUEUE_CAND_COUNT * DUST_SQUEUE_CAPACITY]) return;
    dust_counting = false;
    dust_cluster_seed = dust_hash32(pc.pyramid.v[dust_leaf_rec + 32u * DUST_SQUEUE_CAPACITY]);
    dust_packet_ord = DUST_O_SQ(15u) >> 3u;
    DUST_PROF_COUNT(packets, 1u)
    if ((DUST_O_SQ(15u) & 4u) != 0u) {
        dust_volume_emit_slot(c);
    } else if (dust_pq_on_o(uvec2(pc.pyramid), dust_pqueue_base_o(uvec2(pc.pyramid)), dust_cqueue_capacity_o(uvec2(pc.pyramid), cbase), c)) {
        DUST_PROF_COUNT(leaves, 1u)
        dust_pyr_split_word = pc.pyramid.v[DUST_PYR_SPLIT_DEPTH];
        scatter(c, dust_seq_color(), pc.pyramid.v[DUST_PYR_LEVELS]);
    }
}

#else  // the walk stage: no projection, no seed, no drawing — candidates

// ── The walk's node in the record (not in shared memory) ──────────────────────────────────────
// Each lane's node q as 20 words (mean, S, seg, flux, openStart, openEnd, rho, segT): the walking
// record's words 48-67 (WQ(k) = record word 48 + k, structure-of-arrays at DUST_SQUEUE_CAPACITY),
// written by the piece load and by the step, read word by word at each use through the record's
// pointer, so nothing of q stays live between its uses (the same expressions as
// dust_split_step_one / dust_sub_chord, on the row)
layout(buffer_reference, std430, buffer_reference_align = 4) volatile buffer DustWRow { uint v[]; };
uvec2 dust_w_row;
uvec2 dust_w_addr(uint word) { uvec2 a = uvec2(pc.pyramid); uint o = word * 4u; uint lo = a.x + o; return uvec2(lo, a.y + (lo < o ? 1u : 0u)); }
// the record's words through one per-lane pointer to its word 0 and constant indices k * capacity:
// one address register pair, the stride an immediate / scalar offset (GCN soffset), never a 64-bit
// address per word
shared uint dust_w_lri[64];
#define DUST_O_WRL() { memoryBarrierShared(); dust_w_row = dust_w_addr(dust_w_lri[gl_LocalInvocationID.x]); }
#ifndef DUST_O_WRL
#define DUST_O_WRL()
#endif
#define DUST_REC(k) DustWRow(dust_w_row).v[(k) * DUST_SQUEUE_CAPACITY]
#define WQ(k) DUST_REC(48u + (k))
#define WQF(k) uintBitsToFloat(WQ(k))
#define DUST_O_WREC ((dust_w_row.x - uvec2(pc.pyramid).x) >> 2u)
#define DUST_O_CP(k) DustWRow(cp).v[k]
#define WQ_W(k) WQ(k)
#define REC_W(k) DUST_REC(k)
// the row's words read volatile: a reload the compiler neither merges with an earlier
// load of the word nor hoists; where a word's value would otherwise be kept from its first use to a
// later one, the later use reads it again (the same bits)
layout(buffer_reference, std430, buffer_reference_align = 4) volatile buffer DustWState { uint v[]; };
#define WQV(k) DustWState(dust_w_row).v[(48u + (k)) * DUST_SQUEUE_CAPACITY]
#define WQVF(k) uintBitsToFloat(WQV(k))
void dust_w_store(DustPacket q) {
    WQ_W(0u) = floatBitsToUint(q.mean.x); WQ_W(1u) = floatBitsToUint(q.mean.y); WQ_W(2u) = floatBitsToUint(q.mean.z);
    WQ_W(3u) = floatBitsToUint(q.S[0]); WQ_W(4u) = floatBitsToUint(q.S[1]); WQ_W(5u) = floatBitsToUint(q.S[2]);
    WQ_W(6u) = floatBitsToUint(q.S[3]); WQ_W(7u) = floatBitsToUint(q.S[4]); WQ_W(8u) = floatBitsToUint(q.S[5]);
    WQ_W(9u) = floatBitsToUint(q.seg.x); WQ_W(10u) = floatBitsToUint(q.seg.y); WQ_W(11u) = floatBitsToUint(q.seg.z);
    WQ_W(12u) = floatBitsToUint(q.flux);
    WQ_W(13u) = q.openStart ? 1u : 0u; WQ_W(14u) = q.openEnd ? 1u : 0u;
    WQ_W(15u) = floatBitsToUint(q.rho.x); WQ_W(16u) = floatBitsToUint(q.rho.y);
    WQ_W(17u) = floatBitsToUint(q.segT.x); WQ_W(18u) = floatBitsToUint(q.segT.y); WQ_W(19u) = floatBitsToUint(q.segT.z);
}
// a block boundary inside a pass: a branch on a fresh (volatile) read of the
// walk's state word that always holds (its counting and candidate bits are never both set), so
// the code after it is a block of its own, never merged with the one before: the Adreno
// compiler's per-block word-offset constants are then the words each half touches
#define DUST_W_CUT if ((DustWState(dust_w_row).v[47u * DUST_SQUEUE_CAPACITY] >> 30u) != 3u)

// the row as a packet with only the words its reader uses loaded (the row's
// words are volatile: v/G's dust_w_load read all 20 at every use); the other fields 0, never read by
// these readers: S alone (dust_var_along), mean S seg (dust_shell_classify,
// dust_shell_outside_fraction)
DustPacket dust_w_load_s() {
    DustPacket q;
    q.mean = vec3(0.0);
    q.S[0] = WQF(3u); q.S[1] = WQF(4u); q.S[2] = WQF(5u); q.S[3] = WQF(6u); q.S[4] = WQF(7u); q.S[5] = WQF(8u);
    q.seg = vec3(0.0); q.flux = 0.0; q.openStart = false; q.openEnd = false; q.rho = vec2(0.0); q.segT = vec3(0.0);
    return q;
}
DustPacket dust_w_load_mss() {
    DustPacket q;
    q.mean = vec3(WQF(0u), WQF(1u), WQF(2u));
    q.S[0] = WQF(3u); q.S[1] = WQF(4u); q.S[2] = WQF(5u); q.S[3] = WQF(6u); q.S[4] = WQF(7u); q.S[5] = WQF(8u);
    q.seg = vec3(WQF(9u), WQF(10u), WQF(11u));
    q.flux = 0.0; q.openStart = false; q.openEnd = false; q.rho = vec2(0.0); q.segT = vec3(0.0);
    return q;
}
// dust_sub_chord on the row (the row is p and becomes q; S and segT unchanged)
void dust_sub_chord_w(float u0, float u1, bool openStart, bool openEnd) {
    DUST_O_WRL()
    // (each word read once, volatile: the step's earlier loads of it are not kept to here)
    float r0 = WQVF(15u), r1 = WQVF(16u);
    float ra = r0 + (r1 - r0) * u0;
    float rb = r0 + (r1 - r0) * u1;
    float rm = max(0.5 * (ra + rb), 1e-6);
    vec3 seg = vec3(WQVF(9u), WQVF(10u), WQVF(11u));
    vec3 mean_ = vec3(WQVF(0u), WQVF(1u), WQVF(2u)) + seg * u0;
    WQ_W(0u) = floatBitsToUint(mean_.x); WQ_W(1u) = floatBitsToUint(mean_.y); WQ_W(2u) = floatBitsToUint(mean_.z);
    vec3 seg_ = seg * (u1 - u0);
    WQ_W(9u) = floatBitsToUint(seg_.x); WQ_W(10u) = floatBitsToUint(seg_.y); WQ_W(11u) = floatBitsToUint(seg_.z);
    WQ_W(12u) = floatBitsToUint(WQVF(12u) * (u1 - u0) * rm);
    WQ_W(13u) = openStart ? 1u : 0u;
    WQ_W(14u) = openEnd ? 1u : 0u;
    WQ_W(15u) = floatBitsToUint(ra / rm); WQ_W(16u) = floatBitsToUint(rb / rm);
}
// dust_is_volumetric on the row, its words read volatile (a fresh read of each, only those it uses)
// the trace's words read in a block of their own (DUST_W_CUT), after dot(r, r)
bool dust_w_is_volumetric(vec3 eye) {
    vec3 seg = vec3(WQVF(9u), WQVF(10u), WQVF(11u));
    vec3 q = eye - vec3(WQVF(0u), WQVF(1u), WQVF(2u));
    vec3 r = q - seg * dust_segment_param(q, seg);
    float rr = dot(r, r);
    bool vol = false;
    DUST_W_CUT {
        float tr = WQVF(3u) + WQVF(6u) + WQVF(8u);
        vol = tr > DUST_VOLUME_RATIO * DUST_VOLUME_RATIO * rr;
    }
    return vol;
}
// dust_split_step_one on the row (generated from it: p.* read from and written to the row)
#ifdef DUST_W_VIEW_SHARED
// the frame's view direction (dust_view_forward, xyz) and its perspective flag (w: 1 or 0), found
// once per workgroup by the walk stage's main: read where it is used, never a value the
// compiler keeps (or re-derives from the matrix, with its temporaries) over the walk
// (w: bit 0 the perspective flag, bit 1 the frame's deep-chord bit, dust_deep_chord, as a float)
shared vec4 dust_w_view;
vec3 dust_w_fwd() { return dust_w_view.xyz; }
bool dust_w_persp() { return (uint(dust_w_view.w) & 1u) != 0u; }
#define DUST_W_DEEP_CHORD ((uint(dust_w_view.w) & 2u) != 0u)
#else
vec3 dust_w_fwd() { bool p_; return dust_view_forward(pc.mvp, pc.eyeLocal.xyz, p_); }
#define DUST_W_DEEP_CHORD dust_deep_chord
#endif
// the Gaussian split's update (dust_split_step_one's, after the step returned DUST_SPLIT_GAUSS):
// sigmaV, k and h recomputed from the row's S, which the step left unchanged (the same expressions)
const int DUST_SPLIT_GAUSS = 5;
void dust_w_gauss(bool second) {
    DUST_O_WRL()
    float sigmaV = sqrt(max(dust_var_along(dust_w_load_s().S, dust_w_fwd()), 0.0));
    float k = DUST_SPLIT_A / sigmaV;
    vec3 h = vec3((WQF(3u) * dust_w_fwd().x + WQF(4u) * dust_w_fwd().y + WQF(5u) * dust_w_fwd().z) * k,
                  (WQF(4u) * dust_w_fwd().x + WQF(6u) * dust_w_fwd().y + WQF(7u) * dust_w_fwd().z) * k,
                  (WQF(5u) * dust_w_fwd().x + WQF(7u) * dust_w_fwd().y + WQF(8u) * dust_w_fwd().z) * k);
    WQ_W(3u) = floatBitsToUint(WQF(3u) - h.x * h.x);
    WQ_W(4u) = floatBitsToUint(WQF(4u) - h.x * h.y);
    WQ_W(5u) = floatBitsToUint(WQF(5u) - h.x * h.z);
    WQ_W(6u) = floatBitsToUint(WQF(6u) - h.y * h.y);
    WQ_W(7u) = floatBitsToUint(WQF(7u) - h.y * h.z);
    WQ_W(8u) = floatBitsToUint(WQF(8u) - h.z * h.z);
    WQ_W(12u) = floatBitsToUint(0.5 * WQF(12u));
    WQ_W(17u) = 0u; WQ_W(18u) = 0u; WQ_W(19u) = 0u;
    vec3 m_ = second ? vec3(WQF(0u), WQF(1u), WQF(2u)) + h : vec3(WQF(0u), WQF(1u), WQF(2u)) - h;
    WQ_W(0u) = floatBitsToUint(m_.x); WQ_W(1u) = floatBitsToUint(m_.y); WQ_W(2u) = floatBitsToUint(m_.z);
}
int dust_split_step_one_w(bool clipped, uint level, uint depthMax, vec3 eye, vec3 fwdIn, bool perspective, bool force, bool keepClip, bool second) {
    DUST_O_WRL()
    if (!(dot(dust_w_fwd(), dust_w_fwd()) > 0.0)) return DUST_SPLIT_KEEP;
    // (the volumetric test first: a pure function of the row, evaluated before the depth, the
    // spread and the span, so its temporaries are never live with them; its return keeps its place)
    bool vol = !force && perspective && dust_w_is_volumetric(eye);
    float depth = dot(vec3(WQF(0u), WQF(1u), WQF(2u)) - eye, dust_w_fwd());
    float sigmaV = sqrt(max(dust_var_along(dust_w_load_s().S, dust_w_fwd()), 0.0));
    float spanS = dot(vec3(WQF(9u), WQF(10u), WQF(11u)), dust_w_fwd());
    float span = abs(spanS);
    if (depth + 3.0 * sigmaV + max(spanS, 0.0) < 0.0) return DUST_SPLIT_DROP;
    if (vol) return DUST_SPLIT_VOLUME;
    float d1 = depth + spanS;
    if (!clipped && span > 0.0 && (depth < 0.0) != (d1 < 0.0)) {
        // a shell-forced step (keepClip): the original's forced branch draws the unclipped packet
        // as natural, so the clip is reported and the row is left as it is (the C1 shell fix)
        if (keepClip) return DUST_SPLIT_CLIP;
        float uStar = -depth / spanS;
        if (depth < 0.0) {
            float uLo = uStar + DUST_SPLIT_CLIP_EPS * (1.0 - uStar);
            dust_sub_chord_w(uLo, 1.0, false, WQ(14u) != 0u);
            return DUST_SPLIT_CLIP;
        }
        float uHi = uStar * (1.0 - DUST_SPLIT_CLIP_EPS);
        dust_sub_chord_w(0.0, uHi, WQ(13u) != 0u, false);
        return DUST_SPLIT_CLIP;
    }
    bool want = force || 3.0 * sigmaV > depth
             || (perspective && (sigmaV > DUST_SPLIT_EPS * depth || span > DUST_SPLIT_EPS * depth));
    if (want && level < min(depthMax, DUST_SPLIT_DEPTH_MAX)) {
        if (span > 3.0 * sigmaV) {
            if (second) dust_sub_chord_w(0.5, 1.0, true, WQ(14u) != 0u);
            else dust_sub_chord_w(0.0, 0.5, WQ(13u) != 0u, true);
            return DUST_SPLIT_SPLIT;
        }
        if (sigmaV > 0.0) return DUST_SPLIT_GAUSS;
    }
    if (want && !force && perspective) {
        bool deepOnly = depth > 0.0 && span > 3.0 * sigmaV
                     && !(3.0 * sigmaV > depth) && !(sigmaV > DUST_SPLIT_EPS * depth);
        if (!(deepOnly && DUST_W_DEEP_CHORD)) return DUST_SPLIT_VOLUME;
    }
    return DUST_SPLIT_KEEP;
}
#ifdef DUST_W_VIEW_SHARED
// dust_split_step_one_w's decision alone (the walk stage): the step it takes, the
// sub-chord of a clip or of a chord split not applied here but named (DUST_SPLIT_CLIP_APPLY,
// DUST_SPLIT_CHORD_APPLY) for the walk's next pass, dust_w_chord_apply, which applies it with the
// same expressions (the row is unchanged in between); the decision the same, branch for branch
const int DUST_SPLIT_CLIP_APPLY = 6;
const int DUST_SPLIT_CHORD_APPLY = 7;
int dust_split_decide_w(bool clipped, uint level, uint depthMax, vec3 eye, bool perspective, bool force) {
    DUST_O_WRL()
    // (one exit, the result initialised: the inlined function's returns in the walk's loop would
    // otherwise leave its result a value carried around the loop)
    int res = DUST_SPLIT_KEEP;
    do {
        if (!(dot(dust_w_fwd(), dust_w_fwd()) > 0.0)) { res = DUST_SPLIT_KEEP; break; }
        // (the volumetric test first: a pure function of the row, evaluated before the depth, the
        // spread and the span, so its temporaries are never live with them; its return keeps its place)
        bool vol = !force && perspective && dust_w_is_volumetric(eye);
        float depth = dot(vec3(WQF(0u), WQF(1u), WQF(2u)) - eye, dust_w_fwd());
        float sigmaV = sqrt(max(dust_var_along(dust_w_load_s().S, dust_w_fwd()), 0.0));
        float spanS = dot(vec3(WQF(9u), WQF(10u), WQF(11u)), dust_w_fwd());
        float span = abs(spanS);
        if (depth + 3.0 * sigmaV + max(spanS, 0.0) < 0.0) { res = DUST_SPLIT_DROP; break; }
        if (vol) { res = DUST_SPLIT_VOLUME; break; }
        float d1 = depth + spanS;
        if (!clipped && span > 0.0 && (depth < 0.0) != (d1 < 0.0)) { res = DUST_SPLIT_CLIP_APPLY; break; }
        bool want = force || 3.0 * sigmaV > depth
                 || (perspective && (sigmaV > DUST_SPLIT_EPS * depth || span > DUST_SPLIT_EPS * depth));
        if (want && level < min(depthMax, DUST_SPLIT_DEPTH_MAX)) {
            if (span > 3.0 * sigmaV) { res = DUST_SPLIT_CHORD_APPLY; break; }
            if (sigmaV > 0.0) { res = DUST_SPLIT_GAUSS; break; }
        }
        if (want && !force && perspective) {
            bool deepOnly = depth > 0.0 && span > 3.0 * sigmaV
                         && !(3.0 * sigmaV > depth) && !(sigmaV > DUST_SPLIT_EPS * depth);
            if (!(deepOnly && DUST_W_DEEP_CHORD)) { res = DUST_SPLIT_VOLUME; break; }
        }
        res = DUST_SPLIT_KEEP;
    } while (false);
    return res;
}
// the sub-chord the decision named (dust_split_step_one_w's, its expressions): a clip's (the
// depth, the span and uStar re-evaluated from the row as the decision evaluated them) or a chord
// split's half; the step it was (DUST_SPLIT_CLIP or DUST_SPLIT_SPLIT)
int dust_w_chord_apply(int code, vec3 eye, bool second) {
    DUST_O_WRL()
    // (one exit: see dust_split_decide_w)
    int res = DUST_SPLIT_SPLIT;
    if (code == DUST_SPLIT_CLIP_APPLY) {
        res = DUST_SPLIT_CLIP;
        float depth = dot(vec3(WQF(0u), WQF(1u), WQF(2u)) - eye, dust_w_fwd());
        float spanS = dot(vec3(WQF(9u), WQF(10u), WQF(11u)), dust_w_fwd());
        float uStar = -depth / spanS;
        if (depth < 0.0) {
            float uLo = uStar + DUST_SPLIT_CLIP_EPS * (1.0 - uStar);
            dust_sub_chord_w(uLo, 1.0, false, WQ(14u) != 0u);
        } else {
            float uHi = uStar * (1.0 - DUST_SPLIT_CLIP_EPS);
            dust_sub_chord_w(0.0, uHi, WQ(13u) != 0u, false);
        }
    } else if (second) dust_sub_chord_w(0.5, 1.0, true, WQ(14u) != 0u);
    else dust_sub_chord_w(0.0, 0.5, WQ(13u) != 0u, true);
    return res;
}
#endif

// ── The piece builders on the row ────────────────────────────────────────────────────────────
// The record's root and cut are read where they are used (RQ: the record's word k) and the piece
// is the row (q): dust_support_trunc_piece / dust_conic_piece expression for expression
#define RQ(k) DUST_REC(k)
#define RQF(k) uintBitsToFloat(RQ(k))
// the record's word k read volatile: a fresh read the compiler does not merge with another
#define RQV(k) DustWState(dust_w_row).v[(k) * DUST_SQUEUE_CAPACITY]
#define RQVF(k) uintBitsToFloat(RQV(k))
#define RSN vec3(RQVF(34u), RQVF(35u), RQVF(36u))
#define RN vec3(RQVF(23u), RQVF(24u), RQVF(25u))
// ── The piece in stages ───────────────────────────────────────────────────────────────────────
// v/G's dust_w_load_record / dust_support_trunc_piece_w / dust_conic_piece_w cut at their values, one
// stage per pass of the walk loop (the stage number in the state word's pending field while the
// load flag is set), each stage reading and writing a few of the record's and the row's words: the
// Adreno compiler materialises every word offset a block uses (k x the records' stride, a 32-bit
// constant) in a register at the block's top, so a block's register footprint follows the number
// of words it touches (v/G's piece block: 30-42 words). The values a later stage uses are parked
// in the row's words that hold nothing until the piece's final stores (3-8) and re-read there;
// the sub-chord's mean and segment (WMEAN, WSEG: the record's words with u0, u1) are the row's
// words 0-2 and 9-11, which hold exactly those values (the same expressions, evaluated once, by
// the sub-chord stage) until the piece's mean store. The expressions are the originals', in order.
// A stage's result: DUST_WP_NEXT (the next stage), DUST_WP_DONE (the piece is the row, its flux
// word final), DUST_WP_EMPTY (an empty piece: the next item).
const uint DUST_WP_NEXT = 0u;
const uint DUST_WP_DONE = 1u;
const uint DUST_WP_EMPTY = 2u;
#define WRMEAN vec3(WQVF(0u), WQVF(1u), WQVF(2u))
#define WRSEG vec3(WQVF(9u), WQVF(10u), WQVF(11u))
// the plain kind (v/G's dust_w_load_record) in three stages: words 0-6, 7-13, 14-19
void dust_wl_1() {
    WQ_W(0u) = RQ(0u); WQ_W(1u) = RQ(1u); WQ_W(2u) = RQ(2u);
    WQ_W(3u) = RQ(3u); WQ_W(4u) = RQ(4u); WQ_W(5u) = RQ(5u); WQ_W(6u) = RQ(6u);
}
void dust_wl_2() {
    WQ_W(7u) = RQ(7u); WQ_W(8u) = RQ(8u);
    WQ_W(9u) = RQ(9u); WQ_W(10u) = RQ(10u); WQ_W(11u) = RQ(11u);
    WQ_W(12u) = RQ(12u);
    WQ_W(13u) = (RQ(14u) & 1u) != 0u ? 1u : 0u;
}
void dust_wl_3() {
    WQ_W(14u) = (RQ(14u) & 2u) != 0u ? 1u : 0u;
    WQ_W(15u) = RQ(15u); WQ_W(16u) = RQ(16u);
    WQ_W(17u) = RQ(17u); WQ_W(18u) = RQ(18u); WQ_W(19u) = RQ(19u);
}
// the truncation's sub-chord bound j (dust_trunc_bound on the record's words 41-44)
float dust_wt_bound(uint j) {
    DustTruncBounds bounds;
    bounds.za = RQF(41u); bounds.zb = RQF(42u); bounds.before = RQ(43u); bounds.inner = RQ(44u);
    return dust_trunc_bound(bounds, j);
}
// the sub-chord (v/G's dust_w_sub_chord_rec) in three stages. Stage 0: u0 and u1 (the truncation's
// bounds, empty when they do not advance; the cone's uniform cut), the renormalised rho, the flux
// and the open bits; u0, u1 parked in words 3, 4 for stage 1
uint dust_wp_c1(uint kd, uint j) {
    float u0, u1;
    if (kd == DUST_SQUEUE_KIND_TRUNC) {
        u0 = dust_wt_bound(j); u1 = dust_wt_bound(j + 1u);
        if (!(u1 > u0)) return DUST_WP_EMPTY;
    } else {
        float k = float(DUST_SUPPORT_TRUNC_PIECES);
        u0 = float(j) / k; u1 = float(j + 1u) / k;
    }
    float ra = RQF(15u) + (RQF(16u) - RQF(15u)) * u0;
    float rb = RQF(15u) + (RQF(16u) - RQF(15u)) * u1;
    float rm = max(0.5 * (ra + rb), 1e-6);
    WQ_W(12u) = floatBitsToUint(RQF(12u) * (u1 - u0) * rm);
    WQ_W(13u) = j == 0u && (RQ(14u) & 1u) != 0u ? 1u : 0u;
    WQ_W(14u) = j + 1u == DUST_SUPPORT_TRUNC_PIECES && (RQ(14u) & 2u) != 0u ? 1u : 0u;
    WQ_W(15u) = floatBitsToUint(ra / rm); WQ_W(16u) = floatBitsToUint(rb / rm);
    WQ_W(3u) = floatBitsToUint(u0); WQ_W(4u) = floatBitsToUint(u1);
    return DUST_WP_NEXT;
}
// stage 1: the sub-chord's mean and segment
void dust_wp_c2() {
    float u0 = WQVF(3u), u1 = WQVF(4u);
    vec3 mean_ = vec3(RQF(0u), RQF(1u), RQF(2u)) + vec3(RQF(9u), RQF(10u), RQF(11u)) * u0;
    WQ_W(0u) = floatBitsToUint(mean_.x); WQ_W(1u) = floatBitsToUint(mean_.y); WQ_W(2u) = floatBitsToUint(mean_.z);
    vec3 seg_ = vec3(RQF(9u), RQF(10u), RQF(11u)) * (u1 - u0);
    WQ_W(9u) = floatBitsToUint(seg_.x); WQ_W(10u) = floatBitsToUint(seg_.y); WQ_W(11u) = floatBitsToUint(seg_.z);
}
// stage 2: segT
void dust_wp_c3() {
    WQ_W(17u) = RQ(17u); WQ_W(18u) = RQ(18u); WQ_W(19u) = RQ(19u);
}
// truncation stage 3: the spread along the plane normal, the degenerate case (the record's S, the
// sub-chord's flux), the slab's alpha and t (parked: alpha 3, t 4, v 5, sig 6, ds 7)
uint dust_wt_1(uint e) {
    float ra = WQVF(15u), rb = WQVF(16u);
    float uc = ra + rb > 0.0 ? (ra + 2.0 * rb) / (3.0 * (ra + rb)) : 0.5;
    float ds = dot(RN, WRSEG);
    float m = dot(RN, WRMEAN + WRSEG * uc) - RQF(26u);
    float v = RQF(37u) + ds * ds / 12.0;
    if (!(v > 0.0)) {
        if (e != 0u || !(m >= 0.0)) return DUST_WP_EMPTY;
        WQ_W(3u) = RQ(3u); WQ_W(4u) = RQ(4u); WQ_W(5u) = RQ(5u); WQ_W(6u) = RQ(6u); WQ_W(7u) = RQ(7u); WQ_W(8u) = RQ(8u);
        float w = WQF(12u);
        WQ_W(12u) = floatBitsToUint(w * uintBitsToFloat(DUST_REC(40u)));
        return DUST_WP_DONE;
    }
    float sig = sqrt(v);
    float alpha = -m / sig;
    float pk = dust_nupper(alpha);
    if (!(pk > 1e-6)) return DUST_WP_EMPTY;
    float lam = dust_npdf(alpha) / pk;
    float t = sqrt(max(1.0 + alpha * lam - lam * lam, 1e-4));
    WQ_W(3u) = floatBitsToUint(alpha); WQ_W(4u) = floatBitsToUint(t); WQ_W(5u) = floatBitsToUint(v);
    WQ_W(6u) = floatBitsToUint(sig); WQ_W(7u) = floatBitsToUint(ds);
    return DUST_WP_NEXT;
}
// truncation stage 4: the slab's moments (parked: cv 3, mu 4, ps 5; sig 6, ds 7 kept)
uint dust_wt_2(uint e) {
    float alpha = WQVF(3u), t = WQVF(4u), v = WQVF(5u);
    float za = e == 0u ? alpha : alpha + dust_support_slab_edge(e - 1u) * t;
    bool last = e + 1u == DUST_SUPPORT_TRUNC_SLABS;
    float pa = dust_npdf(za), qa = dust_nupper(za);
    float pb = 0.0, qb = 0.0, bpb = 0.0;
    if (!last) {
        float zb = alpha + dust_support_slab_edge(e) * t;
        pb = dust_npdf(zb); qb = dust_nupper(zb); bpb = zb * pb;
    }
    float ps = qa - qb;
    if (!(ps > 1e-9)) return DUST_WP_EMPTY;
    float mu = (pa - pb) / ps;
    float var_ = clamp(1.0 + (za * pa - bpb) / ps - mu * mu, 0.0, 1.0);
    float cv = (1.0 - var_) / v;
    WQ_W(3u) = floatBitsToUint(cv); WQ_W(4u) = floatBitsToUint(mu); WQ_W(5u) = floatBitsToUint(ps);
    return DUST_WP_NEXT;
}
// truncation stage 5: the mean, the flux (w = the sub-chord's flux x ps; x the record's word 40)
void dust_wt_3() {
    float ds = WQVF(7u), sig = WQVF(6u), mu = WQVF(4u);
    vec3 se = (RSN + WRSEG * (ds / 12.0)) * (1.0 / sig);
    vec3 m_ = WRMEAN + se * mu;
    WQ_W(0u) = floatBitsToUint(m_.x); WQ_W(1u) = floatBitsToUint(m_.y); WQ_W(2u) = floatBitsToUint(m_.z);
    DUST_W_CUT {
    float w = WQF(12u) * WQVF(5u);
    WQ_W(12u) = floatBitsToUint(w * uintBitsToFloat(DUST_REC(40u)));
    }
}
// truncation stage 6: the covariance
void dust_wt_4() {
    float cv = WQVF(3u);
    WQ_W(3u) = floatBitsToUint(RQF(3u) - cv * RQF(34u) * RQF(34u));
    WQ_W(4u) = floatBitsToUint(RQF(4u) - cv * RQF(34u) * RQF(35u));
    WQ_W(5u) = floatBitsToUint(RQF(5u) - cv * RQF(34u) * RQF(36u));
    DUST_W_CUT {
    WQ_W(6u) = floatBitsToUint(RQF(6u) - cv * RQF(35u) * RQF(35u));
    WQ_W(7u) = floatBitsToUint(RQF(7u) - cv * RQF(35u) * RQF(36u));
    WQ_W(8u) = floatBitsToUint(RQF(8u) - cv * RQF(36u) * RQF(36u));
    }
}
// conic stage 3: the spread along the axis and alpha (parked: alpha 3, v 4, sig 5, ds 6, uc 8)
void dust_wc_1() {
    float qa = WQVF(15u), qb = WQVF(16u);
    float uc = qa + qb > 0.0 ? (qa + 2.0 * qb) / (3.0 * (qa + qb)) : 0.5;
    float ds = dot(RN, WRSEG);
    float v = RQF(37u) + ds * ds / 12.0;
    float sig = sqrt(max(v, 1e-30));
    float alpha;
    {
        vec3 qc = WRMEAN + WRSEG * uc;
        alpha = RQ(39u) != 0u ? max((RQF(26u) - dot(RN, qc)) / sig, -DUST_CONIC_FAR) : -DUST_CONIC_FAR;
    }
    WQ_W(3u) = floatBitsToUint(alpha); WQ_W(4u) = floatBitsToUint(v); WQ_W(5u) = floatBitsToUint(sig);
    WQ_W(6u) = floatBitsToUint(ds); WQ_W(8u) = floatBitsToUint(uc);
}
// conic stage 4: the slab's moments, cv (word 7), the flux final (parked: mu 3)
uint dust_wc_2(uint e) {
    float alpha = WQVF(3u);
    // the slab edges above alpha: a suffix of (-1, 0, 1)
    uint c = 0u;
    for (uint i = 0u; i < DUST_SUPPORT_TRUNC_SLABS - 1u; ++i) if (dust_conic_slab_edge(i) > alpha) c++;
    uint ne = c + 1u;
    if (e >= ne) return DUST_WP_EMPTY;
    uint first = DUST_SUPPORT_TRUNC_SLABS - 1u - c;
    float za = e == 0u ? alpha : dust_conic_slab_edge(first + e - 1u);
    float zb = e + 1u == ne ? DUST_CONIC_FAR : dust_conic_slab_edge(first + e);
    float pa = dust_npdf(za), qa_ = dust_nupper(za), apa = za * dust_npdf(za);
    float pb = dust_npdf(zb), qb_ = dust_nupper(zb), bpb = zb * dust_npdf(zb);
    float ps = qa_ - qb_;
    if (!(ps > 1e-9)) return DUST_WP_EMPTY;
    float mu = (pa - pb) / ps;
    float var_ = clamp(1.0 + (apa - bpb) / ps - mu * mu, 0.0, 1.0);
    float v = WQVF(4u);
    WQ_W(7u) = floatBitsToUint((1.0 - var_) / max(v, 1e-30));
    WQ_W(3u) = floatBitsToUint(mu);
    float w = WQF(12u) * ps;
    WQ_W(12u) = floatBitsToUint(w * uintBitsToFloat(DUST_REC(40u)));
    return DUST_WP_NEXT;
}
// conic stage 5: the mean and the offset (words 0-2, 3-5)
void dust_wc_3() {
    float mu = WQVF(3u), sig = WQVF(5u), ds = WQVF(6u);
    vec3 se = (RSN + WRSEG * (ds / 12.0)) * (1.0 / sig);
    vec3 mean = WRMEAN + se * mu;
    WQ_W(0u) = floatBitsToUint(mean.x); WQ_W(1u) = floatBitsToUint(mean.y); WQ_W(2u) = floatBitsToUint(mean.z);
    vec3 off = se * mu;
    WQ_W(3u) = floatBitsToUint(off.x); WQ_W(4u) = floatBitsToUint(off.y); WQ_W(5u) = floatBitsToUint(off.z);
}
// conic stage 6: the S map's factor sk (word 6) and the mean's lateral shift
void dust_wc_4() {
    float uc = WQVF(8u);
    WQ_W(6u) = floatBitsToUint(max((dot(RN, vec3(WQVF(0u), WQVF(1u), WQVF(2u)) + WRSEG * uc) - RQF(27u)) / RQF(38u), 0.0));
    DUST_W_CUT {
        vec3 off = vec3(WQVF(3u), WQVF(4u), WQVF(5u));
        vec3 lat = off - RN * dot(RN, off);
        vec3 m_ = vec3(WQVF(0u), WQVF(1u), WQVF(2u)) + lat * (WQVF(6u) - 1.0);
        WQ_W(0u) = floatBitsToUint(m_.x); WQ_W(1u) = floatBitsToUint(m_.y); WQ_W(2u) = floatBitsToUint(m_.z);
    }
}
// conic stage 7: the covariance (words 3-8)
void dust_wc_5() {
    float sk = WQVF(6u);
    float cv = WQVF(7u);
    float C0 = RQF(3u) - cv * RQF(34u) * RQF(34u);
    float C1 = RQF(4u) - cv * RQF(34u) * RQF(35u);
    float C2 = RQF(5u) - cv * RQF(34u) * RQF(36u);
    float C3 = RQF(6u) - cv * RQF(35u) * RQF(35u);
    float C4 = RQF(7u) - cv * RQF(35u) * RQF(36u);
    float C5 = RQF(8u) - cv * RQF(36u) * RQF(36u);
    vec3 c0 = dust_apex_map(vec3(C0, C1, C2), RN, sk);
    vec3 c1 = dust_apex_map(vec3(C1, C3, C4), RN, sk);
    vec3 c2 = dust_apex_map(vec3(C2, C4, C5), RN, sk);
    vec3 r0 = dust_apex_map(vec3(c0.x, c1.x, c2.x), RN, sk);
    WQ_W(3u) = floatBitsToUint(r0.x); WQ_W(4u) = floatBitsToUint(r0.y); WQ_W(5u) = floatBitsToUint(r0.z);
    vec3 r1 = dust_apex_map(vec3(c0.y, c1.y, c2.y), RN, sk);
    WQ_W(6u) = floatBitsToUint(r1.y); WQ_W(7u) = floatBitsToUint(r1.z);
    vec3 r2 = dust_apex_map(vec3(c0.z, c1.z, c2.z), RN, sk);
    WQ_W(8u) = floatBitsToUint(r2.z);
}
// the candidate put from the row (DUST_CQUEUE_PUT's words)
#define DUST_CQUEUE_PUT_W(pyr, base, cap, r, volume, rec) { \
    uint b_ = (base) + DUST_VQUEUE_HEADER_WORDS + (r) * DUST_CQUEUE_RECORD_WORDS; const uint c_ = 1u; \
    pyr.v[b_ + 0u * c_] = WQ(0u); pyr.v[b_ + 1u * c_] = WQ(1u); pyr.v[b_ + 2u * c_] = WQ(2u); \
    pyr.v[b_ + 3u * c_] = WQ(3u); pyr.v[b_ + 4u * c_] = WQ(4u); pyr.v[b_ + 5u * c_] = WQ(5u); \
    pyr.v[b_ + 6u * c_] = WQ(6u); pyr.v[b_ + 7u * c_] = WQ(7u); pyr.v[b_ + 8u * c_] = WQ(8u); \
    pyr.v[b_ + 9u * c_] = WQ(9u); pyr.v[b_ + 10u * c_] = WQ(10u); pyr.v[b_ + 11u * c_] = WQ(11u); \
    pyr.v[b_ + 12u * c_] = WQ(12u); \
    pyr.v[b_ + 13u * c_] = WQ(15u); pyr.v[b_ + 14u * c_] = WQ(16u); \
    pyr.v[b_ + 15u * c_] = (WQ(13u) != 0u ? 1u : 0u) | (WQ(14u) != 0u ? 2u : 0u) | ((volume) ? 4u : 0u); \
    pyr.v[b_ + 16u * c_] = (rec); \
}

// ── The walk stage's leaf: a candidate, not a drawing ────────────────────────────────────────
// The run's next slot (dust_packet_ord, the walk-order index the seed ordinal was; the record's
// first slot set by dust_split_record), or none when the cluster's run was refused
bool dust_cand_on = false;
void dust_cand_emit(DustPacket q, bool volume) {
    if (dust_counting) { dust_seed_count++; return; }
    DUST_PROF_COUNT(leaves, 1u)
    if (dust_cand_on) {
        uint cbase = dust_cqueue_base_o(uvec2(pc.pyramid));
        DUST_CQUEUE_PUT(pc.pyramid, cbase, dust_cqueue_capacity_o(uvec2(pc.pyramid), cbase), dust_packet_ord, q, volume, dust_leaf_rec)
    }
    dust_packet_ord += 1u;
}
void dust_cand_emit_w(bool volume) {
    if (dust_counting) { dust_seed_count++; return; }
    DUST_PROF_COUNT(leaves, 1u)
    if (dust_cand_on) {
        uint cbase = dust_cqueue_base_o(uvec2(pc.pyramid));
        DUST_CQUEUE_PUT_W(pc.pyramid, cbase, dust_cqueue_capacity_o(uvec2(pc.pyramid), cbase), dust_packet_ord, volume, dust_leaf_rec)
    }
    dust_packet_ord += 1u;
}



// Splits a packet for the perspective (dust_split_step: clipped at the eye plane, sliced by
// depth) and projects and scatters every leaf as it is found (dust::project_split_scatter): an
// walk without a stack (a path code; below), depth first, the same loop as the host
// mirror dust::split_packet_with. Every packet splits to the frame's depth limit depthMax (the
// host-written DUST_PYR_SPLIT_DEPTH word). The children made are counted in
// DUST_PYR_SPLIT_CHILDREN: the host lowers the next frame's depth when they pass
// DUST_SPLIT_BUDGET, and past DUST_FRAME_HARD_FACTOR budgets a thread stops splitting at once —
// the one run-time gate, so it depends on the dispatch order, only in frames the feedback has
// not caught up with (a cold start inside the coma made 5.5e7 children). A dropped packet's flux
// is atomicMax'ed into DUST_PYR_SPLIT_DROPPED_FLUX.
#ifdef DUST_HARD_AXIS
// (the hard-axis sub-leaf diagnostic's blob, dust_split_hard: the former in-walk cut and the
// sub-leaf loop; its spill is the diagnostic's; a sub-leaf's leaf record carries the whole
// leaf's record dust_leaf_rec, set by dust_split_record)
void dust_project_split_scatter_one(vec3 mean, float S[6], vec3 d, float flux, float exposure,
                                bool openStart, bool openEnd, vec2 rho, vec3 segT, vec3 color, uint levels,
                                float W, float H, vec3 fwd, bool perspective, uint depthMax, uint volumeBias,
                                float shellDMax, uint volumeAxis) {
    DUST_PROF_BEGIN(packet)
    DUST_PROF_COUNT(packets, 1u)
    DustPacket p0;
    p0.mean = mean; p0.S = S; p0.seg = d; p0.flux = flux;
    p0.openStart = openStart; p0.openEnd = openEnd; p0.rho = rho; p0.segT = segT;
    // a perspective packet its support cuts (dust::project_split_scatter; not with a speck shell,
    // the split's hand-over): drawn whole by the exact ray integral over the half-space when the
    // eye is near it, else as its truncation's moments (dust_support_trunc_piece), uncut
    vec4 sup = dust_phot_support;
    bool hasO = dust_phot_has_origin;
    float wO = dust_phot_origin;
    // a perspective packet with a cell the eye is not near: its cone of grains (dust_conic_piece,
    // the cut baked in), each slab drawn uncut; not with a speck shell (dust::project_split_scatter)
    bool conic = false;
    vec3 csn = vec3(0.0);
    float cv0 = 0.0, cDC = 0.0;
    bool cCut = false;
    float renorm = 1.0;
    DUST_PROF_BEGIN_AS(cone1)
    // (the eye near it: the exact ray integral below, the work bounded per packet)
    if (perspective && !(shellDMax > 0.0) && dust_has_support(sup) && hasO && !dust_is_volumetric(p0, pc.eyeLocal.xyz) && !dust_legacy_kernel()) {
        float ra = p0.rho.x, rb = p0.rho.y;
        float ucp = ra + rb > 0.0 ? (ra + 2.0 * rb) / (3.0 * (ra + rb)) : 0.5;
        cDC = dot(sup.xyz, p0.mean + p0.seg * ucp) - wO;
        if (cDC > 0.0) {
            csn = vec3(S[0] * sup.x + S[1] * sup.y + S[2] * sup.z,
                       S[1] * sup.x + S[3] * sup.y + S[4] * sup.z,
                       S[2] * sup.x + S[4] * sup.y + S[5] * sup.z);
            cv0 = max(dot(sup.xyz, csn), 0.0);
            cCut = dust_support_mass(p0, sup) < 1.0 - DUST_SUPPORT_CUT_MIN;
            float total = 0.0;
            uint cnt = 0u;
            for (uint it = 0u; it < DUST_SUPPORT_TRUNC_PIECES * DUST_SUPPORT_TRUNC_SLABS; ++it) {
                DustPacket q; float w;
                if (dust_conic_piece(p0, sup, wO, csn, cv0, cDC, cCut, it / DUST_SUPPORT_TRUNC_SLABS, it % DUST_SUPPORT_TRUNC_SLABS, q, w)) {
                    total += w;
                    cnt++;
                }
            }
            if (cnt > 0u) {
                conic = true;
                renorm = p0.flux > 0.0 ? 1.0 / max(total / p0.flux, DUST_SUPPORT_MASS_MIN) : 0.0;
                dust_phot_support = vec4(0.0);
                dust_phot_has_origin = false;
            }
        }
    }
    bool trunc = false;
    if (!conic && perspective && !(shellDMax > 0.0) && dust_has_support(sup)
        && dust_support_mass(p0, sup) < 1.0 - DUST_SUPPORT_CUT_MIN) {
        if (dust_is_volumetric(p0, pc.eyeLocal.xyz)) {
            DUST_PROF_END_AS(cone, cone1)
            // (counted under `split` like the loop's leaves: every leaf-level cost is inside it)
            DUST_PROF_BEGIN_AS(vol0)
            dust_cand_emit(p0, true);
            DUST_PROF_END_AS(split, vol0)
            DUST_PROF_END(packet)
            return;
        }
        trunc = true;
    }
    DustTruncBounds bounds;
    bounds.za = 0.0; bounds.zb = 1.0; bounds.before = 0u; bounds.inner = 0u;
    vec3 sn = vec3(0.0);
    float v0 = 0.0;
    if (trunc) {
        sn = vec3(S[0] * sup.x + S[1] * sup.y + S[2] * sup.z,
                  S[1] * sup.x + S[3] * sup.y + S[4] * sup.z,
                  S[2] * sup.x + S[4] * sup.y + S[5] * sup.z);
        v0 = max(dot(sup.xyz, sn), 0.0);
        bounds = dust_support_trunc_bounds(p0, sup, v0);
        // the kept shares, renormalised over the packet (floor DUST_SUPPORT_MASS_MIN)
        float total = 0.0;
        for (uint it = 0u; it < DUST_SUPPORT_TRUNC_PIECES * DUST_SUPPORT_TRUNC_SLABS; ++it) {
            uint j = it / DUST_SUPPORT_TRUNC_SLABS, e = it % DUST_SUPPORT_TRUNC_SLABS;
            DustPacket q; float w;
            if (dust_support_trunc_piece(p0, sup, sn, v0, dust_trunc_bound(bounds, j), dust_trunc_bound(bounds, j + 1u), j, e, q, w)) total += w;
        }
        renorm = p0.flux > 0.0 ? 1.0 / max(total / p0.flux, DUST_SUPPORT_MASS_MIN) : 0.0;
        dust_phot_support = vec4(0.0);
    }
    DUST_PROF_END_AS(cone, cone1)
    DUST_PROF_COUNT(cone, (trunc || conic) ? 1u : 0u)
    uint nItems = (trunc || conic) ? DUST_SUPPORT_TRUNC_PIECES * DUST_SUPPORT_TRUNC_SLABS : 1u;
    // the in-frame gate (dust::split_packet_with): once a split of this thread finds the frame's
    // children past DUST_FRAME_HARD_FACTOR budgets, its packets split no further (depth 0)
    uint dMax = depthMax;
    // Each item's split tree walked depth first without a stack (a stack would
    // be the splitter's local memory): the node in registers; a split goes on with its first
    // child a, and its second child b is recomputed when its turn comes, from the item's root
    // along the path code (bit L: the child taken at the split from level L, 1 = b). The turn
    // order is the stack's: a node done, the next is the b of the deepest split on the path whose
    // a was taken (findMSB of the pending bits), the stack's top. A replay step is
    // dust_split_step with force at DUST_SPLIT_DEPTH_MAX: on a node the first visit clipped or
    // split (naturally or shell-forced) it returns the same CLIP or SPLIT with the same children
    // (the drop and clip tests come first and read neither force nor the depth limit; a split's
    // want and level tests hold under force). Replays draw nothing and count nothing (no seed, no
    // SPLIT_CHILDREN atomic, no node count): every side effect is a first visit's, in the old
    // order. A shell-forced split is its own iteration on the same node (`forced`, its first step
    // in `natural`), so first visits, forced splits and replays share one dust_split_step.
    uint it = 0u;                    // the item (piece) whose tree is walked
    uint path = 0u;                  // bit L: the child taken at the split from level L (1 = b)
    uint wf = 1u | (uint(DUST_SPLIT_KEEP) << 16u);   // bit 0 load ((re)load item it's root), 1 replay (the node is on the path to that b), 2 forced (the node's shell-forced split is next), 3 clipped; bits 8-15 target (replaying: the level of the b being recomputed); bits 16-19 natural (a forced node's first step, KEEP or VOLUME)
    DustPacket q;
    uint level = 0u;
    while (it < nItems) {
        if (((wf & 1u) != 0u)) {
            bool ok = true;
            float w = 0.0;
            if (conic) {
                ok = dust_conic_piece(p0, sup, wO, csn, cv0, cDC, cCut, it / DUST_SUPPORT_TRUNC_SLABS, it % DUST_SUPPORT_TRUNC_SLABS, q, w);
            } else if (trunc) {
                uint j = it / DUST_SUPPORT_TRUNC_SLABS, e = it % DUST_SUPPORT_TRUNC_SLABS;
                ok = dust_support_trunc_piece(p0, sup, sn, v0, dust_trunc_bound(bounds, j), dust_trunc_bound(bounds, j + 1u), j, e, q, w);
            } else {
                q = p0;
            }
            if (conic || trunc) q.flux = w * renorm;
            dust_w_store(q);
            level = 0u;
            wf &= ~8u;
            wf &= ~1u;
            // an empty piece: the next item (a replayed root loaded on its first visit: never here)
            if (!ok) { it++; path = 0u; wf &= ~2u; wf |= 1u; }
        } else {
            DUST_PROF_BEGIN(split)
            bool first = !((wf & 2u) != 0u) && !((wf & 4u) != 0u);
            DUST_PROF_COUNT(nodes, first ? 1u : 0u)
            // (q becomes the child the walk takes — the second on a ((wf & 2u) != 0u) whose path
            // bit is set — on a clip or a split, and stays itself otherwise)
            bool persp_;
            vec3 fwd_ = dust_view_forward(pc.mvp, pc.eyeLocal.xyz, persp_);
            int step = dust_split_step_one_w(((wf & 8u) != 0u), level, first ? dMax : DUST_SPLIT_DEPTH_MAX, pc.eyeLocal.xyz, fwd_, persp_, !first, ((wf & 4u) != 0u),
                                           ((wf & 2u) != 0u) && ((path >> level) & 1u) != 0u);
            if (step == DUST_SPLIT_GAUSS) { dust_w_gauss(((wf & 2u) != 0u) && ((path >> level) & 1u) != 0u); step = DUST_SPLIT_SPLIT; }
            bool done = false, draw = false;
            int drawAs = step;
            if (((wf & 2u) != 0u)) {
                if (step == DUST_SPLIT_CLIP) {
                    wf |= 8u;
                } else if (step == DUST_SPLIT_SPLIT) {
                    level++;
                    wf &= ~8u;
                    wf = (wf & ~2u) | ((level < ((wf >> 8u) & 0xFFu)) ? 2u : 0u);
                } else {
                    // (unreachable, see above: ends the walk below this node, never loops)
                    wf &= ~2u; done = true;
                }
            } else if (((wf & 4u) != 0u)) {
                // the speck shell's ((wf & 4u) != 0u) split (dust::split_packet_with)
                wf &= ~4u;
                if (step == DUST_SPLIT_SPLIT) {
                    if (!dust_counting && atomicAdd(pc.pyramid.v[DUST_PYR_SPLIT_CHILDREN], 2u) > DUST_FRAME_HARD_FACTOR * DUST_SPLIT_BUDGET && dust_frame_gates_on()) dMax = 0u;
                    path &= ~(1u << level);
                    level++; wf &= ~8u;
                } else {
                    WQ_W(12u) = floatBitsToUint(WQF(12u) * dust_shell_outside_fraction(dust_w_load_mss(), pc.eyeLocal.xyz, shellDMax));
                    draw = WQF(12u) > 0.0; drawAs = int((wf >> 16u) & 0xFu); done = true;
                }
            } else if (step == DUST_SPLIT_DROP) {
                if (!dust_counting) atomicMax(pc.pyramid.v[DUST_PYR_SPLIT_DROPPED_FLUX], floatBitsToUint(max(WQF(12u), 0.0)));
                done = true;
            } else if (step == DUST_SPLIT_KEEP || step == DUST_SPLIT_VOLUME) {
                // the speck shell's hand-over (dust::split_packet_with): wholly inside, the specks'
                // (not drawn); across the boundary, split once more while the depth allows, else
                // drawn with the fraction of its mass outside
                int cls = dust_shell_classify(dust_w_load_mss(), pc.eyeLocal.xyz, shellDMax);
                if (cls == DUST_SHELL_STRADDLES && level < min(dMax, DUST_SPLIT_DEPTH_MAX)) {
                    wf |= 4u; wf = (wf & ~0xF0000u) | (uint(step) << 16u);
                } else {
                    if (cls == DUST_SHELL_OUTSIDE) {
                        draw = true;
                    } else if (cls == DUST_SHELL_STRADDLES) {
                        WQ_W(12u) = floatBitsToUint(WQF(12u) * dust_shell_outside_fraction(dust_w_load_mss(), pc.eyeLocal.xyz, shellDMax));
                        draw = WQF(12u) > 0.0;
                    }
                    done = true;
                }
            } else if (step == DUST_SPLIT_CLIP) {
                wf |= 8u;
            } else {
                if (!dust_counting && atomicAdd(pc.pyramid.v[DUST_PYR_SPLIT_CHILDREN], 2u) > DUST_FRAME_HARD_FACTOR * DUST_SPLIT_BUDGET && dust_frame_gates_on()) dMax = 0u;
                path &= ~(1u << level);
                level++; wf &= ~8u;
            }
            // (the leaf is a candidate for the sequencer stage, which projects
            // and draws it — dust_volume_emit or dust_project_packet + scatter — in this order)
            if (draw) dust_cand_emit_w(drawAs == DUST_SPLIT_VOLUME);
            if (done) {
                // the next node: the b of the deepest split on the path whose a was taken
                uint pending = ~path & ((1u << level) - 1u);
                if (pending == 0u) {
                    it++; path = 0u;
                } else {
                    uint L = uint(findMSB(pending));
                    path = (path & ((1u << L) - 1u)) | (1u << L);
                    wf = (wf & ~0xFF00u) | ((L + 1u) << 8u);
                    wf |= 2u;
                }
                wf |= 1u;
            }
            DUST_PROF_END(split)
        }
    }
#ifdef DUST_HARD_AXIS
    dust_phot_support = sup;
    dust_phot_has_origin = hasO;
#endif
    DUST_PROF_END(packet)
}
#endif  // DUST_HARD_AXIS (its walk body; the production walk below)
#ifndef DUST_HARD_AXIS
// ── The production walk: the record's words through the record pointer dust_w_row
// (DUST_REC), the node in the record's row (WQ), the walk's state in one packed word, one counter ──
// The walk state of dust_project_split_scatter (it, path, level, dMax, the flags, target, natural)
// packed in one word `ws` (DUST_SPLIT_DEPTH_MAX = 4: level, target, dMax <= 4, path 4 bits): bits
// 0-3 the flags (load, replay, forced, clipped), 4-6 level, 7-9 target, 10-12 natural (KEEP or
// VOLUME), 13-15 the step pending (the split step's result for the bookkeeping phase), 16-19
// path, 20-22 dMax, 23 the phase (the step taken, its bookkeeping pending), 24-29 it. The word
// lives in the record (word 47, DUST_SQUEUE_WALK_STATE), read at the top of every pass of the
// walk loop and written at its end: a pass is one phase (the piece, the split step, the step's
// bookkeeping), nothing but the record pointer and the counter carried from one to the next. The
// same transitions, statement for statement (integer bookkeeping only).
#define WS_GET(o, n) bitfieldExtract(ws, int(o), int(n))
#define WS_SET(o, n, x) ws = bitfieldInsert(ws, uint(x), int(o), int(n))
#define WS_LEVEL WS_GET(4, 3)
#define WS_TARGET WS_GET(7, 3)
#define WS_NATURAL WS_GET(10, 3)
#define WS_PEND WS_GET(13, 3)
#define WS_PATH WS_GET(16, 4)
#define WS_DMAX WS_GET(20, 3)
#define WS_IT WS_GET(24, 6)
#define WS_SET_LEVEL(x) WS_SET(4, 3, x)
#define WS_SET_TARGET(x) WS_SET(7, 3, x)
#define WS_SET_NATURAL(x) WS_SET(10, 3, x)
#define WS_SET_PEND(x) WS_SET(13, 3, x)
#define WS_SET_PATH(x) WS_SET(16, 4, x)
#define WS_SET_DMAX(x) WS_SET(20, 3, x)
#define WS_SET_IT(x) WS_SET(24, 6, x)
const uint WS_PHASE = 0x800000u;
// the state word: a volatile access (read once per pass, written once per pass: the compiler
// neither forwards it across the passes nor keeps it in a register over the loop)
const uint DUST_SQUEUE_WALK_STATE = 47u;
#define WSTATE DustWState(dust_w_row).v[DUST_SQUEUE_WALK_STATE * DUST_SQUEUE_CAPACITY]
// the walk's counter (dust_packet_ord's role: the candidates counted from 0 by the counting pass,
// the run's next slot in the writing pass) in the record's word 46 (DUST_SQUEUE_CAND_COUNT, which
// the walk stage writes after the walk), and the pass's two flags in the state word: bit 30 the
// counting pass (dust_counting), bit 31 the candidates written (dust_cand_on)
#define WORDV DustWState(dust_w_row).v[DUST_SQUEUE_CAND_COUNT * DUST_SQUEUE_CAPACITY]
#define WSTATE_W WSTATE
#define WORDV_W WORDV
const uint WS_COUNTING = 0x40000000u;
const uint WS_CAND_ON = 0x80000000u;
// the record's index (recs + r) from its pointer: nothing but the pointer stays live over the walk
uint dust_w_rec() { return (dust_w_row.x - uvec2(pc.pyramid).x) >> 2u; }
// the walk's one counter (dust_packet_ord): the candidates counted from 0 by the counting pass,
// the run's next slot in the writing pass (dust_cand_emit / dust_cand_emit_w: dust_seed_count and
// dust_packet_ord, one of which each pass moves)
// the speck shell's tests on the row (dust_shell_classify,
// dust_shell_outside_fraction: the same expressions in the same order), each row word read volatile
// where it is used, the words of a group of expressions read together and the groups cut into
// blocks of their own (DUST_W_CUT): the Adreno compiler materialises a block's word offsets and
// their 64-bit addresses ahead of the block's loads, so a block loading the packet's 12 words holds
// 24 address registers. What one group hands the next are the originals' scalars
int dust_w_shell_classify(vec3 eye, float dMax) {
    int res = DUST_SHELL_OUTSIDE;
    if (dust_shell_active(dMax)) {
        vec3 centre = vec3(WQVF(0u), WQVF(1u), WQVF(2u)) + vec3(WQVF(9u), WQVF(10u), WQVF(11u)) * 0.5;
        vec3 dd = centre - eye;
        float dist = sqrt(dot(dd, dd));
        DUST_W_CUT {
            float tr = max(WQVF(3u) + WQVF(6u) + WQVF(8u), 0.0);
            DUST_W_CUT {
                vec3 seg = vec3(WQVF(9u), WQVF(10u), WQVF(11u));
                float reach = 3.0 * sqrt(tr) + 0.5 * sqrt(dot(seg, seg));
                if (dist + reach < dMax) res = DUST_SHELL_INSIDE;
                else if (dist - reach > dMax) res = DUST_SHELL_OUTSIDE;
                else res = DUST_SHELL_STRADDLES;
            }
        }
    }
    return res;
}
float dust_w_shell_outside_fraction(vec3 eye, float dMax) {
    float res = 1.0;
    if (dust_shell_active(dMax)) {
        vec3 sg = vec3(WQVF(9u), WQVF(10u), WQVF(11u));
        vec3 centre = vec3(WQVF(0u), WQVF(1u), WQVF(2u)) + sg * 0.5;
        vec3 d = centre - eye;
        float dist = sqrt(dot(d, d));
        DUST_W_CUT {
            float c0 = WQVF(3u) + sg.x * sg.x / 12.0;
            float c1 = WQVF(4u) + sg.x * sg.y / 12.0;
            float c2 = WQVF(5u) + sg.x * sg.z / 12.0;
            float c3 = WQVF(6u) + sg.y * sg.y / 12.0;
            float c4 = WQVF(7u) + sg.y * sg.z / 12.0;
            float c5 = WQVF(8u) + sg.z * sg.z / 12.0;
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
            res = clamp(1.0 - clamp(radial * transverse * kappa, 0.0, 1.0), 0.0, 1.0);
        }
    }
    return res;
}
void dust_w_cand_put(bool volume, bool on) {
    DUST_O_WRL()
    DUST_PROF_COUNT(leaves, dust_counting ? 0u : 1u)
    // (the counter in the record's word 46 while the walk runs: read and written here only)
    uint ord = WORDV;
    if (on) {
        uvec2 cp = dust_w_addr(dust_cqueue_base_o(uvec2(pc.pyramid)) + DUST_VQUEUE_HEADER_WORDS + ord * DUST_CQUEUE_RECORD_WORDS);
        DUST_O_CP(0u) = WQV(0u); DUST_O_CP(1u) = WQV(1u); DUST_O_CP(2u) = WQV(2u);
        DUST_O_CP(3u) = WQV(3u); DUST_O_CP(4u) = WQV(4u); DUST_O_CP(5u) = WQV(5u);
        DUST_O_CP(6u) = WQV(6u); DUST_O_CP(7u) = WQV(7u); DUST_O_CP(8u) = WQV(8u);
        DUST_W_CUT {
        DUST_O_CP(9u) = WQV(9u); DUST_O_CP(10u) = WQV(10u); DUST_O_CP(11u) = WQV(11u);
        DUST_O_CP(12u) = WQV(12u);
        DUST_O_CP(13u) = WQV(15u); DUST_O_CP(14u) = WQV(16u);
        DUST_O_CP(15u) = (WQV(13u) != 0u ? 1u : 0u) | (WQV(14u) != 0u ? 2u : 0u) | (volume ? 4u : 0u);
        DUST_O_CP(16u) = dust_w_rec();
        }
    }
    WORDV_W = ord + 1u;
}
// the VOLUME kind's root as its one candidate (dust_cand_emit(dust_squeue_packet(...), true): the
// record's words copied; the open bits from word 14 as dust_squeue_packet decodes them)
void dust_w_cand_put_root() {
    DUST_PROF_COUNT(leaves, dust_counting ? 0u : 1u)
    uint ord = WORDV;
    if (!dust_counting && dust_cand_on) {
        uvec2 cp = dust_w_addr(dust_cqueue_base_o(uvec2(pc.pyramid)) + DUST_VQUEUE_HEADER_WORDS + ord * DUST_CQUEUE_RECORD_WORDS);
        DUST_O_CP(0u) = RQ(0u); DUST_O_CP(1u) = RQ(1u); DUST_O_CP(2u) = RQ(2u);
        DUST_O_CP(3u) = RQ(3u); DUST_O_CP(4u) = RQ(4u); DUST_O_CP(5u) = RQ(5u);
        DUST_O_CP(6u) = RQ(6u); DUST_O_CP(7u) = RQ(7u); DUST_O_CP(8u) = RQ(8u);
        DUST_W_CUT {
        DUST_O_CP(9u) = RQ(9u); DUST_O_CP(10u) = RQ(10u); DUST_O_CP(11u) = RQ(11u);
        DUST_O_CP(12u) = RQ(12u);
        DUST_O_CP(13u) = RQ(15u); DUST_O_CP(14u) = RQ(16u);
        DUST_O_CP(15u) = (RQ(14u) & 3u) | 4u;
        DUST_O_CP(16u) = dust_w_rec();
        }
    }
    WORDV_W = ord + 1u;
}
// the record's count at the walk's end (the state word's flags: written by dust_walk's start)
void dust_w_count_done() {
    if ((WSTATE & WS_COUNTING) == 0u) WORDV_W = (WSTATE & WS_CAND_ON) != 0u ? WORDV - RQ(DUST_SQUEUE_CAND_BASE) : 0u;
}
#define kind DUST_REC(33u)
#define nItems (kind != DUST_SQUEUE_KIND_PLAIN ? DUST_SUPPORT_TRUNC_PIECES * DUST_SUPPORT_TRUNC_SLABS : 1u)
#define shellDMax ((DUST_REC(14u) & 8u) != 0u ? uintBitsToFloat(pc.pyramid.v[252u]) : 0.0)
// (depthMax: the frame's split depth, min(DUST_PYR_SPLIT_DEPTH & 0xFF, DUST_SPLIT_DEPTH_MAX))
void dust_walk(uint depthMax) {
    DUST_PROF_BEGIN(packet)
    DUST_PROF_COUNT(packets, 1u)
    WSTATE_W = 1u | (uint(DUST_SPLIT_KEEP) << 10u) | (depthMax << 20u) | (dust_counting ? WS_COUNTING : 0u) | (dust_cand_on ? WS_CAND_ON : 0u);
    if (kind == DUST_SQUEUE_KIND_VOLUME) {
        DUST_PROF_BEGIN_AS(vol0)
        dust_w_cand_put_root();
        dust_w_count_done();
        DUST_PROF_END_AS(split, vol0)
        DUST_PROF_END(packet)
        return;
    }
    for (;;) {
        DUST_O_WRL()
        uint ws = WSTATE;
        if (!(WS_IT < nItems)) break;
        // the pass's phase (selects, not branches; the phases then as a sequence of exclusive ifs,
        // not an if-else chain, so no phase is nested in another's mask): 0 the piece, 1 the split
        // step, 2 the sub-chord it named, 3 the Gaussian split's update, 4 the bookkeeping
        uint ph = (ws & 1u) != 0u ? 0u
                : ((ws & WS_PHASE) == 0u ? 1u
                : (WS_PEND >= uint(DUST_SPLIT_CLIP_APPLY) ? 2u
                : (WS_PEND == uint(DUST_SPLIT_GAUSS) ? 3u : 4u)));
        if (ph == 0u) {
            // the piece phase in stages (v/G's dust_w_load_record, dust_support_trunc_piece_w,
            // dust_conic_piece_w), one per pass: the stage in the pending field
            uint kd = RQV(33u);
            uint st = WS_PEND;
            uint e = WS_IT % DUST_SUPPORT_TRUNC_SLABS;
            uint r = DUST_WP_NEXT;
            // (one case per kind and stage: the stages never share a block)
            uint sc = (kd == DUST_SQUEUE_KIND_PLAIN ? 0u : (kd == DUST_SQUEUE_KIND_TRUNC ? 8u : 16u)) + st;
            switch (sc) {
            case 0u: dust_wl_1(); break;
            case 1u: dust_wl_2(); break;
            case 2u: dust_wl_3(); r = DUST_WP_DONE; break;
            case 8u: case 16u: r = dust_wp_c1(kd, WS_IT / DUST_SUPPORT_TRUNC_SLABS); break;
            case 9u: case 17u: dust_wp_c2(); break;
            case 10u: case 18u: dust_wp_c3(); break;
            case 11u: r = dust_wt_1(e); break;
            case 12u: r = dust_wt_2(e); break;
            case 13u: dust_wt_3(); break;
            case 14u: dust_wt_4(); r = DUST_WP_DONE; break;
            case 19u: dust_wc_1(); break;
            case 20u: r = dust_wc_2(e); break;
            case 21u: dust_wc_3(); break;
            case 22u: dust_wc_4(); break;
            case 23u: dust_wc_5(); r = DUST_WP_DONE; break;
            default: break;
            }
            if (r == DUST_WP_NEXT) WS_SET_PEND(st + 1u);
            if (r == DUST_WP_EMPTY) {
                // (the original's flux store of an empty piece: w = 0)
                float w = 0.0;
                WQ_W(12u) = floatBitsToUint(w * uintBitsToFloat(DUST_REC(40u)));
            }
            if (r != DUST_WP_NEXT) {
                WS_SET_LEVEL(0u);
                ws &= ~8u;
                ws &= ~1u;
                // an empty piece: the next item (its stages from 0)
                if (r == DUST_WP_EMPTY) { WS_SET_IT(WS_IT + 1u); WS_SET_PATH(0u); ws &= ~2u; ws |= 1u; WS_SET_PEND(0u); }
            }
        }
        if (ph == 1u) {
            // the split step phase: the node's step, pending
            DUST_PROF_BEGIN(split)
            bool first = !((ws & 2u) != 0u) && !((ws & 4u) != 0u);
            DUST_PROF_COUNT(nodes, first ? 1u : 0u)
            // (the view direction: dust_w_fwd, read inside the step; the perspective flag dust_w_persp)
            int step = dust_split_decide_w(((ws & 8u) != 0u), WS_LEVEL, first ? WS_DMAX : DUST_SPLIT_DEPTH_MAX, pc.eyeLocal.xyz, dust_w_persp(), !first);
            // (the C1 shell fix) a shell-forced node keeps its row unclipped: the clip is reported, not applied
            if ((ws & 4u) != 0u && step == DUST_SPLIT_CLIP_APPLY) step = DUST_SPLIT_CLIP;
            WS_SET_PEND(uint(step));
            ws |= WS_PHASE;
            DUST_PROF_END(split)
        }
        if (ph == 2u) {
            // the sub-chord the step named (its own pass), then the bookkeeping of the step it is
            WS_SET_PEND(uint(dust_w_chord_apply(int(WS_PEND), pc.eyeLocal.xyz, ((ws & 2u) != 0u) && ((WS_PATH >> WS_LEVEL) & 1u) != 0u)));
        }
        if (ph == 3u) {
            // the Gaussian split's update (its own pass: the step's values are not carried into it),
            // then the bookkeeping of the split it is
            dust_w_gauss(((ws & 2u) != 0u) && ((WS_PATH >> WS_LEVEL) & 1u) != 0u);
            WS_SET_PEND(uint(DUST_SPLIT_SPLIT));
        }
        if (ph == 4u) {
            // the bookkeeping phase: the step's transitions, the leaf's hand-over, the next node
            int step = int(WS_PEND);
            ws &= ~WS_PHASE;
            bool done = false, draw = false;
            int drawAs = step;
            // the arm (dust_project_split_scatter's if-else chain, its conditions in its order, as
            // a value; the arms then exclusive ifs): 0 replay, 1 forced, 2 drop, 3 keep or volume,
            // 4 clip, 5 split
            uint arm = (ws & 2u) != 0u ? 0u
                     : ((ws & 4u) != 0u ? 1u
                     : (step == DUST_SPLIT_DROP ? 2u
                     : ((step == DUST_SPLIT_KEEP || step == DUST_SPLIT_VOLUME) ? 3u
                     : (step == DUST_SPLIT_CLIP ? 4u : 5u))));
            if (arm == 0u) {
                if (step == DUST_SPLIT_CLIP) {
                    ws |= 8u;
                } else if (step == DUST_SPLIT_SPLIT) {
                    WS_SET_LEVEL(WS_LEVEL + 1u);
                    ws &= ~8u;
                    ws = (ws & ~2u) | ((WS_LEVEL < WS_TARGET) ? 2u : 0u);
                } else {
                    // (unreachable, see above: ends the walk below this node, never loops)
                    ws &= ~2u; done = true;
                }
            }
            if (arm == 1u) {
                // the speck shell's forced split (dust::split_packet_with)
                ws &= ~4u;
                if (step == DUST_SPLIT_SPLIT) {
                    if ((ws & WS_COUNTING) == 0u && atomicAdd(pc.pyramid.v[DUST_PYR_SPLIT_CHILDREN], 2u) > DUST_FRAME_HARD_FACTOR * DUST_SPLIT_BUDGET && dust_frame_gates_on()) WS_SET_DMAX(0u);
                    WS_SET_PATH(WS_PATH & ~(1u << WS_LEVEL));
                    WS_SET_LEVEL(WS_LEVEL + 1u); ws &= ~8u;
                } else {
                    { float f_ = dust_w_shell_outside_fraction(pc.eyeLocal.xyz, shellDMax); WQ_W(12u) = floatBitsToUint(WQF(12u) * f_); }
                    draw = WQF(12u) > 0.0; drawAs = int(WS_NATURAL); done = true;
                }
            }
            if (arm == 2u) {
                // (the word's index + 0 through the state word: its counting and candidate bits are
                // never both set; a per-lane address, so the compiler's uniform-atomic reduction —
                // two linear VGPRs over the whole shader — is not used)
                if ((ws & WS_COUNTING) == 0u) atomicMax(pc.pyramid.v[DUST_PYR_SPLIT_DROPPED_FLUX + ((ws >> 30u) == 3u ? 1u : 0u)], floatBitsToUint(max(WQF(12u), 0.0)));
                done = true;
            }
            if (arm == 3u) {
                // the speck shell's hand-over (dust::split_packet_with): wholly inside, the specks'
                // (not drawn); across the boundary, split once more while the depth allows, else
                // drawn with the fraction of its mass outside
                int cls = dust_w_shell_classify(pc.eyeLocal.xyz, shellDMax);
                if (cls == DUST_SHELL_STRADDLES && WS_LEVEL < min(WS_DMAX, DUST_SPLIT_DEPTH_MAX)) {
                    ws |= 4u; WS_SET_NATURAL(uint(step));
                } else {
                    if (cls == DUST_SHELL_OUTSIDE) {
                        draw = true;
                    } else if (cls == DUST_SHELL_STRADDLES) {
                        { float f_ = dust_w_shell_outside_fraction(pc.eyeLocal.xyz, shellDMax); WQ_W(12u) = floatBitsToUint(WQF(12u) * f_); }
                        draw = WQF(12u) > 0.0;
                    }
                    done = true;
                }
            }
            if (arm == 4u) {
                ws |= 8u;
            }
            if (arm == 5u) {
                if ((ws & WS_COUNTING) == 0u && atomicAdd(pc.pyramid.v[DUST_PYR_SPLIT_CHILDREN], 2u) > DUST_FRAME_HARD_FACTOR * DUST_SPLIT_BUDGET && dust_frame_gates_on()) WS_SET_DMAX(0u);
                WS_SET_PATH(WS_PATH & ~(1u << WS_LEVEL));
                WS_SET_LEVEL(WS_LEVEL + 1u); ws &= ~8u;
            }
            // (the leaf is a candidate for the sequencer stage, in this order)
            if (draw) dust_w_cand_put(drawAs == DUST_SPLIT_VOLUME, (ws & WS_CAND_ON) != 0u);
            if (done) {
                // the next node: the b of the deepest split on the path whose a was taken
                uint pending = ~WS_PATH & ((1u << WS_LEVEL) - 1u);
                if (pending == 0u) {
                    WS_SET_IT(WS_IT + 1u); WS_SET_PATH(0u);
                } else {
                    uint L = uint(findMSB(pending));
                    WS_SET_PATH((WS_PATH & ((1u << L) - 1u)) | (1u << L));
                    WS_SET_TARGET(L + 1u);
                    ws |= 2u;
                }
                ws |= 1u;
                WS_SET_PEND(0u);
            }
        }
        WSTATE_W = ws;
    }
    // the record's count (word 46): the candidates counted (the counting pass), the run's length
    // (the writing pass: the counter from the record's base, word 45; 0 refused)
    dust_w_count_done();
    DUST_PROF_END(packet)
}
#undef kind
#undef nItems
#undef shellDMax
#endif  // !DUST_HARD_AXIS

#ifdef DUST_HARD_AXIS
// dust_project_split_scatter_one for every hard-axis sub-leaf (dust::hard_axis_leaf: bits 24-27 of
// DUST_PYR_SPLIT_DEPTH, diagnostics; n <= 1: the leaf itself): the Gaussianised axis segT drawn as
// n sub-leaves uniform over [-segT/2, segT/2], each with 1.21 (segT/2n)^2
void dust_project_split_scatter(vec3 mean, float S[6], vec3 d, float flux, float exposure,
                                bool openStart, bool openEnd, vec2 rho, vec3 segT, vec3 color, uint levels,
                                float W, float H, vec3 fwd, bool perspective, uint depthMax, uint volumeBias,
                                float shellDMax, uint volumeAxis) {
    uint n = (pc.pyramid.v[DUST_PYR_SPLIT_DEPTH] >> 24u) & 0xFu;
    vec3 h = 0.5 * segT;
    // one call site of dust_project_split_scatter_one: the driver inlines every call
    // site with its own copy of the function's locals, and two sites held two split stacks; the
    // whole leaf (n <= 1) is the loop's single pass with its arguments passed unchanged
    bool whole = n <= 1u || !(dot(h, h) > 0.0);
    uint subs = whole ? 1u : n;
    float fn = float(max(n, 1u));
    float Sh[6] = S;
    if (!whole) {
        float k = DUST_CHORD_VAR_FACTOR;
        float q = 1.0 - 1.0 / float(n * n);
        Sh[0] = S[0] - k * q * h.x * h.x; Sh[1] = S[1] - k * q * h.x * h.y; Sh[2] = S[2] - k * q * h.x * h.z;
        Sh[3] = S[3] - k * q * h.y * h.y; Sh[4] = S[4] - k * q * h.y * h.z; Sh[5] = S[5] - k * q * h.z * h.z;
    }
    for (uint jj = 0u; jj < subs; ++jj) {
        float off = -1.0 + float(2u * jj + 1u) / fn;
        dust_project_split_scatter_one(whole ? mean : mean + h * off, Sh, d, whole ? flux : flux / fn, exposure,
                                       openStart, openEnd, rho, whole ? segT : segT * (1.0 / fn), color, levels,
                                       W, H, fwd, perspective, depthMax, volumeBias, shellDMax, volumeAxis);
    }
}
#endif  // DUST_HARD_AXIS
#endif  // DUST_SPLIT_SEQ
#endif  // !DUST_SPLIT_PREP

