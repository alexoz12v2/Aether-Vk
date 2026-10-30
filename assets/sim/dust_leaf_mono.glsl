// @assets/sim/dust_leaf_mono.glsl
//
// The split path knob (AETHERVK_DUST_SPLIT_STAGED=0): the pre-split-family leaf path, the
// old dust_leaf.glsl verbatim, included only by dust_split_mono.comp. The one adaptation: its
// projection is dust_project_packet_mono (defined in dust_split_mono.comp: the old dust_common.glsl
// dust_project_packet, expression for expression; the split family re-wrote the current one to call
// dust_radiance_weight_o through a uvec2 address).
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
    DUST_PROF_BEGIN(vemit)
    if (dust_vqueue_on) {
        uint vbase = dust_vqueue_base(pc.pyramid);
        uint r = dust_vqueue_append(pc.pyramid, vbase);
        if (r != 0xFFFFFFFFu) {
            uint cap = dust_vqueue_capacity(pc.pyramid, vbase);
            DUST_VQUEUE_PUT(pc.pyramid, vbase, cap, r, p, seed, exposure, color, dust_phot_support, dust_phot_sun, dust_phot_rh)
            DUST_PROF_END(vemit)
            return;
        }
    }
    uint sbase = dust_vspill_base(pc.pyramid);
    uint rs = dust_vqueue_append(pc.pyramid, sbase);
    if (rs != 0xFFFFFFFFu) {
        uint scap = dust_vqueue_capacity(pc.pyramid, sbase);
        DUST_VQUEUE_PUT(pc.pyramid, sbase, scap, rs, p, seed, exposure, color, dust_phot_support, dust_phot_sun, dust_phot_rh)
    }
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

void scatter(DustScreenPacket P, vec3 color, uint levels) {
    DUST_PROF_BEGIN(scprep)
    // the peak is measured whatever the unit (a view fainter than the unit must still move it:
    // not_flow.rdc), before any texel test
    if (!dust_counting) atomicMax(pc.pyramid.v[DUST_PYR_TAU_MAX], floatBitsToUint(P.amp * P.wMean));
    uint level = dust_splat_level(P.sigmaMin, levels);
    float ex = DUST_SPLAT_SIGMAS * sqrt(P.cov.x), ey = DUST_SPLAT_SIGMAS * sqrt(P.cov.z);
    // the part of the segment the band can draw: inside the frame expanded by the band's margin
    // (dust::splat_scatter: the rows and columns clip the drawn texels; the estimate must too)
    float W = float(pc.pyramid.v[DUST_PYR_WIDTH]), H = float(pc.pyramid.v[DUST_PYR_HEIGHT]);
    float segLen = sqrt(P.seg.x * P.seg.x + P.seg.y * P.seg.y)
                 * dust_visible_fraction(P.mu, P.seg, ex, ey, W, H);
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
    if (level > levelValve) {
        float sz = float(1u << level);
        dust_widen_screen_packet(P, 0.25 * sz * sz);
    }
    uint off = pc.pyramid.v[DUST_PYR_TABLE + 3u * level];
    uint w = pc.pyramid.v[DUST_PYR_TABLE + 3u * level + 1u];
    uint h = pc.pyramid.v[DUST_PYR_TABLE + 3u * level + 2u];
    float s = float(1u << level);
    // a segment shorter than two texels at this level may miss every texel centre with open
    // ends: drawn closed (Gaussian caps), so its flux lands on the texels around it
    if (dot(P.seg, P.seg) < 4.0 * s * s) { P.openStart = false; P.openEnd = false; }
    DustBand B = dust_band_new(P);
    int y0, y1;
    if (!dust_band_rows(B, s, h, y0, y1)) { DUST_PROF_END(scprep) return; }
    if (dust_counting) { dust_seed_count++; DUST_PROF_END(scprep) return; }
    uint seed = dust_next_packet_seed();
    // a large packet goes to the queue (dust_draw_queue.comp: a workgroup per record of at most
    // DUST_QUEUE_RECORD_TEXELS texels, row ranges of the band); a full queue leaves the rest here
    uint qflags = pc.pyramid.v[DUST_PYR_FLAGS];
    float bandAt = dust_band_at(segLen, ex, ey, level);
    bool queued = (qflags & DUST_FRAME_QUEUE_OFF) == 0u
        && ((qflags & DUST_FRAME_QUEUE_ALL) != 0u || bandAt > DUST_QUEUE_MIN_TEXELS);
    int ya = y0;
    DUST_PROF_BEGIN(qput)
    if (queued) {
        uint base = dust_queue_base(pc.pyramid);
        float perRow = max(bandAt / float(y1 - y0 + 1), 1.0);
        int rowsPer = max(1, int(DUST_QUEUE_RECORD_TEXELS / perRow));
        for (; ya <= y1; ya += rowsPer) {
            uint at = dust_queue_append(pc.pyramid, base);
            if (at == 0xFFFFFFFFu) break;
            DUST_PROF_COUNT(records, 1u)
            dust_queue_put_ewa(pc.pyramid, at, level, seed, off, w, h, ya, min(ya + rowsPer - 1, y1), color, pc.invUnit, P);
        }
    }
    DUST_PROF_END(qput)
    DUST_PROF_END(scprep)
    // the rows left to this thread: drawn inline (dust_scatter_segment's arithmetic, segment for
    // segment; a struct argument per segment call cost the in-thread path 2-3x)
    DUST_PROF_BEGIN(ewa)
    bool any_ = false;
    float area = s * s;
    uint depthBits = floatBitsToUint(max(P.depthAu, 0.0));
    for (int ty = ya; ty <= y1; ++ty) {
        float yc = (float(ty) + 0.5) * s;
        int x0, x1;
        if (!dust_band_columns(B, yc, s, w, x0, x1)) continue;
        DUST_PROF_COUNT(ewa_texels, x1 - x0 + 1)
        float carry = 0.0;
        for (int tx = x0; tx <= x1; ++tx) {
            if (uint(tx - x0) % DUST_SEGMENT_TEXELS == 0u) carry = dust_segment_carry(seed, (uint(ty) << 16u) | uint(tx));
            vec2 x = vec2((float(tx) + 0.5) * s, yc);
            float tau = dust_capsule_tau(P, x) * area * pc.invUnit;
            if (P.albedo > 0.0) tau *= dust_texel_weight(P, x);
            if (!(tau >= 0.0) || isinf(tau)) continue;
            float tauC = min(tau, DUST_COUNT_MAX_PER_ADD) + carry;
            uint counts = uint(tauC + 0.5);
            carry = tauC - float(counts);
            if (counts == 0u) continue;
            uint i = off + (uint(ty) * w + uint(tx)) * DUST_PYRAMID_TEXEL_WORDS;
            dust_atomic_add_sat(pc.pyramid, i, counts);
            float hf = float(dust_hash32(i) >> 8u) * (1.0 / 16777216.0);
            uvec3 n = uvec3(float(counts) * color * DUST_COLOR_SCALE + hf);
            dust_atomic_add_sat(pc.pyramid, i + 1u, n.r);
            dust_atomic_add_sat(pc.pyramid, i + 2u, n.g);
            dust_atomic_add_sat(pc.pyramid, i + 3u, n.b);
            uint cur = pc.pyramid.v[i + 4u];
            if (cur == 0u || depthBits < cur) {
                uint seen = atomicCompSwap(pc.pyramid.v[i + 4u], 0u, depthBits);
                if (seen != 0u) atomicMin(pc.pyramid.v[i + 4u], depthBits);
            }
            any_ = true;
        }
    }
    DUST_PROF_END(ewa)
    if (any_) atomicOr(pc.pyramid.v[DUST_PYR_LEVEL_MASK], 1u << level);
    // the white-point measurement grid is derived from the finished pyramid (dust_measure.comp)
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
            dust_volume_emit(p0, exposure, color, levels, W, H, volumeBias, volumeAxis);
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
    uint target = 0u;                // replaying: the level of the b being recomputed
    bool load = true;                // (re)load item it's root
    bool replay = false;             // the node is on the path to that b
    bool forced = false;             // the node's shell-forced split is next
    int natural = DUST_SPLIT_KEEP;   // a forced node's first step (KEEP or VOLUME)
    DustPacket q;
    uint level = 0u;
    bool clipped = false;
    while (it < nItems) {
        if (load) {
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
            level = 0u;
            clipped = false;
            load = false;
            // an empty piece: the next item (a replayed root loaded on its first visit: never here)
            if (!ok) { it++; path = 0u; replay = false; load = true; }
        } else {
            DUST_PROF_BEGIN(split)
            bool first = !replay && !forced;
            DUST_PROF_COUNT(nodes, first ? 1u : 0u)
            // (q becomes the child the walk takes — the second on a replay whose path
            // bit is set — on a clip or a split, and stays itself otherwise)
            int step = dust_split_step_one(q, clipped, level, first ? dMax : DUST_SPLIT_DEPTH_MAX, pc.eyeLocal.xyz, fwd, perspective, !first, forced,
                                           replay && ((path >> level) & 1u) != 0u);
            bool done = false, draw = false;
            int drawAs = step;
            if (replay) {
                if (step == DUST_SPLIT_CLIP) {
                    clipped = true;
                } else if (step == DUST_SPLIT_SPLIT) {
                    level++;
                    clipped = false;
                    replay = level < target;
                } else {
                    // (unreachable, see above: ends the walk below this node, never loops)
                    replay = false; done = true;
                }
            } else if (forced) {
                // the speck shell's forced split (dust::split_packet_with)
                forced = false;
                if (step == DUST_SPLIT_SPLIT) {
                    if (!dust_counting && atomicAdd(pc.pyramid.v[DUST_PYR_SPLIT_CHILDREN], 2u) > DUST_FRAME_HARD_FACTOR * DUST_SPLIT_BUDGET && dust_frame_gates_on()) dMax = 0u;
                    path &= ~(1u << level);
                    level++; clipped = false;
                } else {
                    q.flux *= dust_shell_outside_fraction(q, pc.eyeLocal.xyz, shellDMax);
                    draw = q.flux > 0.0; drawAs = natural; done = true;
                }
            } else if (step == DUST_SPLIT_DROP) {
                if (!dust_counting) atomicMax(pc.pyramid.v[DUST_PYR_SPLIT_DROPPED_FLUX], floatBitsToUint(max(q.flux, 0.0)));
                done = true;
            } else if (step == DUST_SPLIT_KEEP || step == DUST_SPLIT_VOLUME) {
                // the speck shell's hand-over (dust::split_packet_with): wholly inside, the specks'
                // (not drawn); across the boundary, split once more while the depth allows, else
                // drawn with the fraction of its mass outside
                int cls = dust_shell_classify(q, pc.eyeLocal.xyz, shellDMax);
                if (cls == DUST_SHELL_STRADDLES && level < min(dMax, DUST_SPLIT_DEPTH_MAX)) {
                    forced = true; natural = step;
                } else {
                    if (cls == DUST_SHELL_OUTSIDE) {
                        draw = true;
                    } else if (cls == DUST_SHELL_STRADDLES) {
                        q.flux *= dust_shell_outside_fraction(q, pc.eyeLocal.xyz, shellDMax);
                        draw = q.flux > 0.0;
                    }
                    done = true;
                }
            } else if (step == DUST_SPLIT_CLIP) {
                clipped = true;
            } else {
                if (!dust_counting && atomicAdd(pc.pyramid.v[DUST_PYR_SPLIT_CHILDREN], 2u) > DUST_FRAME_HARD_FACTOR * DUST_SPLIT_BUDGET && dust_frame_gates_on()) dMax = 0u;
                path &= ~(1u << level);
                level++; clipped = false;
            }
            if (draw) {
                if (drawAs == DUST_SPLIT_VOLUME) {
                    dust_volume_emit(q, exposure, color, levels, W, H, volumeBias, volumeAxis);
                } else {
                    DustScreenPacket SP;
                    DUST_PROF_BEGIN(project)
                    bool on = dust_project_packet_mono(q.mean, q.S, q.seg, q.flux, exposure, pc.mvp, W, H, pc.eyeLocal.xyz, q.openStart, q.openEnd, q.rho, SP);
                    DUST_PROF_END(project)
                    if (on) {
                        DUST_PROF_COUNT(leaves, 1u)
                        scatter(SP, color, levels);
                    }
                }
            }
            if (done) {
                // the next node: the b of the deepest split on the path whose a was taken
                uint pending = ~path & ((1u << level) - 1u);
                if (pending == 0u) {
                    it++; path = 0u;
                } else {
                    uint L = uint(findMSB(pending));
                    path = (path & ((1u << L) - 1u)) | (1u << L);
                    target = L + 1u;
                    replay = true;
                }
                load = true;
            }
            DUST_PROF_END(split)
        }
    }
    dust_phot_support = sup;
    dust_phot_has_origin = hasO;
    DUST_PROF_END(packet)
}

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

