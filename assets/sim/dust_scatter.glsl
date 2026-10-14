// @assets/sim/dust_scatter.glsl
//
// The EWA capsule scatter shared by dust_splat.comp (a packet drawn by its cluster's thread) and
// dust_draw_queue.comp (a large packet drawn by a whole workgroup, dust::DUST_QUEUE_*): a packet's
// texels are drawn in row segments of DUST_SEGMENT_TEXELS columns, each with its own dither carry
// from a hashed phase (dust::segment_carry), so the counts do not depend on the thread drawing a
// segment or the order. Mirror of `dust::scatter_into`. Include after dust_common.glsl.

// One row segment [xa, xb] of row ty of P into the grid of w texels of s px at word off; `full`
// adds the colour and the nearest depth too. Whether a texel was written.
bool dust_scatter_segment(DustUintBuffer pyr, float invUnit, DustScreenPacket P, vec3 color, float s,
                          uint off, uint w, uint seed, int ty, int xa, int xb, bool full) {
    float yc = (float(ty) + 0.5) * s;
    float area = s * s;
    uint depthBits = floatBitsToUint(max(P.depthAu, 0.0));
    bool any_ = false;
    float carry = dust_segment_carry(seed, (uint(ty) << 16u) | uint(xa));
    for (int tx = xa; tx <= xb; ++tx) {
        vec2 x = vec2((float(tx) + 0.5) * s, yc);
        float tau = dust_capsule_tau(P, x) * area * invUnit;
        if (P.albedo > 0.0) tau *= dust_texel_weight(P, x);
        if (!(tau >= 0.0) || isinf(tau)) continue;
        float tauC = min(tau, DUST_COUNT_MAX_PER_ADD) + carry;
        uint counts = uint(tauC + 0.5);
        carry = tauC - float(counts);
        if (counts == 0u) continue;
        uint i = off + (uint(ty) * w + uint(tx)) * DUST_PYRAMID_TEXEL_WORDS;
        dust_atomic_add_sat(pyr, i, counts);
        if (full) {
            // colour words floor(counts * c + h), h a hashed threshold per texel, an ordered
            // dither (dust::scatter_into): rounding counts * c + 0.5 per deposit turned every
            // faint texel into (1, 1, 0)
            float hf = float(dust_hash32(i) >> 8u) * (1.0 / 16777216.0);
            uvec3 n = uvec3(float(counts) * color * DUST_COLOR_SCALE + hf);
            dust_atomic_add_sat(pyr, i + 1u, n.r);
            dust_atomic_add_sat(pyr, i + 2u, n.g);
            dust_atomic_add_sat(pyr, i + 3u, n.b);
            // nearest dust: 0 means "no dust yet" (positive f32 bits order as uints)
            uint cur = pyr.v[i + 4u];
            if (cur == 0u || depthBits < cur) {
                uint seen = atomicCompSwap(pyr.v[i + 4u], 0u, depthBits);
                if (seen != 0u) atomicMin(pyr.v[i + 4u], depthBits);
            }
        }
        any_ = true;
    }
    return any_;
}

// The segments of rows [rowA, rowB] of P's band numbered in row order: those with
// (number mod stride) == first are drawn (stride 1, first 0: every segment, one thread; stride
// 64, first = the invocation: a workgroup's share). Whether a texel was written.
bool dust_scatter_rows(DustUintBuffer pyr, float invUnit, DustScreenPacket P, vec3 color, float s,
                       uint off, uint w, uint h, uint seed, int rowA, int rowB, bool full,
                       uint stride, uint first) {
    DustBand B = dust_band_new(P);
    int y0, y1;
    if (!dust_band_rows(B, s, h, y0, y1)) return false;
    y0 = max(y0, rowA);
    y1 = min(y1, rowB);
    bool any_ = false;
    uint number = 0u;
    for (int ty = y0; ty <= y1; ++ty) {
        int x0, x1;
        if (!dust_band_columns(B, (float(ty) + 0.5) * s, s, w, x0, x1)) continue;
        uint nseg = uint(x1 - x0) / DUST_SEGMENT_TEXELS + 1u;
        // the first segment of this row that is ours
        uint j = (first + stride - number % stride) % stride;
        for (; j < nseg; j += stride) {
            int xa = x0 + int(j * DUST_SEGMENT_TEXELS);
            int xb = min(xa + int(DUST_SEGMENT_TEXELS) - 1, x1);
            any_ = dust_scatter_segment(pyr, invUnit, P, color, s, off, w, seed, ty, xa, xb, full) || any_;
        }
        number += nseg;
    }
    return any_;
}

// One Gaussian piece of a volumetric packet (dust::VolumePiece)
struct DustVolumePiece {
    float l[6];   // Cholesky factor of its covariance
    vec3  qw;     // L^-1 (eye - mean)
    float k;      // exposure * flux / (2 pi sqrt(det S))
    vec4  bbox;   // projected 3-sigma box (px): x0 x1 y0 y1
    bool  full;   // the whole frame (the eye near it, or a box corner not in front of the eye)
    bool  on;     // on screen
};

// The n pieces of a volumetric packet (dust::volume_pieces): covariance S (pixel filter
// included), piece variance cvar of the spacing^2
// (the axes: base point axMean and covariance S, axis a = segA (the chord, rho ramp, from the base
// point) with nA pieces of variance cvarA spacing^2, axis b = segB (dust::Packet::seg_t, uniform,
// centred) with nB pieces when hasB; piece i * nB + j; without axis b the chord-only arithmetic)
// (the view: the mvp's rows r0, r1, r3 and translations t = (t0, t1, t3), the eye; the packet's
// flux and density ramp rho)
DustVolumePiece dust_volume_piece(uint k, float flux, vec2 rho, float exposure, vec3 axMean, float S[6], vec3 segA,
                                  uint nA, float cvarA, vec3 segB, uint nB, float cvarB, bool hasB, float W, float H,
                                  vec3 r0, vec3 r1, vec3 r3, vec3 t, vec3 eye) {
    float t0 = t.x, t1 = t.y, t3 = t.z;
    float cvar = cvarA;
    float invN = 1.0 / float(nA);
    vec3 h = segA * invN;
    float invB = 1.0 / float(nB);
    vec3 hb = segB * invB;
    uint i = k / nB, j = k % nB;
    float uk = (float(i) + 0.5) * invN;
    vec3 meanK = axMean + segA * uk;
    float Sk[6];
    Sk[0] = S[0] + cvar * h.x * h.x;
    Sk[1] = S[1] + cvar * h.x * h.y;
    Sk[2] = S[2] + cvar * h.x * h.z;
    Sk[3] = S[3] + cvar * h.y * h.y;
    Sk[4] = S[4] + cvar * h.y * h.z;
    Sk[5] = S[5] + cvar * h.z * h.z;
    float fluxK = flux * (rho.x + (rho.y - rho.x) * uk) * invN;
    if (hasB) {
        float vk = (float(j) + 0.5) * invB - 0.5;
        meanK = meanK + segB * vk;
        Sk[0] += cvarB * hb.x * hb.x;
        Sk[1] += cvarB * hb.x * hb.y;
        Sk[2] += cvarB * hb.x * hb.z;
        Sk[3] += cvarB * hb.y * hb.y;
        Sk[4] += cvarB * hb.y * hb.z;
        Sk[5] += cvarB * hb.z * hb.z;
        fluxK *= invB;
    }
    float Lk[6];
    dust_cholesky3(Sk, Lk);
    vec3 qw = dust_cholesky_solve3(Lk, eye - meanK);
    bool full = dot(qw, qw) < DUST_VOLUME_SIGMAS * DUST_VOLUME_SIGMAS;
    vec3 ext = vec3(3.0 * sqrt(Sk[0]), 3.0 * sqrt(Sk[3]), 3.0 * sqrt(Sk[5]));
    float x0 = 1e30, x1 = -1e30, y0 = 1e30, y1 = -1e30;
    if (!full) {
        for (uint c = 0u; c < 8u; ++c) {
            vec3 pt = vec3(meanK.x + ((c & 1u) != 0u ? ext.x : -ext.x),
                           meanK.y + ((c & 2u) != 0u ? ext.y : -ext.y),
                           meanK.z + ((c & 4u) != 0u ? ext.z : -ext.z));
            float w = dot(r3, pt) + t3;
            if (!(w > 0.0)) { full = true; break; }
            float px = ((dot(r0, pt) + t0) / w * 0.5 + 0.5) * W;
            float py = ((dot(r1, pt) + t1) / w * 0.5 + 0.5) * H;
            x0 = min(x0, px); x1 = max(x1, px);
            y0 = min(y0, py); y1 = max(y1, py);
        }
    }
    DustVolumePiece q;
    q.l = Lk;
    q.qw = qw;
    q.k = exposure * fluxK / (2.0 * DUST_PI_F32 * (Lk[0] * Lk[3] * Lk[5]));
    q.bbox = vec4(x0, x1, y0, y1);
    q.full = full;
    q.on = full || !(x1 < 0.0 || y1 < 0.0 || x0 > W || y0 > H || !(x0 <= x1) || !(y0 <= y1));
    return q;
}

void dust_volume_pieces(float flux, vec2 rho, float exposure, vec3 axMean, float S[6], vec3 segA, uint nA, float cvarA,
                        vec3 segB, uint nB, float cvarB, bool hasB, float W, float H,
                        vec3 r0, vec3 r1, vec3 r3, vec3 t, vec3 eye,
                        inout DustVolumePiece pieces[DUST_VOLUME_PIECES_MAX]) {
    for (uint k = 0u; k < nA * nB; ++k) {
        pieces[k] = dust_volume_piece(k, flux, rho, exposure, axMean, S, segA, nA, cvarA, segB, nB, cvarB, hasB, W, H,
                                      r0, r1, r3, t, eye);
    }
}

// texel range of a piece at a level of lw x lh texels (dust::volume_range): empty off screen, the
// whole grid when full
ivec4 dust_volume_range(DustVolumePiece q, uint level, uint lw, uint lh) {
    if (!q.on) return ivec4(0, -1, 0, -1);
    if (q.full) return ivec4(0, int(lw) - 1, 0, int(lh) - 1);
    float sz = float(1u << level);
    return ivec4(int(max(floor(q.bbox.x / sz), 0.0)), int(min(ceil(q.bbox.y / sz), float(lw - 1u))),
                 int(max(floor(q.bbox.z / sz), 0.0)), int(min(ceil(q.bbox.w / sz), float(lh - 1u))));
}

// One row segment [xa, xb] of row ty of a volumetric piece at one level (dust::volume_scatter,
// DUST_SEGMENT_TEXELS): the exact ray integral at each texel centre, from the segment's own
// dither phase; renorm = the support's flux over its kept mass, meanRef = the packet's mean (the
// photometry), sz = the texel size (px). The texels written; the largest tau per px^2 into peak.
// Timing probe 2 (no_volume_writes) evaluates without writing.
bool dust_volume_segment(DustUintBuffer pyr, float invUnit, DustVolumePiece pc_, float renorm, vec3 meanRef,
                         vec3 eye, vec3 r0, vec3 r1, vec3 r3, float W, float H, float sz, vec3 color,
                         uint depthBits, uint off, uint lw, uint seed, int ty, int xa, int xb, inout float peak) {
    bool noWrites = ((dust_pyr_split_word >> 30u) & 3u) == 2u;
    float area = sz * sz;
    float yc = (float(ty) + 0.5) * sz;
    float ny = 2.0 * yc / H - 1.0;
    vec3 bRow = r1 - r3 * ny;
    float carry = dust_segment_carry(seed, (uint(ty) << 16u) | uint(xa));
    bool any_ = false;
    for (int tx = xa; tx <= xb; ++tx) {
        float xc = (float(tx) + 0.5) * sz;
        float nx = 2.0 * xc / W - 1.0;
        vec3 aRow = r0 - r3 * nx;
        // the ray through the texel centre: on both planes of constant ndc, in front of the eye
        precise vec3 v = cross(aRow, bRow);
        if (dot(v, r3) < 0.0) v = -v;
        float vn = sqrt(dot(v, v));
        if (!(vn > 0.0)) continue;
        vec3 u = v * (1.0 / vn);
        vec3 uw = dust_cholesky_solve3(pc_.l, u);
        float a = dot(uw, uw);
        float b = dot(uw, pc_.qw);
        precise vec3 rr = pc_.qw - uw * (b / a);
        float sa = sqrt(a);
        float tauPx = pc_.k * renorm / sa * exp(-0.5 * dot(rr, rr))
                    * dust_ray_kept(a, b, eye, u, dust_phot_support);
        if (dust_phot_albedo > 0.0) {
            vec3 x = eye + u * max(-b / a, 0.0);
            tauPx *= dust_volume_weight(meanRef, x, u);
        }
        if (!(tauPx >= 0.0) || isinf(tauPx)) continue;
        peak = max(peak, tauPx);
        float tau = tauPx * area * invUnit;
        float tauC = min(tau, DUST_COUNT_MAX_PER_ADD) + carry;
        uint counts = uint(tauC + 0.5);
        carry = tauC - float(counts);
        if (counts == 0u || noWrites) continue;
        uint i = off + (uint(ty) * lw + uint(tx)) * DUST_PYRAMID_TEXEL_WORDS;
        dust_atomic_add_sat(pyr, i, counts);
        float hf = float(dust_hash32(i) >> 8u) * (1.0 / 16777216.0);
        uvec3 cw = uvec3(float(counts) * color * DUST_COLOR_SCALE + hf);
        dust_atomic_add_sat(pyr, i + 1u, cw.r);
        dust_atomic_add_sat(pyr, i + 2u, cw.g);
        dust_atomic_add_sat(pyr, i + 3u, cw.b);
        uint cur = pyr.v[i + 4u];
        if (cur == 0u || depthBits < cur) {
            uint seen = atomicCompSwap(pyr.v[i + 4u], 0u, depthBits);
            if (seen != 0u) atomicMin(pyr.v[i + 4u], depthBits);
        }
        any_ = true;
    }
    return any_;
}


// The plan of a volumetric packet before its pieces (dust::volume_plan_pieces; shared by the
// splat's in-thread draw and the queue's workgroups): the axes, the pieces per axis and the level
// from the angular sigma. false: no projection (the packet is not drawn).
struct DustVolumePlan {
    vec3  axMean;
    float axS[6];
    vec3  segA;
    vec3  segB;
    bool  hasB;
    uint  nA;
    uint  nB;
    float cvarA;
    float cvarB;
    uint  level;
};
bool dust_volume_plan_axes(DustPacket p, vec3 r0, vec3 r1, vec3 r3, vec3 eye, float W, float H, uint levels,
                           uint axis, out DustVolumePlan pl) {
    float n3 = dot(r3, r3);
    if (!(n3 > 0.0)) return false;
    float fPx = 0.5 * min(W * sqrt(dot(r0, r0)), H * sqrt(dot(r1, r1))) / sqrt(n3);
    if (!(fPx > 0.0)) return false;
    vec3 q0 = eye - p.mean;
    vec3 rSeg = q0 - p.seg * dust_segment_param(q0, p.seg);
    float dist = sqrt(dot(rSeg, rSeg));
    float rel = dist / fPx;
    float varF = DUST_PIXEL_FILTER_VAR * rel * rel;
    float S[6];
    S[0] = p.S[0] + varF; S[1] = p.S[1]; S[2] = p.S[2];
    S[3] = p.S[3] + varF; S[4] = p.S[4]; S[5] = p.S[5] + varF;
    float L[6];
    dust_cholesky3(S, L);
    // the axes (dust::volume_plan_pieces): the chord only (default), or seg_t as uniform pieces
    // instead of its 1.21 Gaussian (time: the chord then Gaussianised; grid: both gridded)
    uint mode = (axis != DUST_VOLUME_AXIS_CHORD && dot(p.segT, p.segT) > 0.0) ? axis : DUST_VOLUME_AXIS_CHORD;
    float kf = DUST_CHORD_VAR_FACTOR;
    vec3 ht = p.segT * 0.5;
    float lessT[6];
    lessT[0] = S[0] - kf * ht.x * ht.x;
    lessT[1] = S[1] - kf * ht.x * ht.y;
    lessT[2] = S[2] - kf * ht.x * ht.z;
    lessT[3] = S[3] - kf * ht.y * ht.y;
    lessT[4] = S[4] - kf * ht.y * ht.z;
    lessT[5] = S[5] - kf * ht.z * ht.z;
    vec3 axMean = p.mean;
    float axS[6];
    vec3 segA = p.seg, segB = vec3(0.0);
    bool hasB = false;
    if (mode == DUST_VOLUME_AXIS_TIME) {
        // the chord as its 1.21 Gaussian about its rho-weighted centre
        vec3 hc = p.seg * 0.5;
        float ubar = p.rho.x * 0.5 + (p.rho.y - p.rho.x) / 3.0;
        axMean = p.mean + p.seg * ubar;
        axS[0] = lessT[0] + kf * hc.x * hc.x;
        axS[1] = lessT[1] + kf * hc.x * hc.y;
        axS[2] = lessT[2] + kf * hc.x * hc.z;
        axS[3] = lessT[3] + kf * hc.y * hc.y;
        axS[4] = lessT[4] + kf * hc.y * hc.z;
        axS[5] = lessT[5] + kf * hc.z * hc.z;
        segA = vec3(0.0);
        segB = p.segT;
        hasB = true;
    } else if (mode == DUST_VOLUME_AXIS_GRID) {
        axS = lessT;
        segB = p.segT;
        hasB = true;
    } else {
        axS = S;
    }
    // pieces per axis: one per DUST_VOLUME_PIECE_SIGMAS Mahalanobis sigma of the base covariance
    float Lb[6];
    dust_cholesky3(axS, Lb);
    vec3 dw = dust_cholesky_solve3(Lb, segA);
    float lenW = sqrt(dot(dw, dw)) / DUST_VOLUME_PIECE_SIGMAS;
    bool capped = lenW > float(DUST_VOLUME_PIECES_MAX);
    uint nA = capped ? DUST_VOLUME_PIECES_MAX : max(uint(ceil(lenW)), 1u);
    float cvarA = capped ? DUST_VOLUME_PIECE_VAR_CAPPED : 1.0 / 12.0;
    uint nB = 1u;
    float cvarB = 1.0 / 12.0;
    if (hasB) {
        vec3 dwb = dust_cholesky_solve3(Lb, segB);
        float lenB = sqrt(dot(dwb, dwb)) / DUST_VOLUME_PIECE_SIGMAS;
        bool cappedB = lenB > float(DUST_VOLUME_PIECES_MAX);
        nB = cappedB ? DUST_VOLUME_PIECES_MAX : max(uint(ceil(lenB)), 1u);
        cvarB = cappedB ? DUST_VOLUME_PIECE_VAR_CAPPED : 1.0 / 12.0;
    }
    // at most DUST_VOLUME_PIECES_MAX pieces in all: fewer along seg_t first
    while (nA * nB > DUST_VOLUME_PIECES_MAX) {
        if (nB > 1u) { nB = (nB + 1u) / 2u; cvarB = DUST_VOLUME_PIECE_VAR_CAPPED; }
        else { nA = (nA + 1u) / 2u; cvarA = DUST_VOLUME_PIECE_VAR_CAPPED; }
    }
    // the level from the angular sigma
    float tr = S[0] + S[3] + S[5];
    float sigSmall = 2.0 * (L[0] * L[3] * L[5]) / tr;
    float den = max(dist, sigSmall);
    float theta = den > 0.0 ? sigSmall / den : 1.0;
    uint level = dust_volume_level(theta * fPx, levels);
    pl.axMean = axMean;
    pl.axS = axS;
    pl.segA = segA;
    pl.segB = segB;
    pl.hasB = hasB;
    pl.nA = nA;
    pl.nB = nB;
    pl.cvarA = cvarA;
    pl.cvarB = cvarB;
    pl.level = level;
    return true;
}

// ── The packet queue (dust::DUST_QUEUE_*) ──────────────────────────────────────────────────────
// It lives in the pyramid buffer after the measurement grid (PyramidLayout::total_words): a
// header of DUST_QUEUE_HEADER_WORDS words — the indirect dispatch's (x, 1, 1), the raw append
// counter (x = min(raw, capacity): raw − x packets overflowed and were drawn in-thread), a
// capacity override (0 = DUST_QUEUE_CAPACITY) — then DUST_QUEUE_RECORD_WORDS-word records.
const uint DUST_QUEUE_HEADER_WORDS = 8u;
const uint DUST_QUEUE_RECORD_WORDS = 64u;
const uint DUST_QUEUE_CAPACITY = 262144u;
// a packet whose band holds more texels than this is queued (dust::DUST_QUEUE_MIN_TEXELS)
const float DUST_QUEUE_MIN_TEXELS = 64.0;
// texels one record covers at most: a larger packet is queued as several row ranges
const float DUST_QUEUE_RECORD_TEXELS = 4096.0;
// record kinds (word 0, bits 30-31)
const uint DUST_QUEUE_EWA = 0u;
// PYR_FLAGS bits (host-written, dust::DUST_FRAME_QUEUE_OFF / _ALL): no queue / queue every packet
const uint DUST_FRAME_QUEUE_OFF = 1u << 28u;
const uint DUST_FRAME_QUEUE_ALL = 1u << 29u;

// the queue's first word (after the measurement grid)
uint dust_queue_base(DustUintBuffer pyr) {
    uint m = pyr.v[DUST_PYR_MEASURE];
    return m + pyr.v[DUST_PYR_MEASURE + 1u] * pyr.v[DUST_PYR_MEASURE + 2u] * DUST_PYRAMID_TEXEL_WORDS;
}

// the queue's capacity: DUST_QUEUE_CAPACITY, or less when header word 4 says so (a test hook,
// dust::Device::dust_set_queue_capacity; 0 = the default)
uint dust_queue_capacity(DustUintBuffer pyr, uint base) {
    uint c = pyr.v[base + 4u];
    return c == 0u ? DUST_QUEUE_CAPACITY : min(c, DUST_QUEUE_CAPACITY);
}

// A record slot, or ~0u when the queue is full (counted: raw counter past the capacity)
uint dust_queue_append(DustUintBuffer pyr, uint base) {
    uint r = atomicAdd(pyr.v[base + 3u], 1u);
    if (r >= dust_queue_capacity(pyr, base)) return 0xFFFFFFFFu;
    atomicMax(pyr.v[base], r + 1u);
    return base + DUST_QUEUE_HEADER_WORDS + r * DUST_QUEUE_RECORD_WORDS;
}

void dust_queue_put_f(DustUintBuffer pyr, uint at, float v) { pyr.v[at] = floatBitsToUint(v); }
float dust_queue_get_f(DustUintBuffer pyr, uint at) { return uintBitsToFloat(pyr.v[at]); }

// EWA record: 0 kind | level, 1 seed, 2 off, 3 w, 4 h, 5 rowA, 6 rowB (int bits), 7-9 colour,
// 10 invUnit, 11-44 the screen packet (34 words)
void dust_queue_put_ewa(DustUintBuffer pyr, uint at, uint level, uint seed, uint off, uint w, uint h,
                        int rowA, int rowB, vec3 color, float invUnit, DustScreenPacket P) {
    pyr.v[at] = (DUST_QUEUE_EWA << 30u) | level;
    pyr.v[at + 1u] = seed;
    pyr.v[at + 2u] = off;
    pyr.v[at + 3u] = w;
    pyr.v[at + 4u] = h;
    pyr.v[at + 5u] = uint(rowA);
    pyr.v[at + 6u] = uint(rowB);
    dust_queue_put_f(pyr, at + 7u, color.x);
    dust_queue_put_f(pyr, at + 8u, color.y);
    dust_queue_put_f(pyr, at + 9u, color.z);
    dust_queue_put_f(pyr, at + 10u, invUnit);
    uint k = at + 11u;
    dust_queue_put_f(pyr, k, P.mu.x); dust_queue_put_f(pyr, k + 1u, P.mu.y);
    dust_queue_put_f(pyr, k + 2u, P.cov.x); dust_queue_put_f(pyr, k + 3u, P.cov.y); dust_queue_put_f(pyr, k + 4u, P.cov.z);
    dust_queue_put_f(pyr, k + 5u, P.seg.x); dust_queue_put_f(pyr, k + 6u, P.seg.y);
    dust_queue_put_f(pyr, k + 7u, P.amp);
    dust_queue_put_f(pyr, k + 8u, P.sigmaMin);
    dust_queue_put_f(pyr, k + 9u, P.det);
    dust_queue_put_f(pyr, k + 10u, P.depthAu);
    pyr.v[k + 11u] = (P.openStart ? 1u : 0u) | (P.openEnd ? 2u : 0u);
    dust_queue_put_f(pyr, k + 12u, P.rho.x); dust_queue_put_f(pyr, k + 13u, P.rho.y);
    dust_queue_put_f(pyr, k + 14u, P.mean.x); dust_queue_put_f(pyr, k + 15u, P.mean.y); dust_queue_put_f(pyr, k + 16u, P.mean.z);
    dust_queue_put_f(pyr, k + 17u, P.kx.x); dust_queue_put_f(pyr, k + 18u, P.kx.y); dust_queue_put_f(pyr, k + 19u, P.kx.z);
    dust_queue_put_f(pyr, k + 20u, P.ky.x); dust_queue_put_f(pyr, k + 21u, P.ky.y); dust_queue_put_f(pyr, k + 22u, P.ky.z);
    dust_queue_put_f(pyr, k + 23u, P.sun.x); dust_queue_put_f(pyr, k + 24u, P.sun.y); dust_queue_put_f(pyr, k + 25u, P.sun.z);
    dust_queue_put_f(pyr, k + 26u, P.rhAu);
    dust_queue_put_f(pyr, k + 27u, P.eye.x); dust_queue_put_f(pyr, k + 28u, P.eye.y);
    dust_queue_put_f(pyr, k + 29u, P.eye.z); dust_queue_put_f(pyr, k + 30u, P.eye.w);
    dust_queue_put_f(pyr, k + 31u, P.albedo);
    dust_queue_put_f(pyr, k + 32u, P.wMean);
}

DustScreenPacket dust_queue_get_screen_packet(DustUintBuffer pyr, uint k) {
    DustScreenPacket P;
    P.mu = vec2(dust_queue_get_f(pyr, k), dust_queue_get_f(pyr, k + 1u));
    P.cov = vec3(dust_queue_get_f(pyr, k + 2u), dust_queue_get_f(pyr, k + 3u), dust_queue_get_f(pyr, k + 4u));
    P.seg = vec2(dust_queue_get_f(pyr, k + 5u), dust_queue_get_f(pyr, k + 6u));
    P.amp = dust_queue_get_f(pyr, k + 7u);
    P.sigmaMin = dust_queue_get_f(pyr, k + 8u);
    P.det = dust_queue_get_f(pyr, k + 9u);
    P.depthAu = dust_queue_get_f(pyr, k + 10u);
    uint open = pyr.v[k + 11u];
    P.openStart = (open & 1u) != 0u;
    P.openEnd = (open & 2u) != 0u;
    P.rho = vec2(dust_queue_get_f(pyr, k + 12u), dust_queue_get_f(pyr, k + 13u));
    P.mean = vec3(dust_queue_get_f(pyr, k + 14u), dust_queue_get_f(pyr, k + 15u), dust_queue_get_f(pyr, k + 16u));
    P.kx = vec3(dust_queue_get_f(pyr, k + 17u), dust_queue_get_f(pyr, k + 18u), dust_queue_get_f(pyr, k + 19u));
    P.ky = vec3(dust_queue_get_f(pyr, k + 20u), dust_queue_get_f(pyr, k + 21u), dust_queue_get_f(pyr, k + 22u));
    P.sun = vec3(dust_queue_get_f(pyr, k + 23u), dust_queue_get_f(pyr, k + 24u), dust_queue_get_f(pyr, k + 25u));
    P.rhAu = dust_queue_get_f(pyr, k + 26u);
    P.eye = vec4(dust_queue_get_f(pyr, k + 27u), dust_queue_get_f(pyr, k + 28u),
                 dust_queue_get_f(pyr, k + 29u), dust_queue_get_f(pyr, k + 30u));
    P.albedo = dust_queue_get_f(pyr, k + 31u);
    P.wMean = dust_queue_get_f(pyr, k + 32u);
    return P;
}
