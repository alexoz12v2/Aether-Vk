// @assets/sim/dust_scatter.glsl
//
// The EWA capsule scatter shared by dust_splat.comp (a packet drawn by its cluster's thread) and
// dust_draw_queue.comp (a large packet drawn by a whole workgroup, dust::DUST_QUEUE_*): a packet's
// texels are drawn in row segments of DUST_SEGMENT_TEXELS columns, each with its own dither carry
// from a hashed phase (dust::segment_carry), so the counts do not depend on the thread drawing a
// segment or the order. Mirror of `dust::scatter_into`. Include after dust_common.glsl.
#include "dust_profile.glsl"

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
    DUST_PROF_BEGIN(qrows)
    DustBand B = dust_band_new(P);
    int y0, y1;
    if (!dust_band_rows(B, s, h, y0, y1)) { DUST_PROF_END(qrows) return false; }
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
            DUST_PROF_BEGIN_AS(seg)
            any_ = dust_scatter_segment(pyr, invUnit, P, color, s, off, w, seed, ty, xa, xb, full) || any_;
            DUST_PROF_END_AS(qseg, seg)
            DUST_PROF_COUNT(qtexels, xb - xa + 1)
        }
        number += nseg;
    }
    DUST_PROF_END(qrows)
    return any_;
}

// The same rows drawn by a whole workgroup (dust_draw_queue.comp defines DUST_QUEUE_WORKGROUP):
// the rows' column ranges are computed once, a lane per row, into shared memory in chunks of
// DUST_QUEUE_ROW_CHUNK rows, the segment counts prefix-summed, and each lane draws the segments
// numbered lane, lane + lanes, ... of the chunk, finding their rows in the prefix. The segments
// (row, columns, carry seed) are those of dust_scatter_rows, so the pyramid is the same word for
// word (test gpu_dust_queue_draws_the_same_pyramid); what changes is that a lane no longer walks
// every row of the record (profile 2026-10-15: that loop was 93 % of the queue draw's lane-cycles,
// O(rows) per lane for a sixty-fourth of the segments). y0, y1 and the chunks are uniform over the
// workgroup, so the barriers are reached by every lane.
#ifdef DUST_QUEUE_WORKGROUP
const uint DUST_QUEUE_ROW_CHUNK = 256u;
shared int  dust_q_x0[DUST_QUEUE_ROW_CHUNK];
shared int  dust_q_x1[DUST_QUEUE_ROW_CHUNK];
shared uint dust_q_seg[DUST_QUEUE_ROW_CHUNK + 1u];
bool dust_scatter_rows_wg(DustUintBuffer pyr, float invUnit, DustScreenPacket P, vec3 color, float s,
                          uint off, uint w, uint h, uint seed, int rowA, int rowB, uint lanes, uint lane) {
    DUST_PROF_BEGIN(qrows)
    DustBand B = dust_band_new(P);
    int y0, y1;
    if (!dust_band_rows(B, s, h, y0, y1)) { DUST_PROF_END(qrows) return false; }
    y0 = max(y0, rowA);
    y1 = min(y1, rowB);
    bool any_ = false;
    for (int yc0 = y0; yc0 <= y1; yc0 += int(DUST_QUEUE_ROW_CHUNK)) {
        int n = min(int(DUST_QUEUE_ROW_CHUNK), y1 - yc0 + 1);
        for (int r = int(lane); r < n; r += int(lanes)) {
            int x0, x1;
            bool ok = dust_band_columns(B, (float(yc0 + r) + 0.5) * s, s, w, x0, x1);
            dust_q_x0[r] = x0;
            dust_q_x1[r] = x1;
            dust_q_seg[r] = ok ? uint(x1 - x0) / DUST_SEGMENT_TEXELS + 1u : 0u;
        }
        barrier();
        if (lane == 0u) {
            uint acc = 0u;
            for (int r = 0; r < n; ++r) {
                uint c = dust_q_seg[r];
                dust_q_seg[r] = acc;
                acc += c;
            }
            dust_q_seg[n] = acc;
        }
        barrier();
        uint total = dust_q_seg[n];
        int r = 0;
        for (uint k = lane; k < total; k += lanes) {
            while (dust_q_seg[r + 1] <= k) r++;
            uint j = k - dust_q_seg[r];
            int xa = dust_q_x0[r] + int(j * DUST_SEGMENT_TEXELS);
            int xb = min(xa + int(DUST_SEGMENT_TEXELS) - 1, dust_q_x1[r]);
            DUST_PROF_BEGIN_AS(seg)
            any_ = dust_scatter_segment(pyr, invUnit, P, color, s, off, w, seed, yc0 + r, xa, xb, true) || any_;
            DUST_PROF_END_AS(qseg, seg)
            DUST_PROF_COUNT(qtexels, xb - xa + 1)
        }
        barrier();
    }
    DUST_PROF_END(qrows)
    return any_;
}
#endif

// One Gaussian piece of a volumetric packet (dust::VolumePiece)
struct DustVolumePiece {
    float l[6];   // Cholesky factor of its covariance
    vec3  qw;     // L^-1 (eye - mean)
    float k;      // exposure * flux / (2 pi sqrt(det S))
    vec4  bbox;   // projected 3-sigma box (px): x0 x1 y0 y1
    bool  full;   // the whole frame (the eye near it, or a box corner not in front of the eye)
    bool  on;     // on screen
    uint  lp;     // its own level (dust::VolumePiece::lp): its screen sigma, or for a full piece the smaller of that and its angular sigma
    uint  line;   // the screen axis its band is thinner along (1 x, 2 y; 0 none: dust::DUST_VOLUME_CLASS_LINE)
    uint  ls;     // the level of its screen sigma alone (0xFFFFFFFF: none)
    bool  cap;    // its ray integral is clipped by the packet's capsule too (the two pieces at each end of an axis)
};

// L x and L^T x for a lower Cholesky factor (dust::l_mul / lt_mul)
vec3 dust_l_mul(float L[6], vec3 v) {
    return vec3(L[0] * v.x, L[1] * v.x + L[3] * v.y, L[2] * v.x + L[4] * v.y + L[5] * v.z);
}
vec3 dust_lt_mul(float L[6], vec3 v) {
    return vec3(L[0] * v.x + L[1] * v.y + L[2] * v.z, L[3] * v.y + L[4] * v.z, L[5] * v.z);
}
// the smaller sigma (px) of a 2 x 2 screen covariance xx xy yy (dust::volume_screen_sigma_min)
float dust_volume_screen_sigma_min(float xx, float xy, float yy) {
    float half_ = 0.5 * (xx + yy);
    float dd = sqrt((0.5 * (xx - yy)) * (0.5 * (xx - yy)) + xy * xy);
    float lamMax = half_ + dd;
    if (!(lamMax > 0.0)) return 0.0;
    return sqrt(max((xx * yy - xy * xy) / lamMax, 0.0));
}
// the screen sigma and band axis of a piece at x (clip w; the lateral covariance less g g^T gg)
// (the screen_at closure of dust::volume_pieces)
vec2 dust_volume_screen_at(float Lk[6], vec3 x, float w, vec3 g, float gg, vec3 r0, vec3 r1, vec3 r3, vec3 t,
                           float W, float H) {
    float ndx = (dot(r0, x) + t.x) / w;
    float ndy = (dot(r1, x) + t.y) / w;
    vec3 px = (r0 - r3 * ndx) * (0.5 * W / w);
    vec3 py = (r1 - r3 * ndy) * (0.5 * H / w);
    vec3 lpx = dust_lt_mul(Lk, px), lpy = dust_lt_mul(Lk, py);
    float pgx = dot(px, g), pgy = dot(py, g);
    float cxx = dot(lpx, lpx) - gg * pgx * pgx;
    float cyy = dot(lpy, lpy) - gg * pgy * pgy;
    float cxy = dot(lpx, lpy) - gg * pgx * pgy;
    return vec2(dust_volume_screen_sigma_min(cxx, cxy, cyy), cxx <= cyy ? 1.0 : 2.0);
}

// The n pieces of a volumetric packet (dust::volume_pieces): covariance S (pixel filter
// included), piece variance cvar of the spacing^2
// (the axes: base point axMean and covariance S, axis a = segA (the chord, rho ramp, from the base
// point) with nA pieces of variance cvarA spacing^2, axis b = segB (dust::Packet::seg_t, uniform,
// centred) with nB pieces when hasB; piece i * nB + j; without axis b the chord-only arithmetic)
// (the view: the mvp's rows r0, r1, r3 and translations t = (t0, t1, t3), the eye; the packet's
// flux and density ramp rho)
DustVolumePiece dust_volume_piece(uint k, float flux, vec2 rho, float exposure, vec3 axMean, float S[6], vec3 segA,
                                  uint nA, float cvarA, vec3 segB, uint nB, float cvarB, bool hasB, float W, float H,
                                  vec3 r0, vec3 r1, vec3 r3, vec3 t, vec3 eye, float fPx, uint levels,
                                  bool withLevel) {
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
    // a piece whose 3-sigma ellipsoid lies wholly behind the eye plane draws nothing
    // (dust::volume_pieces, step F5)
    {
        vec3 lt30 = dust_lt_mul(Lk, r3);
        if (dot(r3, meanK) + t3 + DUST_VOLUME_SIGMAS * sqrt(dot(lt30, lt30)) < 0.0) { q.on = false; full = false; q.full = false; }
    }
    q.cap = nA <= 4u || i < 2u || i + 2u >= nA || (hasB && (nB <= 4u || j < 2u || j + 2u >= nB));
    // (withLevel false: pass B rebuilding a piece whose class pass A packed; K1c off: none needed)
#ifndef DUST_K1C_OFF
    if (!withLevel)
#endif
    {
        q.lp = 0u;
        q.line = 0u;
        q.ls = 0xFFFFFFFFu;
        return q;
    }
    // its own level (dust::volume_pieces, K1c): the screen sigma at its mean (in front), or at
    // the conditional mean of its front part for a full piece whose mean is behind the eye; a
    // full piece also by its angular sigma
    float wM = dot(r3, meanK) + t3;
    float sigScr = 3.4e38;
    uint line = 0u;
    if (wM > 0.0) {
        vec2 sa = dust_volume_screen_at(Lk, meanK, wM, vec3(0.0), 0.0, r0, r1, r3, t, W, H);
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
                vec2 sa = dust_volume_screen_at(Lk, xF, wF, g, 1.0 / sw2, r0, r1, r3, t, W, H);
                sigScr = sa.x;
                line = uint(sa.y);
            }
        }
    }
    float sigPx = sigScr;
    if (full) {
        float sigK = 2.0 * (Lk[0] * Lk[3] * Lk[5]) / (Sk[0] + Sk[3] + Sk[5]);
        vec3 dq = eye - meanK;
        float denK = max(sqrt(dot(dq, dq)), sigK);
        sigPx = min((denK > 0.0 ? sigK / denK : 1.0) * fPx, sigScr);
    }
    q.lp = dust_volume_level(min(sigPx, 1e30), levels);
    q.line = line;
    q.ls = line != 0u ? dust_volume_level(min(sigScr, 1e30), levels) : 0xFFFFFFFFu;
    return q;
}

void dust_volume_pieces(float flux, vec2 rho, float exposure, vec3 axMean, float S[6], vec3 segA, uint nA, float cvarA,
                        vec3 segB, uint nB, float cvarB, bool hasB, float W, float H,
                        vec3 r0, vec3 r1, vec3 r3, vec3 t, vec3 eye, float fPx, uint levels,
                        inout DustVolumePiece pieces[DUST_VOLUME_PIECES_MAX]) {
    for (uint k = 0u; k < nA * nB; ++k) {
        pieces[k] = dust_volume_piece(k, flux, rho, exposure, axMean, S, segA, nA, cvarA, segB, nB, cvarB, hasB, W, H,
                                      r0, r1, r3, t, eye, fPx, levels, true);
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

// How a piece's texels at `level` are drawn and the sub-ray order it asks for (dust::volume_class,
// K1c, the texel's integral): (class, k)
uvec2 dust_volume_class(DustVolumePiece q, uint level, uint levels) {
#ifdef DUST_K1C_OFF
    // K1c off (a diagnostics build, dust::k1c_off): every piece by its texel-centre rays
    return uvec2(DUST_VOLUME_CLASS_CENTRE, 1u);
#endif
    if (level < q.lp + DUST_VOLUME_COARSE_LEVELS) return uvec2(DUST_VOLUME_CLASS_CENTRE, 1u);
    uint c = level - q.lp;
    float lim = DUST_VOLUME_SMALL_TEXELS * float(1u << level);
    if (!q.full && ((q.bbox.y - q.bbox.x <= lim && q.bbox.w - q.bbox.z <= lim) || c >= DUST_VOLUME_THIN_LEVELS))
        return uvec2(DUST_VOLUME_CLASS_SMALL, 1u);
    bool lsOk = q.ls <= 0xFFFFFFFFu - DUST_VOLUME_LINE_LEVELS;
    if (q.full && q.line != 0u && lsOk && level >= q.ls + DUST_VOLUME_LINE_LEVELS
        && dot(q.qw, q.qw) >= DUST_VOLUME_SIGMAS * DUST_VOLUME_SIGMAS)
        return uvec2(DUST_VOLUME_CLASS_LINE, DUST_VOLUME_SUBRAY_ORDER_MAX);
    // the order min(c, 4) (dust::volume_class: the step-D caps 3 / 2 cost 14 % of the 1-10 d
    // volume class's light and are off in production, dust::DUST_K1C_D_PRODUCTION)
    return uvec2(DUST_VOLUME_CLASS_SUBRAY, min(c, DUST_VOLUME_SUBRAY_ORDER_MAX));
}

// The evaluations a piece of `t` range texels makes at a sub-ray limit K (dust::volume_subray_limit:
// centre and small 1, lines min(4, K), a full sub-ray piece min(k, K)^2, a non-full one k^2 when
// k <= K else 1: drawn small)
uint dust_volume_piece_cost(DustVolumePiece q, uvec2 ck, uint K) {
    if (ck.x == DUST_VOLUME_CLASS_LINE) return min(DUST_VOLUME_SUBRAY_ORDER_MAX, K);
    if (ck.x != DUST_VOLUME_CLASS_SUBRAY) return 1u;
    if (q.full) { uint m = min(ck.y, K); return m * m; }
    return ck.y <= K ? ck.y * ck.y : 1u;
}

// The i-th node (half-interval units) and weight (sum 1) of the k-point Gauss-Legendre rule, k in
// 1..4 (dust::gl_node; branches, no table)
vec2 dust_gl_node(uint k, uint i) {
    if (k == 2u) return i == 0u ? vec2(-0.57735026, 0.5) : vec2(0.57735026, 0.5);
    if (k == 3u) {
        if (i == 0u) return vec2(-0.7745967, 5.0 / 18.0);
        if (i == 1u) return vec2(0.0, 8.0 / 18.0);
        return vec2(0.7745967, 5.0 / 18.0);
    }
    if (k == 4u) {
        if (i == 0u) return vec2(-0.8611363, 0.17392743);
        if (i == 1u) return vec2(-0.33998104, 0.32607257);
        if (i == 2u) return vec2(0.33998104, 0.32607257);
        return vec2(0.8611363, 0.17392743);
    }
    return vec2(0.0, 1.0);
}

// The 3-sigma bound of a volumetric packet (dust::VolumeBound, step F2): its capsule (the segment
// start + u seg, u in [0, 1], plus the 3-sigma ellipsoid of its base covariance) whitened by that
// covariance's Cholesky factor l (qe = L^-1 (eye - start), g = L^-1 seg): the support of the
// bounded Monte-Carlo's grains
struct DustVolumeBound {
    float l[6];
    vec3  qe;
    vec3  g;
};
DustVolumeBound dust_volume_bound(float base[6], vec3 start, vec3 seg, vec3 eye) {
    DustVolumeBound b;
    float L[6];
    dust_cholesky3(base, L);
    b.l = L;
    b.g = dust_cholesky_solve3(L, seg);
    b.qe = dust_cholesky_solve3(L, eye - start);
    return b;
}
// the interval of t where the ray eye + t u is inside the capsule (empty: x > y): its end
// spheres' and its cylinder's (within the axial slab) intervals joined (dust::capsule_interval)
vec2 dust_capsule_interval(DustVolumeBound bd, vec3 u) {
    float r2 = DUST_BOUND_R2;
    vec3 w = dust_cholesky_solve3(bd.l, u);
    vec3 q = bd.qe;
    float aa = dot(w, w);
    float lo = 3.4e38, hi = -3.4e38;
    if (!(aa > 0.0)) return vec2(lo, hi);
    for (uint e = 0u; e < 2u; ++e) {
        vec3 qc = e == 0u ? q : q - bd.g;
        float bb = dot(qc, w);
        float cc = dot(qc, qc) - r2;
        float disc = bb * bb - aa * cc;
        if (disc >= 0.0) {
            float sd = sqrt(disc);
            lo = min(lo, (-bb - sd) / aa);
            hi = max(hi, (-bb + sd) / aa);
        }
    }
    float gg = dot(bd.g, bd.g);
    if (gg > 0.0) {
        float gl = sqrt(gg);
        vec3 gh = bd.g * (1.0 / gl);
        float wa = dot(w, gh), qa = dot(q, gh);
        vec3 wp = w - gh * wa;
        vec3 qp = q - gh * qa;
        float a2 = dot(wp, wp), b2 = dot(qp, wp), c2 = dot(qp, qp) - r2;
        float t1 = 3.4e38, t2 = -3.4e38;
        if (a2 > 1e-12 * aa) {
            float disc = b2 * b2 - a2 * c2;
            if (disc >= 0.0) {
                float sd = sqrt(disc);
                t1 = (-b2 - sd) / a2;
                t2 = (-b2 + sd) / a2;
            }
        } else if (c2 <= 0.0) {
            t1 = -3.4e38;
            t2 = 3.4e38;
        }
        if (abs(wa) > 1e-6 * sqrt(aa)) {
            float ta = (0.0 - qa) / wa, tb = (gl - qa) / wa;
            t1 = max(t1, min(ta, tb));
            t2 = min(t2, max(ta, tb));
        } else if (!(qa >= 0.0 && qa <= gl)) {
            t1 = 3.4e38;
            t2 = -3.4e38;
        }
        if (t2 > t1) {
            lo = min(lo, t1);
            hi = max(hi, t2);
        }
    }
    return vec2(lo, hi);
}

// The optical depth per px^2 of a piece along the ray on the planes aRow, bRow (dust::volume_ray_tau);
// -1 when they give no ray
float dust_volume_ray_tau(DustVolumePiece pc_, DustVolumeBound bd_, float renorm, vec3 eye, vec3 aRow, vec3 bRow, vec3 r3, vec3 meanRef) {
    precise vec3 v = cross(aRow, bRow);
    if (dot(v, r3) < 0.0) v = -v;
    float vn = sqrt(dot(v, v));
    if (!(vn > 0.0)) return -1.0;
    vec3 u = v * (1.0 / vn);
    vec3 uw = dust_cholesky_solve3(pc_.l, u);
    float a = dot(uw, uw);
    float b = dot(uw, pc_.qw);
    precise vec3 rr = pc_.qw - uw * (b / a);
    // the capsule only for the pieces at the chord's ends (dust::volume_ray_tau, step F5)
    vec2 cap = pc_.cap ? dust_capsule_interval(bd_, u) : vec2(-3.4e38, 3.4e38);
    if (!(cap.y > max(cap.x, 0.0))) return 0.0;
    float rr2 = dot(rr, rr);
    float sa = sqrt(a);
    float tauPx = pc_.k * renorm / sa * exp(-0.5 * rr2)
                * dust_ray_kept(a, b, rr2, cap.x, cap.y, eye, u, dust_phot_support);
    if (dust_phot_albedo > 0.0) {
        vec3 x = eye + u * max(-b / a, 0.0);
        tauPx *= dust_volume_weight(meanRef, x, u);
    }
    return tauPx;
}

// the ray integral at the line coordinate t, 0 when invalid (the tau_at closure of
// dust::volume_line_integral)
float dust_volume_line_tau(DustVolumePiece pc_, DustVolumeBound bd_, float renorm, vec3 eye, vec3 r0, vec3 r1, vec3 r3, float W, float H,
                           vec3 meanRef, bool alongX, float fixed_, float t) {
    vec3 aR = alongX ? r0 - r3 * (2.0 * t / W - 1.0) : r0 - r3 * (2.0 * fixed_ / W - 1.0);
    vec3 bR = alongX ? r1 - r3 * (2.0 * fixed_ / H - 1.0) : r1 - r3 * (2.0 * t / H - 1.0);
    float v = dust_volume_ray_tau(pc_, bd_, renorm, eye, aR, bR, r3, meanRef);
    return (v >= 0.0 && !isinf(v)) ? v : 0.0;
}

// int tau dt along one screen line of a texel: the row y = fixed_ over x in [lo, hi] (alongX) or
// the column x = fixed_ over y; Laplace about the line's closed-form minimum of the ray's
// Mahalanobis distance, 4-point Gauss-Legendre when it has none (dust::volume_line_integral)
float dust_volume_line_integral(DustVolumePiece pc_, DustVolumeBound bd_, float renorm, vec3 eye, vec3 r0, vec3 r1, vec3 r3, float W,
                                float H, vec3 meanRef, bool alongX, float fixed_, float lo, float hi) {
    float mid = 0.5 * (lo + hi);
    vec3 aM = alongX ? r0 - r3 * (2.0 * mid / W - 1.0) : r0 - r3 * (2.0 * fixed_ / W - 1.0);
    vec3 bM = alongX ? r1 - r3 * (2.0 * fixed_ / H - 1.0) : r1 - r3 * (2.0 * mid / H - 1.0);
    vec3 vm = cross(aM, bM);
    vec3 v1 = alongX ? cross(r3, bM) * (-2.0 / W) : cross(aM, r3) * (-2.0 / H);
    if (dot(vm, r3) < 0.0) { vm = -vm; v1 = -v1; }
    vec3 wm = dust_cholesky_solve3(pc_.l, vm);
    vec3 w1 = dust_cholesky_solve3(pc_.l, v1);
    float a0 = dot(wm, wm), a1 = dot(wm, w1), a2 = dot(w1, w1);
    float b0 = dot(wm, pc_.qw), b1 = dot(w1, pc_.qw);
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
        float h = 0.5 * (hi - lo), acc = 0.0;
        for (uint i = 0u; i < DUST_VOLUME_SUBRAY_ORDER_MAX; ++i) {
            vec2 nw = dust_gl_node(DUST_VOLUME_SUBRAY_ORDER_MAX, i);
            acc += nw.y * dust_volume_line_tau(pc_, bd_, renorm, eye, r0, r1, r3, W, H, meanRef, alongX, fixed_, mid + h * nw.x);
        }
        return acc * (hi - lo);
    }
    float sg = sqrt(2.0 / d2);
    float t = mid + ts;
    // the band's rays reach the piece's 3-sigma ellipsoid where their distance^2 <= 9 (step F)
    float atS = a0 + 2.0 * a1 * ts + a2 * ts * ts;
    float btS = b0 + b1 * ts;
    float r2 = dot(pc_.qw, pc_.qw) - btS * btS / atS;
    if (!(r2 < DUST_BOUND_R2)) return 0.0;
    float half_ = sg * sqrt(DUST_BOUND_R2 - r2);
    lo = max(lo, t - half_);
    hi = min(hi, t + half_);
    if (!(hi > lo)) return 0.0;
    float k = 1.0 / (sg * DUST_SQRT_2);
    return dust_volume_line_tau(pc_, bd_, renorm, eye, r0, r1, r3, W, H, meanRef, alongX, fixed_, t) * sg * DUST_SQRT_2PI
         * 0.5 * (dust_erf((hi - t) * k) - dust_erf((lo - t) * k));
}

// One point of a small piece's depth quadrature (dust::VolumeSmallPoint): its screen position,
// 1 / (sigma sqrt 2) along x and y, the correlation, its share of the piece's screen integral
struct DustSmallPoint {
    float mx;
    float my;
    float ix;
    float iy;
    float rho;
    float m;
};

// The depth node (in sigma) and Gauss-Hermite weight of point q of a small piece (0: the centre;
// 3 points 1: -sqrt3, 2: +sqrt3; 5 points 1, 2: -+n1, 3, 4: -+n2; dust::volume_small_points)
vec2 dust_small_node(uint q, bool five) {
    float off, wq;
    if (five) {
        off = q == 0u ? 0.0 : (q == 1u ? -DUST_GH5_N1 : (q == 2u ? DUST_GH5_N1 : (q == 3u ? -DUST_GH5_N2 : DUST_GH5_N2)));
        wq = q == 0u ? DUST_GH5_W0 : (q <= 2u ? DUST_GH5_W1 : DUST_GH5_W2);
    } else {
        off = q == 0u ? 0.0 : (q == 1u ? -DUST_GH3_NODE : DUST_GH3_NODE);
        wq = q == 0u ? 2.0 / 3.0 : 1.0 / 6.0;
    }
    return vec2(off, wq);
}

// The radiance weights of a small piece's depth slices (dust::volume_small_points' weight_at;
// 1 without albedo), computed before the slices' projections. Final round A2: the Sun's direction
// and r_h of the centre slice x0 serve every slice (dust_volume_weight's arithmetic at x0; the
// slices lie <= 2.9 sigma_depth from it against r_h AU); each slice has its own phase angle
// (dust_radiance_weight's arithmetic) and its own interpolation of the phase table, whose two
// entries are read once when the slices share the degree index (a slice in another degree reads
// its own). The centre's weight is bit-identical to dust_volume_weight's
// the Sun's direction (unit, from the dust to the Sun) and r_h (AU) at x
void dust_small_sun(vec3 x, vec3 meanRef, out vec3 sunDir, out float rAu) {
    vec3 rVec = dust_phot_sun * (-dust_phot_rh * DUST_AU_M) + (x - meanRef);
    float rn = max(sqrt(dot(rVec, rVec)), 1.0);
    sunDir = rVec * (-1.0 / rn);
    rAu = rn / DUST_AU_M;
}
// the phase angle (degrees, clamped to 0..180) of the dust at xq seen from the eye
float dust_small_phase_deg(vec3 xq, vec3 eye, vec3 sunDir) {
    vec3 dd = xq - eye;
    vec3 toEye = -(dd * (1.0 / sqrt(dot(dd, dd))));
    float n = dot(toEye, toEye);
    float cosA = n > 0.0 ? dot(sunDir, toEye) / sqrt(n) : 1.0;
    float alpha = acos(clamp(cosA, -1.0, 1.0));
    return clamp(alpha * (180.0 / DUST_PI_F32), 0.0, 180.0);
}
float dust_small_lut(uint i) { return uintBitsToFloat(dust_phot_lut.v[DUST_PYR_PHASE_LUT + i]); }
// one slice's weight from its phase angle (its own table entries)
float dust_small_weight_deg(float deg, float r2) {
    uint i = min(uint(deg), 179u);
    float a = dust_small_lut(i), b = dust_small_lut(i + 1u);
    return dust_phot_albedo * (a + (b - a) * (deg - float(i))) / r2;
}
// the three slices' weights (x0 the centre); the table entries shared when the degree matches
void dust_small_weights(vec3 x0, vec3 x1, vec3 x2, vec3 eye, vec3 meanRef, out float wr0, out float wr1,
                        out float wr2, out vec3 sun0, out float r20) {
    wr0 = 1.0; wr1 = 1.0; wr2 = 1.0; sun0 = vec3(0.0); r20 = 1.0;
    if (!(dust_phot_albedo > 0.0)) return;
    vec3 sun1, sun2;
    float rAu0, rAu1, rAu2;
    dust_small_sun(x0, meanRef, sun0, rAu0);
    // (A2b: the centre's Sun geometry for every slice)
    sun1 = sun0; sun2 = sun0; rAu1 = rAu0; rAu2 = rAu0;
    float r0_ = max(rAu0, 1e-6), r1_ = max(rAu1, 1e-6), r2_ = max(rAu2, 1e-6);
    r20 = r0_ * r0_;
    float g0 = dust_small_phase_deg(x0, eye, sun0);
    float g1 = dust_small_phase_deg(x1, eye, sun1);
    float g2 = dust_small_phase_deg(x2, eye, sun2);
    uint i0 = min(uint(g0), 179u), i1 = min(uint(g1), 179u), i2 = min(uint(g2), 179u);
    float a0 = dust_small_lut(i0), b0 = dust_small_lut(i0 + 1u);
    float a1 = a0, b1 = b0, a2 = a0, b2 = b0;
    if (i1 != i0) { a1 = dust_small_lut(i1); b1 = dust_small_lut(i1 + 1u); }
    if (i2 != i0) { a2 = dust_small_lut(i2); b2 = dust_small_lut(i2 + 1u); }
    wr0 = dust_phot_albedo * (a0 + (b0 - a0) * (g0 - float(i0))) / r20;
    wr1 = dust_phot_albedo * (a1 + (b1 - a1) * (g1 - float(i1))) / (r1_ * r1_);
    wr2 = dust_phot_albedo * (a2 + (b2 - a2) * (g2 - float(i2))) / (r2_ * r2_);
}
// a further slice's weight (the 5-point lab rule's nodes 3, 4) with the centre's Sun geometry
float dust_small_weight_more(vec3 xq, vec3 eye, vec3 meanRef, vec3 sun0, float r20) {
    if (!(dust_phot_albedo > 0.0)) return 1.0;
    return dust_small_weight_deg(dust_small_phase_deg(xq, eye, sun0), r20);
}

// 1 / sqrt2 (dust::volume_small_points' finish: ix = 1 / (sigma sqrt2) by inversesqrt)
const float DUST_SMALL_INV_SQRT2 = 0.70710678118654752;

// A small point from its slice's moments (dust::volume_small_points' finish): the correlation
// clamped, the mass wq massW sj wr; none (m = 0) unless ok and the mass is positive and finite.
// Branch-free (selects), so the three fast slices stay one straight-line block
DustSmallPoint dust_small_finish(bool ok, float wq, float massW, float wr, float sj, float mx, float my,
                                 float cxx, float cyy, float rhoRaw) {
    float rho = (isnan(rhoRaw) || isinf(rhoRaw)) ? 0.0 : clamp(rhoRaw, -DUST_SMALL_RHO_MAX, DUST_SMALL_RHO_MAX);
    float wgt = wq * massW * sj;
    // the radiance weight per depth point (dust::volume_small_points: one a piece cost 2.5 % of
    // the 1-10 d split class and is off in production)
    wgt *= wr;
    bool keep = ok && wgt > 0.0 && !isinf(wgt);
    DustSmallPoint o;
    o.mx = keep ? mx : 0.0;
    o.my = keep ? my : 0.0;
    o.ix = keep ? inversesqrt(cxx) * DUST_SMALL_INV_SQRT2 : 0.0;
    o.iy = keep ? inversesqrt(cyy) * DUST_SMALL_INV_SQRT2 : 0.0;
    o.rho = keep ? rho : 0.0;
    o.m = keep ? wgt : 0.0;
    return o;
}

// One depth point q of a small piece by one projection with the screen Jacobian's lateral
// covariance (dust::volume_small_points' fast_point; mean, g, the lateral axes' dots x1 = r0.lat1,
// x2 = r0.lat2, y1 = r1.lat1, y2 = r1.lat2, w1 = r3.lat1, w2 = r3.lat2, the mass and the slice's
// radiance weight wr precomputed by the caller): the columns (W / 2w) (r0.lat_i - ndx r3.lat_i),
// (H / 2w) (r1.lat_i - ndy r3.lat_i); one reciprocal of w a slice (round 3); branch-free
DustSmallPoint dust_volume_small_fast(uint q, bool five, vec3 mean, vec3 g, float x1, float x2, float y1, float y2,
                                      float w1, float w2, float massW, float wr, vec3 r0, vec3 r1, vec3 r3, vec3 t,
                                      float W, float H) {
    vec2 nw = dust_small_node(q, five);
    vec3 xq = mean + g * nw.x;
    float w = dot(r3, xq) + t.z;
    bool ok = w > 0.0;
    // (behind the eye: a finite stand-in, the point dropped by ok)
    float iw = 1.0 / (ok ? w : 1.0);
    float ndx = (dot(r0, xq) + t.x) * iw;
    float ndy = (dot(r1, xq) + t.y) * iw;
    vec3 c = cross(r0 - r3 * ndx, r1 - r3 * ndy);
    float sj = 0.25 * W * H * sqrt(dot(c, c)) * (iw * iw);
    float kx = 0.5 * W * iw, ky = 0.5 * H * iw;
    float a1 = kx * (x1 - ndx * w1), a2 = kx * (x2 - ndx * w2);
    float b1 = ky * (y1 - ndy * w1), b2 = ky * (y2 - ndy * w2);
    float mx = (ndx * 0.5 + 0.5) * W;
    float my = (ndy * 0.5 + 0.5) * H;
    float cxx = max(a1 * a1 + a2 * a2, 1e-6);
    float cyy = max(b1 * b1 + b2 * b2, 1e-6);
    float rhoRaw = (a1 * b1 + a2 * b2) * inversesqrt(cxx * cyy);
    return dust_small_finish(ok, nw.y, massW, wr, sj, mx, my, cxx, cyy, rhoRaw);
}

// One depth point q of a small piece through the 4 lateral sigma points of its slice
// (dust::volume_small_points' edge_point: x_q +- sqrt2 lat_i projects through r.x_q +- sqrt2
// r.lat_i; J3-weighted moments about the first point's position; one reciprocal a sigma point)
DustSmallPoint dust_volume_small_edge(uint q, bool five, vec3 mean, vec3 g, float x1, float x2, float y1, float y2,
                                      float w1, float w2, float massW, float wr, vec3 r0, vec3 r1, vec3 r3, vec3 t,
                                      float W, float H) {
    vec2 nw = dust_small_node(q, five);
    vec3 xq = mean + g * nw.x;
    float bx = dot(r0, xq) + t.x, by = dot(r1, xq) + t.y, bw = dot(r3, xq) + t.z;
    float tj = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    float ox = 0.0, oy = 0.0;
    bool haveO = false;
    for (uint j = 0u; j < 4u; ++j) {
        float s = (j & 1u) == 0u ? -DUST_UT2_NODE : DUST_UT2_NODE;
        bool ax1 = j < 2u;
        float w = bw + s * (ax1 ? w1 : w2);
        if (!(w > 0.0)) continue;
        float iw = 1.0 / w;
        float ndx = (bx + s * (ax1 ? x1 : x2)) * iw;
        float ndy = (by + s * (ax1 ? y1 : y2)) * iw;
        vec3 c = cross(r0 - r3 * ndx, r1 - r3 * ndy);
        float j3 = 0.25 * (0.25 * W * H * sqrt(dot(c, c)) * (iw * iw));
        float sxp = (ndx * 0.5 + 0.5) * W, syp = (ndy * 0.5 + 0.5) * H;
        if (!haveO) { ox = sxp; oy = syp; haveO = true; }
        float px = sxp - ox, py = syp - oy;
        tj += j3;
        sx += j3 * px;
        sy += j3 * py;
        sxx += j3 * px * px;
        syy += j3 * py * py;
        sxy += j3 * px * py;
    }
    bool ok = tj > 0.0;
    float itj = 1.0 / (ok ? tj : 1.0);
    float dx = sx * itj, dy = sy * itj;
    float cxx = max(sxx * itj - dx * dx, 1e-6);
    float cyy = max(syy * itj - dy * dy, 1e-6);
    float rhoRaw = (sxy * itj - dx * dy) * inversesqrt(cxx * cyy);
    return dust_small_finish(ok, nw.y, massW, wr, tj, ox + dx, oy + dy, cxx, cyy, rhoRaw);
}

// a small piece's empty point (m = 0: none)
DustSmallPoint dust_small_none() {
    DustSmallPoint o;
    o.mx = 0.0; o.my = 0.0; o.ix = 0.0; o.iy = 0.0; o.rho = 0.0; o.m = 0.0;
    return o;
}

// The three depth points of a small volumetric piece (dust::volume_small_points), as three
// values (no array). Live across the slices: mean, g, the lateral axes' six dots, the mass and the
// slices' weights. Round 3: the weights and the fast projections of every slice as one
// straight-line block (independent chains the scheduler overlaps), then the edge decision on s0
// as before; an edge piece redraws every slice by sigma points (its fast s1, s2 discarded).
// Profile parts (DUST_PROF_K1C0): vsmprep the prelude, vsmfast the weights, the fast slices and
// the decision (every piece), vsmedge an edge piece's slices; the pieces by path counted
void dust_volume_small_points(DustVolumePiece pc_, vec3 eye, vec3 r0, vec3 r1, vec3 r3, vec3 t, float W, float H,
                              vec3 meanRef, float renorm, out DustSmallPoint s0, out DustSmallPoint s1,
                              out DustSmallPoint s2, out DustSmallPoint s3, out DustSmallPoint s4) {
    s0 = dust_small_none(); s1 = dust_small_none(); s2 = dust_small_none(); s3 = dust_small_none(); s4 = dust_small_none();
    DUST_PROF_BEGIN(vsmprep)
    vec3 lq = dust_l_mul(pc_.l, pc_.qw);
    vec3 mean = eye - lq;
    float dn = sqrt(dot(lq, lq));
    if (!(dn > 0.0)) { DUST_PROF_END(vsmprep) return; }
    vec3 e = lq * (-1.0 / dn);
    vec3 lte = dust_lt_mul(pc_.l, e);
    float se = sqrt(dot(lte, lte));
    if (!(se > 0.0)) { DUST_PROF_END(vsmprep) return; }
    vec3 z = lte * (1.0 / se);
    vec3 g = dust_l_mul(pc_.l, z);
    float sgn = z.z >= 0.0 ? 1.0 : -1.0;
    float ia = -1.0 / (sgn + z.z);
    float ib = z.x * z.y * ia;
    // bounded at 3 sigma (dust::volume_small_points, step F2): the spread narrowed to the 3-sigma
    // ball's second moment
    float sb = sqrt(DUST_BOUND_VAR);
    g = g * sb;
    vec3 lat1 = dust_l_mul(pc_.l, vec3(1.0 + sgn * z.x * z.x * ia, sgn * ib, -sgn * z.x)) * sb;
    vec3 lat2 = dust_l_mul(pc_.l, vec3(ib, sgn + z.y * z.y * ia, -z.y)) * sb;
    // the lateral axes against the projection's rows (lat1, lat2 dead after these)
    float x1 = dot(r0, lat1), x2 = dot(r0, lat2);
    float y1 = dot(r1, lat1), y2 = dot(r1, lat2);
    float w1 = dot(r3, lat1), w2 = dot(r3, lat2);
    vec4 sp = dust_phot_support;
    float kept = 1.0;
    if (dust_has_support(sp)) {
        vec3 ltn = dust_lt_mul(pc_.l, sp.xyz);
        float sn = sqrt(dot(ltn, ltn));
        float mn = dot(sp.xyz, mean) - sp.w;
        kept = sn > 0.0 ? 0.5 * (1.0 + dust_erf(mn / (sn * DUST_SQRT_2))) : (mn >= 0.0 ? 1.0 : 0.0);
    }
    float mass = 2.0 * DUST_PI_F32 * pc_.k * (pc_.l[0] * pc_.l[3] * pc_.l[5]) * renorm * kept;
    float massW = mass;
    // (the 5-point depth rule of step F5 is off in production: dust::DUST_K1C_D_GH5)
    bool five = false;
    DUST_PROF_END(vsmprep)
    // every slice's weight and one projection, then the decision: when the centre slice's 3 sigma
    // crosses the frame's edge (or it has none), the 4 lateral sigma points a slice
    // (dust::volume_small_points, step D3); the weights serve both paths
    DUST_PROF_BEGIN(vsmfast)
    float wr0, wr1, wr2;
    vec3 sun0;
    float r20;
    dust_small_weights(mean + g * dust_small_node(0u, five).x, mean + g * dust_small_node(1u, five).x,
                       mean + g * dust_small_node(2u, five).x, eye, meanRef, wr0, wr1, wr2, sun0, r20);
#ifdef DUST_PROF_SMALL_NO_WEIGHT
    // profile experiment P1b only (a hand-built .prof blob): the other slices reuse the centre's
    // weight, the deposits' magnitude kept
    wr1 = wr0; wr2 = wr0;
#endif
    float wr3 = 1.0, wr4 = 1.0;
    if (five) {
        wr3 = dust_small_weight_more(mean + g * dust_small_node(3u, five).x, eye, meanRef, sun0, r20);
        wr4 = dust_small_weight_more(mean + g * dust_small_node(4u, five).x, eye, meanRef, sun0, r20);
    }
    s0 = dust_volume_small_fast(0u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr0, r0, r1, r3, t, W, H);
    s1 = dust_volume_small_fast(1u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr1, r0, r1, r3, t, W, H);
    s2 = dust_volume_small_fast(2u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr2, r0, r1, r3, t, W, H);
    if (five) {
        s3 = dust_volume_small_fast(3u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr3, r0, r1, r3, t, W, H);
        s4 = dust_volume_small_fast(4u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr4, r0, r1, r3, t, W, H);
    }
    bool edge = true;
    if (s0.m > 0.0) {
        float ex = 3.0 / (s0.ix * DUST_SQRT_2), ey = 3.0 / (s0.iy * DUST_SQRT_2);
        edge = s0.mx - ex < 0.0 || s0.mx + ex > W || s0.my - ey < 0.0 || s0.my + ey > H;
    }
    DUST_PROF_END(vsmfast)
    if (edge) {
        DUST_PROF_BEGIN(vsmedge)
        s0 = dust_volume_small_edge(0u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr0, r0, r1, r3, t, W, H);
        s1 = dust_volume_small_edge(1u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr1, r0, r1, r3, t, W, H);
        s2 = dust_volume_small_edge(2u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr2, r0, r1, r3, t, W, H);
        if (five) {
            s3 = dust_volume_small_edge(3u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr3, r0, r1, r3, t, W, H);
            s4 = dust_volume_small_edge(4u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr4, r0, r1, r3, t, W, H);
        }
        DUST_PROF_END(vsmedge)
        DUST_PROF_COUNT(small_edge, 1u)
    } else {
        DUST_PROF_COUNT(small_fast, 1u)
    }
}

// the texels a small piece's points cover (their +-DUST_SMALL_COVER sigma) within its range r
// (dust::volume_scatter's small branch); empty when none
ivec4 dust_volume_small_range(DustSmallPoint s0, DustSmallPoint s1, DustSmallPoint s2, DustSmallPoint s3,
                              DustSmallPoint s4, ivec4 r, float sz) {
    float lx = 3.4e38, hx = -3.4e38, ly = 3.4e38, hy = -3.4e38;
    for (uint q = 0u; q < 5u; ++q) {
        DustSmallPoint p = q == 0u ? s0 : (q == 1u ? s1 : (q == 2u ? s2 : (q == 3u ? s3 : s4)));
        if (!(p.m > 0.0)) continue;
        float ex = DUST_SMALL_COVER / (p.ix * DUST_SQRT_2), ey = DUST_SMALL_COVER / (p.iy * DUST_SQRT_2);
        lx = min(lx, p.mx - ex);
        hx = max(hx, p.mx + ex);
        ly = min(ly, p.my - ey);
        hy = max(hy, p.my + ey);
    }
    if (!(lx <= hx && ly <= hy)) return ivec4(0, -1, 0, -1);
    return ivec4(int(max(floor(lx / sz), float(r.x))), int(min(floor(hx / sz), float(r.y))),
                 int(max(floor(ly / sz), float(r.z))), int(min(floor(hy / sz), float(r.w))));
}

// a small point's integral over the rectangle [xa, xb] x [ya, yb] (the product of its marginals and
// the first-order correlation term; dust::volume_scatter's small branch)
float dust_small_rect(DustSmallPoint q, float xa, float xb, float ya, float yb) {
    if (!(q.m > 0.0)) return 0.0;
    float uxa = (xa - q.mx) * q.ix, uxb = (xb - q.mx) * q.ix;
    float uya = (ya - q.my) * q.iy, uyb = (yb - q.my) * q.iy;
    float fx = dust_erf(uxb) - dust_erf(uxa);
    float fy = dust_erf(uyb) - dust_erf(uya);
    float gx = exp(-uxb * uxb) - exp(-uxa * uxa);
    float gy = exp(-uyb * uyb) - exp(-uya * uya);
    return q.m * (0.25 * fx * fy + q.rho * DUST_INV_2PI * gx * gy);
}

// One texel's deposit of tauPx (its mean optical depth per px^2) with the segment's carry
// (dust::volume_scatter's tail: the counts, the colour words, the nearest depth); true when written
// the most counts one volumetric deposit makes (dust::DUST_COUNT_TEXEL_MAX), in adds of at most
// DUST_COUNT_MAX_PER_ADD; the volume queue's header word counting the deposits split so
// (dust::DUST_VQUEUE_SPLIT_ADDS)
const float DUST_COUNT_TEXEL_MAX = 4.0e9;
const uint DUST_VQUEUE_SPLIT_ADDS = 12u;
uint dust_vqueue_base(DustUintBuffer pyr);
bool dust_volume_deposit(DustUintBuffer pyr, float tauPx, float area, float invUnit, vec3 color, uint depthBits,
                         uint off, uint lw, int ty, int tx, bool noWrites, inout float carry, inout float peak) {
    if (!(tauPx >= 0.0) || isinf(tauPx)) return false;
    peak = max(peak, tauPx);
    float tau = tauPx * area * invUnit;
    // the counts, split into adds of at most DUST_COUNT_MAX_PER_ADD (step F4: a bright piece on a
    // coarse texel was clipped at one add; below the cap one add, as before)
    float tauC = tau + carry;
    float counts = min(floor(tauC + 0.5), DUST_COUNT_TEXEL_MAX);
    carry = tauC - counts;
    if (!(counts >= 1.0) || noWrites) return false;
    uint i = off + (uint(ty) * lw + uint(tx)) * DUST_PYRAMID_TEXEL_WORDS;
    float hf = float(dust_hash32(i) >> 8u) * (1.0 / 16777216.0);
    if (counts > DUST_COUNT_MAX_PER_ADD) atomicAdd(pyr.v[dust_vqueue_base(pyr) + DUST_VQUEUE_SPLIT_ADDS], 1u);
    for (float rest = counts; rest > 0.5; ) {
        uint c = uint(min(rest, DUST_COUNT_MAX_PER_ADD));
        dust_atomic_add_sat(pyr, i, c);
        uvec3 cw = uvec3(float(c) * color * DUST_COLOR_SCALE + hf);
        dust_atomic_add_sat(pyr, i + 1u, cw.r);
        dust_atomic_add_sat(pyr, i + 2u, cw.g);
        dust_atomic_add_sat(pyr, i + 3u, cw.b);
        rest -= float(c);
    }
    uint cur = pyr.v[i + 4u];
    if (cur == 0u || depthBits < cur) {
        uint seen = atomicCompSwap(pyr.v[i + 4u], 0u, depthBits);
        if (seen != 0u) atomicMin(pyr.v[i + 4u], depthBits);
    }
    return true;
}

// One row segment [xa, xb] of row ty of a volumetric piece at one level (dust::volume_scatter,
// DUST_SEGMENT_TEXELS), from the segment's own dither phase; renorm = the support's flux over its
// kept mass, meanRef = the packet's mean (the photometry), sz = the texel size (px). One function
// per class (step D: the class is decided once per piece, no branch per texel, the small points
// live only in their own loop). The centre class: the exact ray integral at each texel centre.
// The texels written; the largest tau per px^2 into peak. Timing probe 2 (no_volume_writes)
// evaluates without writing.
bool dust_volume_segment(DustUintBuffer pyr, float invUnit, DustVolumePiece pc_, DustVolumeBound bd_, float renorm, vec3 meanRef,
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
        float tauPx = dust_volume_ray_tau(pc_, bd_, renorm, eye, aRow, bRow, r3, meanRef);
        any_ = dust_volume_deposit(pyr, tauPx, area, invUnit, color, depthBits, off, lw, ty, tx, noWrites, carry, peak)
            || any_;
    }
    return any_;
}

// The small class's row segment: each texel's in-frame rectangle integral of the three depth points
bool dust_volume_segment_small(DustUintBuffer pyr, float invUnit, float W, float H, float sz, vec3 color,
                               uint depthBits, uint off, uint lw, uint seed, int ty, int xa, int xb,
                               DustSmallPoint s0, DustSmallPoint s1, DustSmallPoint s2, DustSmallPoint s3,
                               DustSmallPoint s4, inout float peak) {
    bool noWrites = ((dust_pyr_split_word >> 30u) & 3u) == 2u;
    float area = sz * sz;
    float ya = max(float(ty) * sz, 0.0);
    float yb = min(float(ty + 1) * sz, H);
    float carry = dust_segment_carry(seed, (uint(ty) << 16u) | uint(xa));
    bool any_ = false;
    for (int tx = xa; tx <= xb; ++tx) {
        float xa_ = max(float(tx) * sz, 0.0);
        float xb_ = min(float(tx + 1) * sz, W);
        if (!(xb_ > xa_ && yb > ya)) continue;
        DUST_PROF_BEGIN(vsmall)
        float acc = dust_small_rect(s0, xa_, xb_, ya, yb) + dust_small_rect(s1, xa_, xb_, ya, yb)
                  + dust_small_rect(s2, xa_, xb_, ya, yb) + dust_small_rect(s3, xa_, xb_, ya, yb)
                  + dust_small_rect(s4, xa_, xb_, ya, yb);
        float tauPx = max(acc, 0.0) / ((xb_ - xa_) * (yb - ya));
        DUST_PROF_END(vsmall)
        any_ = dust_volume_deposit(pyr, tauPx, area, invUnit, color, depthBits, off, lw, ty, tx, noWrites, carry, peak)
            || any_;
    }
    return any_;
}

// The sub-ray and line classes' row segment: each texel's in-frame mean by kSub x kSub
// Gauss-Legendre sub-rays, or kSub lines across the band each integrated in closed form
bool dust_volume_segment_sub(DustUintBuffer pyr, float invUnit, DustVolumePiece pc_, DustVolumeBound bd_, float renorm, vec3 meanRef,
                             vec3 eye, vec3 r0, vec3 r1, vec3 r3, float W, float H, float sz, vec3 color,
                             uint depthBits, uint off, uint lw, uint seed, int ty, int xa, int xb, bool lines,
                             uint kSub, inout float peak) {
    bool noWrites = ((dust_pyr_split_word >> 30u) & 3u) == 2u;
    float area = sz * sz;
    float ya = max(float(ty) * sz, 0.0);
    float yb = min(float(ty + 1) * sz, H);
    float carry = dust_segment_carry(seed, (uint(ty) << 16u) | uint(xa));
    bool any_ = false;
    for (int tx = xa; tx <= xb; ++tx) {
        float xa_ = max(float(tx) * sz, 0.0);
        float xb_ = min(float(tx + 1) * sz, W);
        if (!(xb_ > xa_ && yb > ya)) continue;
        DUST_PROF_BEGIN(vsubline)
        float tauPx;
        if (lines) {
            bool alongX = pc_.line == 1u;
            float c = alongX ? 0.5 * (ya + yb) : 0.5 * (xa_ + xb_);
            float h = alongX ? 0.5 * (yb - ya) : 0.5 * (xb_ - xa_);
            float lo = alongX ? xa_ : ya, hi = alongX ? xb_ : yb;
            float acc = 0.0;
            for (uint j = 0u; j < kSub; ++j) {
                vec2 nw = dust_gl_node(kSub, j);
                acc += nw.y * dust_volume_line_integral(pc_, bd_, renorm, eye, r0, r1, r3, W, H, meanRef, alongX,
                                                        c + h * nw.x, lo, hi);
            }
            tauPx = acc / (hi - lo);
        } else {
            float cx = 0.5 * (xa_ + xb_), cy = 0.5 * (ya + yb);
            float hx = 0.5 * (xb_ - xa_), hy = 0.5 * (yb - ya);
            float acc = 0.0;
            for (uint j = 0u; j < kSub; ++j) {
                vec2 nwj = dust_gl_node(kSub, j);
                float ys = cy + hy * nwj.x;
                vec3 bs = r1 - r3 * (2.0 * ys / H - 1.0);
                for (uint i = 0u; i < kSub; ++i) {
                    vec2 nwi = dust_gl_node(kSub, i);
                    float xs = cx + hx * nwi.x;
                    vec3 as_ = r0 - r3 * (2.0 * xs / W - 1.0);
                    float tt = dust_volume_ray_tau(pc_, bd_, renorm, eye, as_, bs, r3, meanRef);
                    if (tt >= 0.0 && !isinf(tt)) acc += (nwi.y * nwj.y) * tt;
                }
            }
            tauPx = acc;
        }
        DUST_PROF_END(vsubline)
        any_ = dust_volume_deposit(pyr, tauPx, area, invUnit, color, depthBits, off, lw, ty, tx, noWrites, carry, peak)
            || any_;
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
    float fPx;
    DustVolumeBound bound;
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
    pl.fPx = fPx;
    // the packet's 3-sigma capsule (dust::volume_plan_pieces, step F2)
    if (mode == DUST_VOLUME_AXIS_TIME) {
        pl.bound = dust_volume_bound(axS, axMean - segB * 0.5, segB, eye);
    } else if (mode == DUST_VOLUME_AXIS_GRID) {
        vec3 hb = segB * 0.5;
        float sbv[6];
        sbv[0] = axS[0] + kf * hb.x * hb.x;
        sbv[1] = axS[1] + kf * hb.x * hb.y;
        sbv[2] = axS[2] + kf * hb.x * hb.z;
        sbv[3] = axS[3] + kf * hb.y * hb.y;
        sbv[4] = axS[4] + kf * hb.y * hb.z;
        sbv[5] = axS[5] + kf * hb.z * hb.z;
        pl.bound = dust_volume_bound(sbv, axMean, segA, eye);
    } else {
        pl.bound = dust_volume_bound(axS, axMean, segA, eye);
    }
    return true;
}

// ── The packet queue (dust::DUST_QUEUE_*) ──────────────────────────────────────────────────────
// It lives in the pyramid buffer after the measurement grid (PyramidLayout::total_words): a
// header of DUST_QUEUE_HEADER_WORDS words — the indirect dispatch's (x, y, 1), the raw append
// counter (min(raw, capacity) records held: the rest overflowed and were drawn in-thread), a
// capacity override (0 = DUST_QUEUE_CAPACITY), words 8..127 the profiler's counters
// (dust_profile.glsl, zero in production) — then DUST_QUEUE_RECORD_WORDS-word records.
const uint DUST_QUEUE_HEADER_WORDS = 128u;
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

// the queue's first word, in words from `pyr` (header word DUST_PYR_QUEUE_BASE, host-written: the
// frame's is its measure grid's end plus the class images, a class image's the same word less its
// own offset in the view buffer, so a draw into a class image appends to the view's queues)
uint dust_queue_base(DustUintBuffer pyr) {
    return pyr.v[DUST_PYR_QUEUE_BASE];
}

// the queue's capacity: DUST_QUEUE_CAPACITY, or less when header word 4 says so (a test hook,
// dust::Device::dust_set_queue_capacity; 0 = the default)
uint dust_queue_capacity(DustUintBuffer pyr, uint base) {
    uint c = pyr.v[base + 4u];
    return c == 0u ? DUST_QUEUE_CAPACITY : min(c, DUST_QUEUE_CAPACITY);
}

// workgroups per dispatch dimension every device supports (dust::DUST_QUEUE_DISPATCH_X): the
// draw's indirect dispatch is (min(count, X), ceil(count / X), 1), record y * X + x
const uint DUST_QUEUE_DISPATCH_X = 65535u;
// the records the queue holds this frame: the raw appends up to the capacity
uint dust_queue_held(DustUintBuffer pyr, uint base) {
    return min(pyr.v[base + 3u], dust_queue_capacity(pyr, base));
}

// A record slot, or ~0u when the queue is full (counted: raw counter past the capacity). The
// dispatch (host-initialised (0, 1, 1)) ends as the maxima over the held records r of
// (min(r + 1, X), r / X + 1): exactly (min(count, X), ceil(count / X)), whatever the order
uint dust_queue_append(DustUintBuffer pyr, uint base) {
    uint r = atomicAdd(pyr.v[base + 3u], 1u);
    if (r >= dust_queue_capacity(pyr, base)) return 0xFFFFFFFFu;
    atomicMax(pyr.v[base], min(r + 1u, DUST_QUEUE_DISPATCH_X));
    if (r >= DUST_QUEUE_DISPATCH_X) atomicMax(pyr.v[base + 1u], r / DUST_QUEUE_DISPATCH_X + 1u);
    return base + DUST_QUEUE_HEADER_WORDS + r * DUST_QUEUE_RECORD_WORDS;
}

void dust_queue_put_f(DustUintBuffer pyr, uint at, float v) { pyr.v[at] = floatBitsToUint(v); }
float dust_queue_get_f(DustUintBuffer pyr, uint at) { return uintBitsToFloat(pyr.v[at]); }

// ── The volume packet queue (dust::DUST_VQUEUE_*) ───────────────────────────────────────────────
// After the EWA queue (DUST_QUEUE_WORDS): a header of DUST_VQUEUE_HEADER_WORDS words — the
// consumer's indirect dispatch (x = ceil(records / 64), 1, 1), the raw append counter (reset per
// chunk by the host), a capacity override (0 = DUST_VQUEUE_CAPACITY), the chunk's cluster range
// [first, end) the producer threads cover, then the frame's counts (records drawn, overflowed:
// sent to the volume spill) — then DUST_VQUEUE_RECORD_WORDS-word records: the volumetric packets a chunk's
// producer threads emit (dust_splat.comp dust_volume_emit) with their own rounding seed, drawn one
// lane per record by dust_draw_volume.comp (the same dust_volume_draw, so the pyramid is the one
// the producer would have drawn, word for word; the atomics are order-free).
const uint DUST_VQUEUE_HEADER_WORDS = 16u;
const uint DUST_VQUEUE_RECORD_WORDS = 32u;
const uint DUST_VQUEUE_CAPACITY = 262144u;
const uint DUST_VQUEUE_FIRST = 5u;
const uint DUST_VQUEUE_END = 6u;
const uint DUST_VQUEUE_DRAWN = 7u;
const uint DUST_VQUEUE_OVERFLOWED = 8u;
const uint DUST_VQUEUE_RECORDS = 9u;
// the tier's live range start (ring slot) and ring mask (host-written per chunk; dust_splat.comp)
const uint DUST_VQUEUE_FIRST_SLOT = 10u;
const uint DUST_VQUEUE_RING_MASK = 11u;
const uint DUST_QUEUE_WORDS = DUST_QUEUE_HEADER_WORDS + DUST_QUEUE_CAPACITY * DUST_QUEUE_RECORD_WORDS;
// PYR_FLAGS bit (host-written, dust::DUST_FRAME_VQUEUE_OFF): the main volume queue refuses every
// record (all through the volume spill: the spill path's A/B)
const uint DUST_FRAME_VQUEUE_OFF = 1u << 30u;
// PYR_FLAGS bit (a test hook, dust::DUST_FRAME_SPLIT_SERIAL): dust_split.comp walks a cluster's
// records in one lane with a running ordinal (the former per-cluster thread's order), the
// word-identical reference for its counting pass and prefix
const uint DUST_FRAME_SPLIT_SERIAL = 1u << 31u;

uint dust_vqueue_base(DustUintBuffer pyr) { return dust_queue_base(pyr) + DUST_QUEUE_WORDS; }
uint dust_vqueue_capacity(DustUintBuffer pyr, uint base) {
    uint c = pyr.v[base + 4u];
    return c == 0u ? DUST_VQUEUE_CAPACITY : min(c, DUST_VQUEUE_CAPACITY);
}
// A record index, or ~0u when the chunk's queue is full (the main queue's refusal goes to the
// volume spill; the spill's is dropped and counted)
uint dust_vqueue_append(DustUintBuffer pyr, uint base) {
    uint r = atomicAdd(pyr.v[base + 3u], 1u);
    if (r >= dust_vqueue_capacity(pyr, base)) return 0xFFFFFFFFu;
    atomicMax(pyr.v[base], (r >> 6u) + 1u);
    return r;
}
// The volume spill (dust::dust_vspill_records): a second volume queue in the same format (header
// words 0-2 its consumer's indirect dispatch, 3 its raw append counter, 4 its capacity; the
// records structure-of-arrays at its capacity's stride) after the speck spill, at the main
// header's DUST_VQUEUE_SPILL_OFFSET (host-written per frame). The records the main queue refuses
// go there (dust_volume_emit), drained by a second dispatch of dust_draw_volume.comp
// (DUST_VIEW_VOLUME_SPILL); a record it refuses too is dropped and counted (DUST_VQUEUE_LOST).
// The frame's counts stay in the main header: words 13 (spill drawn) and 14 (lost).
const uint DUST_VQUEUE_SPILL_DRAWN = 13u;
const uint DUST_VQUEUE_LOST = 14u;
const uint DUST_VQUEUE_SPILL_OFFSET = 15u;
uint dust_vspill_base(DustUintBuffer pyr) {
    uint vbase = dust_vqueue_base(pyr);
    return vbase + pyr.v[vbase + DUST_VQUEUE_SPILL_OFFSET];
}
// A volume record, structure-of-arrays: word w of record r at base + DUST_VQUEUE_HEADER_WORDS +
// w * capacity + r, so a warp's 32 consecutive records load and store each word from 128
// contiguous bytes (one transaction a word; the former 128-B stride made 32). Words: 0-2 mean,
// 3-8 S, 9-11 seg, 12 flux, 13-14 rho, 15-17 segT, 18 seed, 19 exposure, 20-22 colour, 23-26
// the support plane, 27-29 the Sun direction, 30 r_h (AU), 31 open ends (bit 0 start, bit 1
// end). Macros: a struct argument would be copied per call.
#define DUST_VQUEUE_WORD(base, cap, r, w) ((base) + DUST_VQUEUE_HEADER_WORDS + (w) * (cap) + (r))

// ── The sub-packet queue (dust::DUST_SQUEUE_*) ──────────────────────────────────────────────────
// After the volume queue (DUST_VQUEUE_WORDS): per chunk, the packets dust_splat.comp's lanes
// construct (one record per dust_project_split_scatter call, in the cluster's sub-packet order),
// split and drawn by dust_split.comp (a workgroup per cluster, dust_leaf.glsl). A header of
// DUST_SQUEUE_HEADER_WORDS (the record counter, the capacity, the records that overflowed), a
// table of DUST_SQUEUE_TABLE_WORDS per cluster of the chunk (the cluster's first record, its
// count, its index) for DUST_SQUEUE_CLUSTERS clusters, then the records, structure-of-arrays
// (word w of record r at the records' base + w * capacity + r). A cluster makes at most
// DUST_SQUEUE_PER_CLUSTER records (17 sub-packets x DUST_TIME_PIECES_MAX), so a chunk of
// DUST_SQUEUE_CLUSTERS never overflows the capacity.
const uint DUST_SQUEUE_HEADER_WORDS = 16u;
const uint DUST_SQUEUE_TABLE_WORDS = 4u;
const uint DUST_SQUEUE_CLUSTERS = 1024u;
const uint DUST_SQUEUE_PER_CLUSTER = 136u;
const uint DUST_SQUEUE_CAPACITY = DUST_SQUEUE_CLUSTERS * DUST_SQUEUE_PER_CLUSTER;
const uint DUST_SQUEUE_RECORD_WORDS = 36u;
const uint DUST_VQUEUE_WORDS = DUST_VQUEUE_HEADER_WORDS + DUST_VQUEUE_CAPACITY * DUST_VQUEUE_RECORD_WORDS;
// After the volume records this view allocated (its header's DUST_VQUEUE_RECORDS word,
// host-written each drawn frame, 0 = DUST_VQUEUE_CAPACITY: a device with less memory allocates
// fewer, dust::dust_queue_offsets places the same words)
uint dust_squeue_base(DustUintBuffer pyr) {
    uint vbase = dust_vqueue_base(pyr);
    uint n = pyr.v[vbase + DUST_VQUEUE_RECORDS];
    n = n == 0u ? DUST_VQUEUE_CAPACITY : min(n, DUST_VQUEUE_CAPACITY);
    return vbase + DUST_VQUEUE_HEADER_WORDS + n * DUST_VQUEUE_RECORD_WORDS;
}
uint dust_squeue_table(uint base) { return base + DUST_SQUEUE_HEADER_WORDS; }
// the speck walk's spill after the sub-packet queue (dust::DUST_SPILL_*): a header (word 0 the
// slots claimed this pass, word 1 the slots the view has), then the slots of DUST_SPILL_NODES
// nodes of DUST_SPILL_NODE_WORDS words each
const uint DUST_SQUEUE_WORDS = DUST_SQUEUE_HEADER_WORDS + DUST_SQUEUE_CLUSTERS * DUST_SQUEUE_TABLE_WORDS + DUST_SQUEUE_CAPACITY * DUST_SQUEUE_RECORD_WORDS;
const uint DUST_SPILL_HEADER_WORDS = 16u;
const uint DUST_SPILL_NODE_WORDS = 6u;
const uint DUST_SPILL_NODES = 1024u;
uint dust_spill_base(DustUintBuffer pyr) { return dust_squeue_base(pyr) + DUST_SQUEUE_WORDS; }
// the speck trace after the spill's slots (dust::DUST_SPECKS_TRACE_*): (hash, dp xyz, w xyz,
// coordinates xyz, kappa, node level) per speck drawn while the spill header's word 2 (the
// capacity) is set, word 3 counting
const uint DUST_SPECKS_TRACE_WORDS = 12u;
uint dust_specks_trace_base(uint spillBase, uint slots) { return spillBase + DUST_SPILL_HEADER_WORDS + slots * DUST_SPILL_NODES * DUST_SPILL_NODE_WORDS; }
uint dust_squeue_records(uint base) { return base + DUST_SQUEUE_HEADER_WORDS + DUST_SQUEUE_CLUSTERS * DUST_SQUEUE_TABLE_WORDS; }
// A sub-packet record: 0-2 mean, 3-8 S, 9-11 the segment d, 12 flux, 13 exposure, 14 flags
// (bit 0 openStart, 1 openEnd, 2 the support has an origin), 15-16 rho, 17-19 segT, 20-22
// colour, 23-26 the support plane, 27 its origin, 28-30 the Sun direction, 31 r_h, 32 the
// cluster's index (its seed). Macros: a struct argument would be copied per call.
#define DUST_SQUEUE_PUT(pyr, recs, r, mean_, S_, d_, flux_, exposure_, openStart_, openEnd_, rho_, segT_, color_, support_, hasO_, origin_, sun_, rh_, i_, walked_) { \
    uint a_ = (recs) + (r); const uint c_ = DUST_SQUEUE_CAPACITY; \
    pyr.v[a_ + 0u * c_] = floatBitsToUint((mean_).x); pyr.v[a_ + 1u * c_] = floatBitsToUint((mean_).y); pyr.v[a_ + 2u * c_] = floatBitsToUint((mean_).z); \
    pyr.v[a_ + 3u * c_] = floatBitsToUint((S_)[0]); pyr.v[a_ + 4u * c_] = floatBitsToUint((S_)[1]); pyr.v[a_ + 5u * c_] = floatBitsToUint((S_)[2]); \
    pyr.v[a_ + 6u * c_] = floatBitsToUint((S_)[3]); pyr.v[a_ + 7u * c_] = floatBitsToUint((S_)[4]); pyr.v[a_ + 8u * c_] = floatBitsToUint((S_)[5]); \
    pyr.v[a_ + 9u * c_] = floatBitsToUint((d_).x); pyr.v[a_ + 10u * c_] = floatBitsToUint((d_).y); pyr.v[a_ + 11u * c_] = floatBitsToUint((d_).z); \
    pyr.v[a_ + 12u * c_] = floatBitsToUint(flux_); \
    pyr.v[a_ + 13u * c_] = floatBitsToUint(exposure_); \
    pyr.v[a_ + 14u * c_] = ((openStart_) ? 1u : 0u) | ((openEnd_) ? 2u : 0u) | ((hasO_) ? 4u : 0u) | ((walked_) ? 8u : 0u); \
    pyr.v[a_ + 15u * c_] = floatBitsToUint((rho_).x); pyr.v[a_ + 16u * c_] = floatBitsToUint((rho_).y); \
    pyr.v[a_ + 17u * c_] = floatBitsToUint((segT_).x); pyr.v[a_ + 18u * c_] = floatBitsToUint((segT_).y); pyr.v[a_ + 19u * c_] = floatBitsToUint((segT_).z); \
    pyr.v[a_ + 20u * c_] = floatBitsToUint((color_).x); pyr.v[a_ + 21u * c_] = floatBitsToUint((color_).y); pyr.v[a_ + 22u * c_] = floatBitsToUint((color_).z); \
    pyr.v[a_ + 23u * c_] = floatBitsToUint((support_).x); pyr.v[a_ + 24u * c_] = floatBitsToUint((support_).y); \
    pyr.v[a_ + 25u * c_] = floatBitsToUint((support_).z); pyr.v[a_ + 26u * c_] = floatBitsToUint((support_).w); \
    pyr.v[a_ + 27u * c_] = floatBitsToUint(origin_); \
    pyr.v[a_ + 28u * c_] = floatBitsToUint((sun_).x); pyr.v[a_ + 29u * c_] = floatBitsToUint((sun_).y); pyr.v[a_ + 30u * c_] = floatBitsToUint((sun_).z); \
    pyr.v[a_ + 31u * c_] = floatBitsToUint(rh_); \
    pyr.v[a_ + 32u * c_] = (i_); \
}
#define DUST_SQUEUE_GET(pyr, recs, r, mean_, S_, d_, flux_, exposure_, openStart_, openEnd_, rho_, segT_, color_, support_, hasO_, origin_, sun_, rh_, i_, walked_) { \
    uint a_ = (recs) + (r); const uint c_ = DUST_SQUEUE_CAPACITY; \
    mean_ = vec3(uintBitsToFloat(pyr.v[a_ + 0u * c_]), uintBitsToFloat(pyr.v[a_ + 1u * c_]), uintBitsToFloat(pyr.v[a_ + 2u * c_])); \
    S_[0] = uintBitsToFloat(pyr.v[a_ + 3u * c_]); S_[1] = uintBitsToFloat(pyr.v[a_ + 4u * c_]); S_[2] = uintBitsToFloat(pyr.v[a_ + 5u * c_]); \
    S_[3] = uintBitsToFloat(pyr.v[a_ + 6u * c_]); S_[4] = uintBitsToFloat(pyr.v[a_ + 7u * c_]); S_[5] = uintBitsToFloat(pyr.v[a_ + 8u * c_]); \
    d_ = vec3(uintBitsToFloat(pyr.v[a_ + 9u * c_]), uintBitsToFloat(pyr.v[a_ + 10u * c_]), uintBitsToFloat(pyr.v[a_ + 11u * c_])); \
    flux_ = uintBitsToFloat(pyr.v[a_ + 12u * c_]); \
    exposure_ = uintBitsToFloat(pyr.v[a_ + 13u * c_]); \
    { uint f_ = pyr.v[a_ + 14u * c_]; openStart_ = (f_ & 1u) != 0u; openEnd_ = (f_ & 2u) != 0u; hasO_ = (f_ & 4u) != 0u; walked_ = (f_ & 8u) != 0u; } \
    rho_ = vec2(uintBitsToFloat(pyr.v[a_ + 15u * c_]), uintBitsToFloat(pyr.v[a_ + 16u * c_])); \
    segT_ = vec3(uintBitsToFloat(pyr.v[a_ + 17u * c_]), uintBitsToFloat(pyr.v[a_ + 18u * c_]), uintBitsToFloat(pyr.v[a_ + 19u * c_])); \
    color_ = vec3(uintBitsToFloat(pyr.v[a_ + 20u * c_]), uintBitsToFloat(pyr.v[a_ + 21u * c_]), uintBitsToFloat(pyr.v[a_ + 22u * c_])); \
    support_ = vec4(uintBitsToFloat(pyr.v[a_ + 23u * c_]), uintBitsToFloat(pyr.v[a_ + 24u * c_]), uintBitsToFloat(pyr.v[a_ + 25u * c_]), uintBitsToFloat(pyr.v[a_ + 26u * c_])); \
    origin_ = uintBitsToFloat(pyr.v[a_ + 27u * c_]); \
    sun_ = vec3(uintBitsToFloat(pyr.v[a_ + 28u * c_]), uintBitsToFloat(pyr.v[a_ + 29u * c_]), uintBitsToFloat(pyr.v[a_ + 30u * c_])); \
    rh_ = uintBitsToFloat(pyr.v[a_ + 31u * c_]); \
    i_ = pyr.v[a_ + 32u * c_]; \
}
#define DUST_VQUEUE_PUT(pyr, base, cap, r, p, seed, exposure, color, support, sun, rh) { \
    uint a_ = (base) + DUST_VQUEUE_HEADER_WORDS + (r); uint c_ = (cap); \
    pyr.v[a_ + 0u * c_] = floatBitsToUint((p).mean.x); pyr.v[a_ + 1u * c_] = floatBitsToUint((p).mean.y); pyr.v[a_ + 2u * c_] = floatBitsToUint((p).mean.z); \
    pyr.v[a_ + 3u * c_] = floatBitsToUint((p).S[0]); pyr.v[a_ + 4u * c_] = floatBitsToUint((p).S[1]); pyr.v[a_ + 5u * c_] = floatBitsToUint((p).S[2]); \
    pyr.v[a_ + 6u * c_] = floatBitsToUint((p).S[3]); pyr.v[a_ + 7u * c_] = floatBitsToUint((p).S[4]); pyr.v[a_ + 8u * c_] = floatBitsToUint((p).S[5]); \
    pyr.v[a_ + 9u * c_] = floatBitsToUint((p).seg.x); pyr.v[a_ + 10u * c_] = floatBitsToUint((p).seg.y); pyr.v[a_ + 11u * c_] = floatBitsToUint((p).seg.z); \
    pyr.v[a_ + 12u * c_] = floatBitsToUint((p).flux); \
    pyr.v[a_ + 13u * c_] = floatBitsToUint((p).rho.x); pyr.v[a_ + 14u * c_] = floatBitsToUint((p).rho.y); \
    pyr.v[a_ + 15u * c_] = floatBitsToUint((p).segT.x); pyr.v[a_ + 16u * c_] = floatBitsToUint((p).segT.y); pyr.v[a_ + 17u * c_] = floatBitsToUint((p).segT.z); \
    pyr.v[a_ + 18u * c_] = (seed); \
    pyr.v[a_ + 19u * c_] = floatBitsToUint(exposure); \
    pyr.v[a_ + 20u * c_] = floatBitsToUint((color).x); pyr.v[a_ + 21u * c_] = floatBitsToUint((color).y); pyr.v[a_ + 22u * c_] = floatBitsToUint((color).z); \
    pyr.v[a_ + 23u * c_] = floatBitsToUint((support).x); pyr.v[a_ + 24u * c_] = floatBitsToUint((support).y); \
    pyr.v[a_ + 25u * c_] = floatBitsToUint((support).z); pyr.v[a_ + 26u * c_] = floatBitsToUint((support).w); \
    pyr.v[a_ + 27u * c_] = floatBitsToUint((sun).x); pyr.v[a_ + 28u * c_] = floatBitsToUint((sun).y); pyr.v[a_ + 29u * c_] = floatBitsToUint((sun).z); \
    pyr.v[a_ + 30u * c_] = floatBitsToUint(rh); \
    pyr.v[a_ + 31u * c_] = ((p).openStart ? 1u : 0u) | ((p).openEnd ? 2u : 0u); \
}
#define DUST_VQUEUE_GET(pyr, base, cap, r, p, seed, exposure, color, support, sun, rh) { \
    uint a_ = (base) + DUST_VQUEUE_HEADER_WORDS + (r); uint c_ = (cap); \
    (p).mean = vec3(uintBitsToFloat(pyr.v[a_ + 0u * c_]), uintBitsToFloat(pyr.v[a_ + 1u * c_]), uintBitsToFloat(pyr.v[a_ + 2u * c_])); \
    (p).S[0] = uintBitsToFloat(pyr.v[a_ + 3u * c_]); (p).S[1] = uintBitsToFloat(pyr.v[a_ + 4u * c_]); (p).S[2] = uintBitsToFloat(pyr.v[a_ + 5u * c_]); \
    (p).S[3] = uintBitsToFloat(pyr.v[a_ + 6u * c_]); (p).S[4] = uintBitsToFloat(pyr.v[a_ + 7u * c_]); (p).S[5] = uintBitsToFloat(pyr.v[a_ + 8u * c_]); \
    (p).seg = vec3(uintBitsToFloat(pyr.v[a_ + 9u * c_]), uintBitsToFloat(pyr.v[a_ + 10u * c_]), uintBitsToFloat(pyr.v[a_ + 11u * c_])); \
    (p).flux = uintBitsToFloat(pyr.v[a_ + 12u * c_]); \
    (p).rho = vec2(uintBitsToFloat(pyr.v[a_ + 13u * c_]), uintBitsToFloat(pyr.v[a_ + 14u * c_])); \
    (p).segT = vec3(uintBitsToFloat(pyr.v[a_ + 15u * c_]), uintBitsToFloat(pyr.v[a_ + 16u * c_]), uintBitsToFloat(pyr.v[a_ + 17u * c_])); \
    seed = pyr.v[a_ + 18u * c_]; \
    exposure = uintBitsToFloat(pyr.v[a_ + 19u * c_]); \
    color = vec3(uintBitsToFloat(pyr.v[a_ + 20u * c_]), uintBitsToFloat(pyr.v[a_ + 21u * c_]), uintBitsToFloat(pyr.v[a_ + 22u * c_])); \
    support = vec4(uintBitsToFloat(pyr.v[a_ + 23u * c_]), uintBitsToFloat(pyr.v[a_ + 24u * c_]), uintBitsToFloat(pyr.v[a_ + 25u * c_]), uintBitsToFloat(pyr.v[a_ + 26u * c_])); \
    sun = vec3(uintBitsToFloat(pyr.v[a_ + 27u * c_]), uintBitsToFloat(pyr.v[a_ + 28u * c_]), uintBitsToFloat(pyr.v[a_ + 29u * c_])); \
    rh = uintBitsToFloat(pyr.v[a_ + 30u * c_]); \
    (p).openStart = (pyr.v[a_ + 31u * c_] & 1u) != 0u; (p).openEnd = (pyr.v[a_ + 31u * c_] & 2u) != 0u; \
}

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
