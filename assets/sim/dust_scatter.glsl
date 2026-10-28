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
float dust_queue_get_f(DustUintBuffer pyr, uint at);
DustScreenPacket dust_queue_get_screen_packet(DustUintBuffer pyr, uint k);
// Round 9 step Z: the workgroup consumer's segment in two passes over its texels, the capsule's
// tau of each texel into the lane's row of dust_q_tau, then the photometry and the deposit
// (dust_scatter_segment's texel arithmetic in its order: capsule x area x invUnit, x the weight
// when lit, the rounding with the carry). The record's packet is read from the queue at each pass
// (reads of the pyramid buffer inside a loop with its atomics stay where they are), so a pass holds
// its own fields of it: the capsule's twelve, then the photometry's twenty, never the whole packet
// over the texel loop (Mali: 184 B of spill)
#ifndef DUST_DRAW_LEAF  // (_ws) not in the leaf consumer
shared float dust_q_tau[64u * DUST_SEGMENT_TEXELS];
bool dust_scatter_segment_wg(DustUintBuffer pyr, uint at, float s, uint off, uint w, uint seed, int ty, int xa, int xb, uint lane) {
    float yc = (float(ty) + 0.5) * s;
    float area = s * s;
    uint row = lane * DUST_SEGMENT_TEXELS;
    {
        DustScreenPacket P = dust_queue_get_screen_packet(pyr, at + 11u);
        float invUnit = dust_queue_get_f(pyr, at + 10u);
        for (int tx = xa; tx <= xb; ++tx) {
            vec2 x = vec2((float(tx) + 0.5) * s, yc);
            dust_q_tau[row + uint(tx - xa)] = dust_capsule_tau(P, x) * area * invUnit;
        }
    }
    {
        DustScreenPacket P = dust_queue_get_screen_packet(pyr, at + 11u);
        if (P.albedo > 0.0) {
            for (int tx = xa; tx <= xb; ++tx) {
                vec2 x = vec2((float(tx) + 0.5) * s, yc);
                dust_q_tau[row + uint(tx - xa)] *= dust_texel_weight(P, x);
            }
        }
    }
    vec3 color = vec3(dust_queue_get_f(pyr, at + 7u), dust_queue_get_f(pyr, at + 8u), dust_queue_get_f(pyr, at + 9u));
    uint depthBits = floatBitsToUint(max(dust_queue_get_f(pyr, at + 11u + 10u), 0.0));
    bool any_ = false;
    float carry = dust_segment_carry(seed, (uint(ty) << 16u) | uint(xa));
    for (int tx = xa; tx <= xb; ++tx) {
        float tau = dust_q_tau[row + uint(tx - xa)];
        if (!(tau >= 0.0) || isinf(tau)) continue;
        float tauC = min(tau, DUST_COUNT_MAX_PER_ADD) + carry;
        uint counts = uint(tauC + 0.5);
        carry = tauC - float(counts);
        if (counts == 0u) continue;
        uint i = off + (uint(ty) * w + uint(tx)) * DUST_PYRAMID_TEXEL_WORDS;
        dust_atomic_add_sat(pyr, i, counts);
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
        any_ = true;
    }
    return any_;
}
#endif  // !DUST_DRAW_LEAF
// (perf9_O) a blob may set the chunk's rows lower (dust_draw_leaf.comp's workgroup blob: its shared
// budget); the rows, segments and carries are the same at any chunk size
#ifndef DUST_QUEUE_ROW_CHUNK_N
#define DUST_QUEUE_ROW_CHUNK_N 256u
#endif
const uint DUST_QUEUE_ROW_CHUNK = DUST_QUEUE_ROW_CHUNK_N;
shared int  dust_q_x0[DUST_QUEUE_ROW_CHUNK];
shared int  dust_q_x1[DUST_QUEUE_ROW_CHUNK];
shared uint dust_q_seg[DUST_QUEUE_ROW_CHUNK + 1u];
#ifndef DUST_DRAW_LEAF  // (_ws) the queue-record rows: not in the leaf consumer
bool dust_scatter_rows_wg(DustUintBuffer pyr, uint at, float s,
                          uint off, uint w, uint h, uint seed, int rowA, int rowB, uint lanes, uint lane) {
    DUST_PROF_BEGIN(qrows)
    int y0, y1;
    if (!dust_band_rows(dust_band_new(dust_queue_get_screen_packet(pyr, at + 11u)), s, h, y0, y1)) { DUST_PROF_END(qrows) return false; }
    y0 = max(y0, rowA);
    y1 = min(y1, rowB);
    bool any_ = false;
    for (int yc0 = y0; yc0 <= y1; yc0 += int(DUST_QUEUE_ROW_CHUNK)) {
        DustBand B = dust_band_new(dust_queue_get_screen_packet(pyr, at + 11u));
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
            any_ = dust_scatter_segment_wg(pyr, at, s, off, w, seed, yc0 + r, xa, xb, lane) || any_;
            DUST_PROF_END_AS(qseg, seg)
            DUST_PROF_COUNT(qtexels, xb - xa + 1)
        }
        barrier();
    }
    DUST_PROF_END(qrows)
    return any_;
}
#endif  // !DUST_DRAW_LEAF
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
// The covariance every piece of a packet has (dust::volume_pieces: the base S plus the spacing
// terms of the axes, the same for every piece k; only the mean and the flux depend on k), its
// Cholesky factor Lk and its diagonal skd = (Sk[0], Sk[3], Sk[5]). Split out of
// dust_volume_piece (step 9C) so that the volume consumer's pass B builds it once and keeps it in
// the lane's workgroup slots; dust_volume_piece = dust_volume_piece_cov + dust_volume_piece_at,
// the same operations in the same order
void dust_volume_piece_cov(float S[6], vec3 segA, uint nA, float cvarA, vec3 segB, uint nB, float cvarB, bool hasB,
                           out float Lk[6], out vec3 skd) {
    float cvar = cvarA;
    float invN = 1.0 / float(nA);
    vec3 h = segA * invN;
    float invB = 1.0 / float(nB);
    vec3 hb = segB * invB;
    float Sk[6];
    Sk[0] = S[0] + cvar * h.x * h.x;
    Sk[1] = S[1] + cvar * h.x * h.y;
    Sk[2] = S[2] + cvar * h.x * h.z;
    Sk[3] = S[3] + cvar * h.y * h.y;
    Sk[4] = S[4] + cvar * h.y * h.z;
    Sk[5] = S[5] + cvar * h.z * h.z;
    if (hasB) {
        Sk[0] += cvarB * hb.x * hb.x;
        Sk[1] += cvarB * hb.x * hb.y;
        Sk[2] += cvarB * hb.x * hb.z;
        Sk[3] += cvarB * hb.y * hb.y;
        Sk[4] += cvarB * hb.y * hb.z;
        Sk[5] += cvarB * hb.z * hb.z;
    }
    dust_cholesky3(Sk, Lk);
    skd = vec3(Sk[0], Sk[3], Sk[5]);
}
// piece k from its packet's shared covariance (Lk, skd: dust_volume_piece_cov)
DustVolumePiece dust_volume_piece_at(uint k, float flux, vec2 rho, float exposure, vec3 axMean, float Lk[6], vec3 skd,
                                     vec3 segA, uint nA, vec3 segB, uint nB, bool hasB, float W, float H,
                                     vec3 r0, vec3 r1, vec3 r3, vec3 t, vec3 eye, float fPx, uint levels,
                                     bool withLevel) {
    float t0 = t.x, t1 = t.y, t3 = t.z;
    float invN = 1.0 / float(nA);
    float invB = 1.0 / float(nB);
    uint i = k / nB, j = k % nB;
    float uk = (float(i) + 0.5) * invN;
    vec3 meanK = axMean + segA * uk;
    float fluxK = flux * (rho.x + (rho.y - rho.x) * uk) * invN;
    if (hasB) {
        float vk = (float(j) + 0.5) * invB - 0.5;
        meanK = meanK + segB * vk;
        fluxK *= invB;
    }
    vec3 qw = dust_cholesky_solve3(Lk, eye - meanK);
    bool full = dot(qw, qw) < DUST_VOLUME_SIGMAS * DUST_VOLUME_SIGMAS;
    vec3 ext = vec3(3.0 * sqrt(skd.x), 3.0 * sqrt(skd.y), 3.0 * sqrt(skd.z));
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
        float sigK = 2.0 * (Lk[0] * Lk[3] * Lk[5]) / (skd.x + skd.y + skd.z);
        vec3 dq = eye - meanK;
        float denK = max(sqrt(dot(dq, dq)), sigK);
        sigPx = min((denK > 0.0 ? sigK / denK : 1.0) * fPx, sigScr);
    }
    q.lp = dust_volume_level(min(sigPx, 1e30), levels);
    q.line = line;
    q.ls = line != 0u ? dust_volume_level(min(sigScr, 1e30), levels) : 0xFFFFFFFFu;
    return q;
}
DustVolumePiece dust_volume_piece(uint k, float flux, vec2 rho, float exposure, vec3 axMean, float S[6], vec3 segA,
                                  uint nA, float cvarA, vec3 segB, uint nB, float cvarB, bool hasB, float W, float H,
                                  vec3 r0, vec3 r1, vec3 r3, vec3 t, vec3 eye, float fPx, uint levels,
                                  bool withLevel) {
    float Lk[6];
    vec3 skd;
    dust_volume_piece_cov(S, segA, nA, cvarA, segB, nB, cvarB, hasB, Lk, skd);
    return dust_volume_piece_at(k, flux, rho, exposure, axMean, Lk, skd, segA, nA, segB, nB, hasB, W, H, r0, r1, r3,
                                t, eye, fPx, levels, withLevel);
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

// Track 9A, the edge-piece demotion (dust::DUST_SMALL_DEMOTE_EPS, dust::volume_small_demote_err):
// an edge piece keeps its one projection a slice (with its predicted second-order terms,
// dust_small_demote_slice) when its gate (dust_small_demote_gate) is under DUST_SMALL_DEMOTE_GATE x
// the constant and its predicted in-frame light change, sigma
// points against one projection, mass-weighted over its slices, is under the constant (strict; 0:
// the branch folds away, D3's decisions as before)
const float DUST_SMALL_DEMOTE_EPS = 0.001;
const float DUST_SMALL_DEMOTE_GATE = 0.5;
const float DUST_SMALL_DEMOTE_KAPPA = 0.1;
const float DUST_SMALL_DEMOTE_SAFETY = 1.25;
const float DUST_SMALL_INV_SQRT_2PI = 0.3989423;
const float DUST_SMALL_DEMOTE_SLOPE_MAX = 0.60653066;

// one frame edge of a demoted slice's marginal (dust::small_demote_edge): the edge tv sigma from
// the mean, its move dt = u + |tv| al; the marginal's change bound dP, the Gaussian's dG, e^(-a^2/2)
// at the closest approach a, the marginal's upper bound pu (Mills beyond 1 sigma outside)
void dust_small_demote_edge(float tv, float u, float al, out float dP, out float dG, out float ea, out float pu) {
    float dt = u + abs(tv) * al;
    float a = max(abs(tv) - dt, 0.0);
    ea = exp(-0.5 * a * a);
    dP = DUST_SMALL_INV_SQRT_2PI * dt * ea;
    dG = dt * (a >= 1.0 ? a * ea : DUST_SMALL_DEMOTE_SLOPE_MAX);
    pu = tv < -1.0 ? DUST_SMALL_INV_SQRT_2PI * ea / -tv : 1.0;
}

// Track 9A (t9a2c): the demotion's gate (dust::small_demote_gate): from the centre slice's one
// projection c and its sigma points' relative depth spread kap = sqrt2 max|r3.lat_i| / w (the first
// order of every predicted term), the in-frame light an edge of the frame can carry over
// (kap / sqrt(2 pi)) (e^(-ax^2/2) py + px e^(-ay^2/2)), ax, ay the signed distances (sigma) from the
// mean to the nearer edge of each axis, px, py the marginals' upper bounds (Mills beyond 1 sigma
// outside); 3.4e38 when the centre is behind the eye or kap is out of the predictor's range. A piece
// is predicted only when this is under DUST_SMALL_DEMOTE_GATE x DUST_SMALL_DEMOTE_EPS: about thirty
// operations and two exponentials in place of the predictor for most of the edge pieces it would not
// demote (part of the rule: the CPU applies the same gate)
float dust_small_demote_gate(DustSmallPoint c, vec3 mean, vec3 r3, float t3, float w1, float w2, float W, float H) {
    float w0 = dot(r3, mean) + t3;
    if (!(w0 > 0.0)) return 3.4e38;
    // (the centre slice's kappa_i as dust_small_demote_terms computes them)
    float iw = 1.0 / w0;
    float kap = DUST_UT2_NODE * max(abs(w1 * iw), abs(w2 * iw));
    if (!(kap < DUST_SMALL_DEMOTE_KAPPA)) return 3.4e38;
    float isx = c.ix * DUST_SQRT_2, isy = c.iy * DUST_SQRT_2;
    float ax = min(c.mx, W - c.mx) * isx;
    float ay = min(c.my, H - c.my) * isy;
    float ex = exp(-0.5 * ax * ax);
    float ey = exp(-0.5 * ay * ay);
    float pxu = ax < -1.0 ? DUST_SMALL_INV_SQRT_2PI * ex / -ax : 1.0;
    float pyu = ay < -1.0 ? DUST_SMALL_INV_SQRT_2PI * ey / -ay : 1.0;
    return kap * DUST_SMALL_INV_SQRT_2PI * (ex * pyu + pxu * ey);
}

// Track 9A: one predicted slice's second-order terms (dust::SmallDemoteTerms): the mass term mu,
// the J3-weighted mean shift dx, dy (px), the one projection's lateral covariance cxx, cyy, cxy
// (px^2) and the sigma points' change of it dcxx, dcyy, dcxy, the one projection's mean mx, my (px)
struct DustSmallTerms {
    float mu, dx, dy, cxx, cyy, cxy, dcxx, dcyy, dcxy, mx, my;
};

// The second-order terms of slice q (dust::small_demote_terms); false when not predicted (behind
// the eye, sqrt2 |kappa_i| >= DUST_SMALL_DEMOTE_KAPPA, no Jacobian; k is then unset)
bool dust_small_demote_terms(uint q, bool five, vec3 mean, vec3 g, float x1, float x2, float y1, float y2,
                             float w1, float w2, vec3 r0, vec3 r1, vec3 r3, vec3 t, float W, float H,
                             out DustSmallTerms k) {
    vec3 xq = mean + g * dust_small_node(q, five).x;
    float w = dot(r3, xq) + t.z;
    if (!(w > 0.0)) return false;
    float iw = 1.0 / w;
    float k1 = w1 * iw, k2 = w2 * iw;
    if (!(DUST_UT2_NODE * max(abs(k1), abs(k2)) < DUST_SMALL_DEMOTE_KAPPA)) return false;
    float ndx = (dot(r0, xq) + t.x) * iw;
    float ndy = (dot(r1, xq) + t.y) * iw;
    vec3 c = cross(r0 - r3 * ndx, r1 - r3 * ndy);
    float cn2 = dot(c, c);
    if (!(cn2 > 0.0)) return false;
    float icn = 1.0 / cn2;
    float al1 = (x1 - ndx * w1) * iw, al2 = (x2 - ndx * w2) * iw;
    float be1 = (y1 - ndy * w1) * iw, be2 = (y2 - ndy * w2) * iw;
    vec3 cxv = cross(r1, r3), cyv = cross(r3, r0);
    vec3 v1 = cxv * al1 + cyv * be1;
    vec3 v2 = cxv * al2 + cyv * be2;
    float g1 = dot(c, v1) * icn, g2 = dot(c, v2) * icn;
    float n1 = max(dot(v1, v1) * icn - g1 * g1, 0.0);
    float n2 = max(dot(v2, v2) * icn - g2 * g2, 0.0);
    float mu = (3.0 * k1 * k1 - 3.0 * k1 * g1 + 0.5 * n1) + (3.0 * k2 * k2 - 3.0 * k2 * g2 + 0.5 * n2);
    float h1 = g1 - 3.0 * k1, h2 = g2 - 3.0 * k2;
    float a1 = 0.5 * W * al1, a2 = 0.5 * W * al2;
    float b1 = 0.5 * H * be1, b2 = 0.5 * H * be2;
    float dx = a1 * h1 + a2 * h2;
    float dy = b1 * h1 + b2 * h2;
    float f1 = 20.0 * k1 * k1 - 10.0 * k1 * g1 + n1;
    float f2 = 20.0 * k2 * k2 - 10.0 * k2 * g2 + n2;
    float cxx = max(a1 * a1 + a2 * a2, 1e-6);
    float cyy = max(b1 * b1 + b2 * b2, 1e-6);
    float cxy = a1 * b1 + a2 * b2;
    float dcxx = a1 * a1 * f1 + a2 * a2 * f2 - mu * cxx - dx * dx;
    float dcyy = b1 * b1 * f1 + b2 * b2 * f2 - mu * cyy - dy * dy;
    float dcxy = a1 * b1 * f1 + a2 * b2 * f2 - mu * cxy - dx * dy;
    k.mu = mu; k.dx = dx; k.dy = dy;
    k.cxx = cxx; k.cyy = cyy; k.cxy = cxy;
    k.dcxx = dcxx; k.dcyy = dcyy; k.dcxy = dcxy;
    k.mx = (ndx * 0.5 + 0.5) * W;
    k.my = (ndy * 0.5 + 0.5) * H;
    return true;
}

// The predicted change of a predicted slice's in-frame light (sigma points against one projection)
// as a fraction of its mass, from its terms k (dust::volume_small_demote_err: the second-order mass
// term mu, mean shift delta and covariance change of the 4 sigma points' J3-weighted moments, bounded
// edge by edge on the frame's rectangle integral, x DUST_SMALL_DEMOTE_SAFETY)
float dust_small_demote_err_of(DustSmallTerms k, float W, float H) {
    float mu = k.mu, dx = k.dx, dy = k.dy, cxx = k.cxx, cyy = k.cyy, cxy = k.cxy;
    float dcxx = k.dcxx, dcyy = k.dcyy, dcxy = k.dcxy, mx = k.mx, my = k.my;
    float lx = dcxx / (2.0 * cxx);
    float ly = dcyy / (2.0 * cyy);
    float sx = sqrt(cxx);
    float sy = sqrt(cyy);
    float isc = 1.0 / sqrt(cxx * cyy);
    float rho = cxy * isc;
    float drho = dcxy * isc - rho * (lx + ly);
    float tl = mx / sx, tr = (W - mx) / sx;
    float tb = my / sy, tt = (H - my) / sy;
    float s = DUST_SMALL_DEMOTE_SAFETY;
    float ux = s * abs(dx) / sx, uy = s * abs(dy) / sy;
    float alx = s * abs(lx), aly = s * abs(ly);
    float adr = s * abs(drho), amu = s * abs(mu);
    float dPl, dGl, eal, pul, dPr, dGr, ear, pur, dPb, dGb, eab, pub, dPt, dGt, eat, put;
    dust_small_demote_edge(tl, ux, alx, dPl, dGl, eal, pul);
    dust_small_demote_edge(tr, ux, alx, dPr, dGr, ear, pur);
    dust_small_demote_edge(tb, uy, aly, dPb, dGb, eab, pub);
    dust_small_demote_edge(tt, uy, aly, dPt, dGt, eat, put);
    float dpx = dPl + dPr, dpy = dPb + dPt;
    float dgx = dGl + dGr, dgy = dGb + dGt;
    float gx = max(eal, ear), gy = max(eab, eat);
    float pxu = min(min(1.0, pul), pur);
    float pyu = min(min(1.0, pub), put);
    float arho = min(abs(rho), DUST_SMALL_RHO_MAX);
    return amu * (pxu * pyu + arho * DUST_INV_2PI * gx * gy)
         + (1.0 + amu) * (dpx * (pyu + dpy) + pxu * dpy
                          + (arho + adr) * DUST_INV_2PI * (dgx * (gy + dgy) + gx * dgy)
                          + adr * DUST_INV_2PI * gx * gy);
}

// a small piece's empty point (m = 0: none)
DustSmallPoint dust_small_none() {
    DustSmallPoint o;
    o.mx = 0.0; o.my = 0.0; o.ix = 0.0; o.iy = 0.0; o.rho = 0.0; o.m = 0.0;
    return o;
}

// Track 9A (t9a2): a demoted slice's one projection of mass mq (> 0) with its predicted second-order
// terms k applied (dust::volume_small_demote_fix): mass x (1 + mu), mean + delta, covariance + its
// change (the variances floored at 1e-6 px^2, rho clamped as dust_small_finish): the sigma points'
// moments to second order. k.mx, k.my are the one projection's mean, bit for bit the fast point's
DustSmallPoint dust_small_demote_fix_of(DustSmallTerms k, float mq) {
    float cxx = max(k.cxx + k.dcxx, 1e-6);
    float cyy = max(k.cyy + k.dcyy, 1e-6);
    float rhoRaw = (k.cxy + k.dcxy) * inversesqrt(cxx * cyy);
    float rho = (isnan(rhoRaw) || isinf(rhoRaw)) ? 0.0 : clamp(rhoRaw, -DUST_SMALL_RHO_MAX, DUST_SMALL_RHO_MAX);
    float m = mq * (1.0 + k.mu);
    if (!(m > 0.0) || isinf(m)) return dust_small_none();
    DustSmallPoint o;
    o.mx = k.mx + k.dx;
    o.my = k.my + k.dy;
    o.ix = inversesqrt(cxx) * DUST_SMALL_INV_SQRT2;
    o.iy = inversesqrt(cyy) * DUST_SMALL_INV_SQRT2;
    o.rho = rho;
    o.m = m;
    return o;
}

// Track 9A (t9a2b): slice q's prediction and its corrected one projection from one evaluation of its
// terms (dust::volume_small_demote_err and dust::volume_small_demote_fix on the same terms, which
// die here: nothing of the slice but p and the returned change outlives the call). Returns the
// predicted change as a fraction of the slice's mass (-1 when not predicted, p then none); p the
// slice's one projection of mass mq with its terms applied
float dust_small_demote_slice(uint q, bool five, vec3 mean, vec3 g, float x1, float x2, float y1, float y2,
                              float w1, float w2, vec3 r0, vec3 r1, vec3 r3, vec3 t, float W, float H, float mq,
                              out DustSmallPoint p) {
    DustSmallTerms k;
    p = dust_small_none();
    if (!dust_small_demote_terms(q, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, k)) return -1.0;
    p = dust_small_demote_fix_of(k, mq);
    return dust_small_demote_err_of(k, W, H);
}

// Track 9A (t9a2d): an edge small piece's inputs as the consumer computed them, the record the edge
// pass (dust_draw_small_edge.comp) draws it from (dust_small_edge_put / _get): its centre, its depth
// axis g (3 sigma bounded), the lateral axes' six dots against the projection's rows, its mass and
// its slices' radiance weights
struct DustSmallEdge {
    vec3 mean;
    vec3 g;
    float x1, x2, y1, y2, w1, w2;
    float massW;
    float wr0, wr1, wr2;
};

// The three depth points of a small volumetric piece (dust::volume_small_points), as three
// values (no array). Live across the slices: mean, g, the lateral axes' six dots, the mass and the
// slices' weights. Round 3: the weights and the fast projections of every slice as one
// straight-line block (independent chains the scheduler overlaps), then the edge decision on s0
// as before. Track 9A (t9a2d): the consumer never draws an edge piece: it returns true with the
// piece's inputs in ed (s0..s4 then unused), and the consumer defers it to the edge pass
// (dust_small_edge_put), which draws it by dust_volume_small_edge_points (the demotion or the
// sigma points); so neither the predictor nor the sigma points are code of the consumer.
// Profile parts (DUST_PROF_K1C0): vsmprep the prelude, vsmfast the weights, the fast slices and
// the decision (every piece); an interior piece counted by its path here, an edge piece by the pass
bool dust_volume_small_points(DustVolumePiece pc_, vec3 eye, vec3 r0, vec3 r1, vec3 r3, vec3 t, float W, float H,
                              vec3 meanRef, float renorm, out DustSmallPoint s0, out DustSmallPoint s1,
                              out DustSmallPoint s2, out DustSmallPoint s3, out DustSmallPoint s4,
                              out DustSmallEdge ed) {
    s0 = dust_small_none(); s1 = dust_small_none(); s2 = dust_small_none(); s3 = dust_small_none(); s4 = dust_small_none();
    DUST_PROF_BEGIN(vsmprep)
    vec3 lq = dust_l_mul(pc_.l, pc_.qw);
    vec3 mean = eye - lq;
    float dn = sqrt(dot(lq, lq));
    if (!(dn > 0.0)) { DUST_PROF_END(vsmprep) return false; }
    vec3 e = lq * (-1.0 / dn);
    vec3 lte = dust_lt_mul(pc_.l, e);
    float se = sqrt(dot(lte, lte));
    if (!(se > 0.0)) { DUST_PROF_END(vsmprep) return false; }
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
    // (t9a2d: the edge piece's record; dead in the caller unless edge)
    ed.mean = mean;
    ed.g = g;
    ed.x1 = x1; ed.x2 = x2; ed.y1 = y1; ed.y2 = y2; ed.w1 = w1; ed.w2 = w2;
    ed.massW = massW;
    ed.wr0 = wr0; ed.wr1 = wr1; ed.wr2 = wr2;
    if (!edge) {
        DUST_PROF_COUNT(small_fast, 1u)
    }
    return edge;
}

// Track 9A (t9a2d): an edge small piece's depth points in the edge pass (dust_draw_small_edge.comp;
// dust::volume_small_points' edge branch), from the inputs the consumer recorded: the one projection
// of every slice again (dust_volume_small_fast on the same inputs: the consumer's points), then the
// demotion (the slices' predictions in order against DUST_SMALL_DEMOTE_EPS, each demoted slice
// corrected in place: t9a2b's sequential prefix) or the 4 lateral sigma points a slice. The 5-point
// depth rule is off in production (dust::DUST_K1C_D_GH5), as in the consumer. Profile parts: vsmfast
// the one projections, vsmedge the demotion and the sigma points; the piece counted by its path.
// Returns whether the piece was demoted (t9a2e: the pass counts them, DUST_VEDGE_DEMOTED)
bool dust_volume_small_edge_points(DustSmallEdge ed, vec3 r0, vec3 r1, vec3 r3, vec3 t, float W, float H,
                                   out DustSmallPoint s0, out DustSmallPoint s1, out DustSmallPoint s2,
                                   out DustSmallPoint s3, out DustSmallPoint s4) {
    bool five = false;
    vec3 mean = ed.mean, g = ed.g;
    float x1 = ed.x1, x2 = ed.x2, y1 = ed.y1, y2 = ed.y2, w1 = ed.w1, w2 = ed.w2;
    float massW = ed.massW;
    float wr0 = ed.wr0, wr1 = ed.wr1, wr2 = ed.wr2, wr3 = 1.0, wr4 = 1.0;
    s3 = dust_small_none(); s4 = dust_small_none();
    DUST_PROF_BEGIN(vsmfast)
    s0 = dust_volume_small_fast(0u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr0, r0, r1, r3, t, W, H);
    s1 = dust_volume_small_fast(1u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr1, r0, r1, r3, t, W, H);
    s2 = dust_volume_small_fast(2u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr2, r0, r1, r3, t, W, H);
    DUST_PROF_END(vsmfast)
    DUST_PROF_BEGIN(vsmedge)
    // track 9A: the demotion (dust::volume_small_points: the gate (t9a2c, on the fast centre slice:
    // a few operations, so most edge pieces that keep the sigma points never reach the predictor),
    // then the slices' predictions in order, the mass-weighted mean against DUST_SMALL_DEMOTE_EPS,
    // strict); vsmedge holds its cost.
    // t9a2b: the fast points' masses are kept and the fast points die; each slice is predicted
    // and corrected from one evaluation of its terms (dust_small_demote_slice, its corrected
    // point written in place: an edge piece that is not demoted overwrites every slice below),
    // and the next slice is evaluated only while the running sum acc / ms is under the constant.
    // The sum only grows (m > 0, e >= 0, each rounding monotone), so a prefix at or over the
    // constant means the full sum is too: the decision is the full rule's exactly, the slices
    // are sequential (no two slices' terms live at once) and most edge pieces that keep the
    // sigma points stop after the centre slice
    bool demote = false;
    if (DUST_SMALL_DEMOTE_EPS > 0.0 && s0.m > 0.0 && s1.m > 0.0 && s2.m > 0.0
        && (!five || (s3.m > 0.0 && s4.m > 0.0))
        && dust_small_demote_gate(s0, mean, r3, t.z, w1, w2, W, H) < DUST_SMALL_DEMOTE_GATE * DUST_SMALL_DEMOTE_EPS) {
        float m0 = s0.m, m1 = s1.m, m2 = s2.m, m3 = s3.m, m4 = s4.m;
        float ms = m0 + m1 + m2;
        if (five) {
            ms += m3;
            ms += m4;
        }
        float e = dust_small_demote_slice(0u, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, m0, s0);
        float acc = m0 * e;
        bool go = e >= 0.0 && acc / ms < DUST_SMALL_DEMOTE_EPS;
        if (go) {
            e = dust_small_demote_slice(1u, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, m1, s1);
            acc += m1 * e;
            go = e >= 0.0 && acc / ms < DUST_SMALL_DEMOTE_EPS;
        }
        if (go) {
            e = dust_small_demote_slice(2u, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, m2, s2);
            acc += m2 * e;
            go = e >= 0.0 && acc / ms < DUST_SMALL_DEMOTE_EPS;
        }
        if (five && go) {
            e = dust_small_demote_slice(3u, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, m3, s3);
            acc += m3 * e;
            go = e >= 0.0 && acc / ms < DUST_SMALL_DEMOTE_EPS;
        }
        if (five && go) {
            e = dust_small_demote_slice(4u, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, m4, s4);
            acc += m4 * e;
            go = e >= 0.0 && acc / ms < DUST_SMALL_DEMOTE_EPS;
        }
        demote = go;
    }
    if (!demote) {
        s0 = dust_volume_small_edge(0u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr0, r0, r1, r3, t, W, H);
        s1 = dust_volume_small_edge(1u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr1, r0, r1, r3, t, W, H);
        s2 = dust_volume_small_edge(2u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr2, r0, r1, r3, t, W, H);
        if (five) {
            s3 = dust_volume_small_edge(3u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr3, r0, r1, r3, t, W, H);
            s4 = dust_volume_small_edge(4u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr4, r0, r1, r3, t, W, H);
        }
    }
    DUST_PROF_END(vsmedge)
    if (demote) {
        DUST_PROF_COUNT(small_fast, 1u)
    } else {
        DUST_PROF_COUNT(small_edge, 1u)
    }
    // t9a2e: whether the piece was demoted (the edge pass counts them, DUST_VEDGE_DEMOTED)
    return demote;
}

#ifdef DUST_EDGE_PASS
// Round 9 step Z, the edge pass: the small segment in two passes over its texels, the points'
// rectangle integrals (tau per px^2) into the lane's row of dust_e_tau, then the deposits
// (dust_volume_segment_small's texel arithmetic in its order; the frame test of a texel made
// again in the second pass, the same expressions)
float dust_small_rect(DustSmallPoint q, float xa, float xb, float ya, float yb);
bool dust_volume_deposit(DustUintBuffer pyr, float tauPx, float area, float invUnit, vec3 color, uint depthBits,
                         uint off, uint lw, int ty, int tx, bool noWrites, inout float carry, inout float peak);
shared float dust_e_pts[64u * 18u];
void dust_small_pt_put(uint lane, uint q, DustSmallPoint p) {
    uint a = lane * 18u + q * 6u;
    dust_e_pts[a] = p.mx; dust_e_pts[a + 1u] = p.my; dust_e_pts[a + 2u] = p.ix;
    dust_e_pts[a + 3u] = p.iy; dust_e_pts[a + 4u] = p.rho; dust_e_pts[a + 5u] = p.m;
}
DustSmallPoint dust_small_pt_get(uint lane, uint q) {
    uint a = lane * 18u + q * 6u;
    DustSmallPoint p;
    p.mx = dust_e_pts[a]; p.my = dust_e_pts[a + 1u]; p.ix = dust_e_pts[a + 2u];
    p.iy = dust_e_pts[a + 3u]; p.rho = dust_e_pts[a + 4u]; p.m = dust_e_pts[a + 5u];
    return p;
}
shared float dust_e_tau[64u * DUST_SEGMENT_TEXELS];
bool dust_volume_segment_small_wg(DustUintBuffer pyr, float invUnit, float W, float H, float sz, vec3 color,
                                  uint depthBits, uint off, uint lw, uint seed, int ty, int xa, int xb,
                                  DustSmallPoint s0, DustSmallPoint s1, DustSmallPoint s2, DustSmallPoint s3,
                                  DustSmallPoint s4, inout float peak, uint lane) {
    bool noWrites = ((dust_pyr_split_word >> 30u) & 3u) == 2u;
    float area = sz * sz;
    float ya = max(float(ty) * sz, 0.0);
    float yb = min(float(ty + 1) * sz, H);
    uint row = lane * DUST_SEGMENT_TEXELS;
    DUST_PROF_BEGIN(vsmall)
    for (int tx = xa; tx <= xb; ++tx) {
        float xa_ = max(float(tx) * sz, 0.0);
        float xb_ = min(float(tx + 1) * sz, W);
        if (!(xb_ > xa_ && yb > ya)) continue;
        float acc = dust_small_rect(dust_small_pt_get(lane, 0u), xa_, xb_, ya, yb) + dust_small_rect(dust_small_pt_get(lane, 1u), xa_, xb_, ya, yb)
                  + dust_small_rect(dust_small_pt_get(lane, 2u), xa_, xb_, ya, yb) + dust_small_rect(s3, xa_, xb_, ya, yb)
                  + dust_small_rect(s4, xa_, xb_, ya, yb);
        dust_e_tau[row + uint(tx - xa)] = max(acc, 0.0) / ((xb_ - xa_) * (yb - ya));
    }
    DUST_PROF_END(vsmall)
    float carry = dust_segment_carry(seed, (uint(ty) << 16u) | uint(xa));
    bool any_ = false;
    DUST_PROF_BEGIN(vdeposit)
    for (int tx = xa; tx <= xb; ++tx) {
        float xa_ = max(float(tx) * sz, 0.0);
        float xb_ = min(float(tx + 1) * sz, W);
        if (!(xb_ > xa_ && yb > ya)) continue;
        float tauPx = dust_e_tau[row + uint(tx - xa)];
        any_ = dust_volume_deposit(pyr, tauPx, area, invUnit, color, depthBits, off, lw, ty, tx, noWrites, carry, peak)
            || any_;
    }
    DUST_PROF_END(vdeposit)
    return any_;
}
// the edge pass (dust_draw_small_edge.comp): the same slices, each slice's final
// point stored to the lane's slot of dust_e_pts as it is made (dust_small_pt_put) instead of held
// across the next slices' predictions (Mali: the pass's spill)
void dust_small_edge_get_ed(DustUintBuffer pyr, uint ebase, uint cap, uint i, out DustSmallEdge ed);
#define DUST_EDGE_LOAD(ed) dust_small_edge_get_ed(pyr, ebase, cap, i, ed); \
    W = float(pyr.v[DUST_PYR_WIDTH]); H = float(pyr.v[DUST_PYR_HEIGHT]); \
    mean = ed.mean; g = ed.g; x1 = ed.x1; x2 = ed.x2; y1 = ed.y1; y2 = ed.y2; w1 = ed.w1; w2 = ed.w2; \
    massW = ed.massW; wr0 = ed.wr0; wr1 = ed.wr1; wr2 = ed.wr2;
bool dust_volume_small_edge_points_st(DustUintBuffer pyr, uint ebase, uint cap, uint i, vec3 r0, vec3 r1, vec3 r3, vec3 t, float W, float H, uint lane) {
    DustSmallPoint s0, s1, s2, s3, s4;
    bool five = false;
    DustSmallEdge ed;
    vec3 mean, g;
    float x1, x2, y1, y2, w1, w2, massW, wr0, wr1, wr2, wr3 = 1.0, wr4 = 1.0;
    DUST_EDGE_LOAD(ed)
    s3 = dust_small_none(); s4 = dust_small_none();
    DUST_PROF_BEGIN(vsmfast)
    s0 = dust_volume_small_fast(0u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr0, r0, r1, r3, t, W, H);
    s1 = dust_volume_small_fast(1u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr1, r0, r1, r3, t, W, H);
    s2 = dust_volume_small_fast(2u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr2, r0, r1, r3, t, W, H);
    DUST_PROF_END(vsmfast)
    DUST_PROF_BEGIN(vsmedge)
    // track 9A: the demotion (dust::volume_small_points: the gate (t9a2c, on the fast centre slice:
    // a few operations, so most edge pieces that keep the sigma points never reach the predictor),
    // then the slices' predictions in order, the mass-weighted mean against DUST_SMALL_DEMOTE_EPS,
    // strict); vsmedge holds its cost.
    // t9a2b: the fast points' masses are kept and the fast points die; each slice is predicted
    // and corrected from one evaluation of its terms (dust_small_demote_slice, its corrected
    // point written in place: an edge piece that is not demoted overwrites every slice below),
    // and the next slice is evaluated only while the running sum acc / ms is under the constant.
    // The sum only grows (m > 0, e >= 0, each rounding monotone), so a prefix at or over the
    // constant means the full sum is too: the decision is the full rule's exactly, the slices
    // are sequential (no two slices' terms live at once) and most edge pieces that keep the
    // sigma points stop after the centre slice
    bool demote = false;
    if (DUST_SMALL_DEMOTE_EPS > 0.0 && s0.m > 0.0 && s1.m > 0.0 && s2.m > 0.0
        && (!five || (s3.m > 0.0 && s4.m > 0.0))
        && dust_small_demote_gate(s0, mean, r3, t.z, w1, w2, W, H) < DUST_SMALL_DEMOTE_GATE * DUST_SMALL_DEMOTE_EPS) {
        float m0 = s0.m, m1 = s1.m, m2 = s2.m, m3 = s3.m, m4 = s4.m;
        float ms = m0 + m1 + m2;
        if (five) {
            ms += m3;
            ms += m4;
        }
        DUST_EDGE_LOAD(ed)
        float e = dust_small_demote_slice(0u, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, m0, s0);
        dust_small_pt_put(lane, 0u, s0);
        float acc = m0 * e;
        bool go = e >= 0.0 && acc / ms < DUST_SMALL_DEMOTE_EPS;
        if (go) {
            DUST_EDGE_LOAD(ed)
            e = dust_small_demote_slice(1u, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, m1, s1);
            dust_small_pt_put(lane, 1u, s1);
            acc += m1 * e;
            go = e >= 0.0 && acc / ms < DUST_SMALL_DEMOTE_EPS;
        }
        if (go) {
            DUST_EDGE_LOAD(ed)
            e = dust_small_demote_slice(2u, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, m2, s2);
            dust_small_pt_put(lane, 2u, s2);
            acc += m2 * e;
            go = e >= 0.0 && acc / ms < DUST_SMALL_DEMOTE_EPS;
        }
        if (five && go) {
            e = dust_small_demote_slice(3u, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, m3, s3);
            acc += m3 * e;
            go = e >= 0.0 && acc / ms < DUST_SMALL_DEMOTE_EPS;
        }
        if (five && go) {
            e = dust_small_demote_slice(4u, five, mean, g, x1, x2, y1, y2, w1, w2, r0, r1, r3, t, W, H, m4, s4);
            acc += m4 * e;
            go = e >= 0.0 && acc / ms < DUST_SMALL_DEMOTE_EPS;
        }
        demote = go;
    }
    if (!demote) {
        DUST_EDGE_LOAD(ed)
        s0 = dust_volume_small_edge(0u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr0, r0, r1, r3, t, W, H);
        dust_small_pt_put(lane, 0u, s0);
        s1 = dust_volume_small_edge(1u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr1, r0, r1, r3, t, W, H);
        dust_small_pt_put(lane, 1u, s1);
        s2 = dust_volume_small_edge(2u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr2, r0, r1, r3, t, W, H);
        dust_small_pt_put(lane, 2u, s2);
        if (five) {
            s3 = dust_volume_small_edge(3u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr3, r0, r1, r3, t, W, H);
            s4 = dust_volume_small_edge(4u, five, mean, g, x1, x2, y1, y2, w1, w2, massW, wr4, r0, r1, r3, t, W, H);
        }
    }
    DUST_PROF_END(vsmedge)
    if (demote) {
        DUST_PROF_COUNT(small_fast, 1u)
    } else {
        DUST_PROF_COUNT(small_edge, 1u)
    }
    // t9a2e: whether the piece was demoted (the edge pass counts them, DUST_VEDGE_DEMOTED)
    return demote;
}
#endif

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
// (round 8 step F2) the accumulate mode of the volume consumer's top-level path: the frame's bias
// puts every piece on the 1 x 1 top texel (dust_volume_draw topKnown, frame-uniform, set by
// dust_draw_volume.comp main with the texel's word dust_vacc_off), so each lane sums its deposits
// here and main writes the subgroup's sums with one atomic per word (dust_volume_flush_top).
// Exact: saturating adds of non-negatives are associative (the word is 0xFFFFFFFF iff the total
// passed 2^32: a lane's wrap bit or the subgroup's carry), the depth a min. A deposit at another
// address (never, by the path's construction) takes the per-deposit atomics as before.
bool  dust_vacc = false;
uint  dust_vacc_off = 0u;
uvec4 dust_vacc_sum = uvec4(0u);   // counts, r, g, b, mod 2^32
uint  dust_vacc_wrap = 0u;         // bit w: the lane's word w passed 2^32
uint  dust_vacc_depth = 0u;        // the lane's nearest depth (0: none written)
uint  dust_vacc_splits = 0u;       // the lane's deposits split into several adds
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
    bool acc = dust_vacc && i == dust_vacc_off;
    if (counts > DUST_COUNT_MAX_PER_ADD) {
        if (acc) dust_vacc_splits++;
        else atomicAdd(pyr.v[dust_vqueue_base(pyr) + DUST_VQUEUE_SPLIT_ADDS], 1u);
    }
    for (float rest = counts; rest > 0.5; ) {
        uint c = uint(min(rest, DUST_COUNT_MAX_PER_ADD));
        uvec3 cw = uvec3(float(c) * color * DUST_COLOR_SCALE + hf);
        if (acc) {
            uvec4 a = uvec4(c, cw), s = dust_vacc_sum + a;
            dust_vacc_wrap |= (s.x < a.x ? 1u : 0u) | (s.y < a.y ? 2u : 0u) | (s.z < a.z ? 4u : 0u) | (s.w < a.w ? 8u : 0u);
            dust_vacc_sum = s;
        } else {
            dust_atomic_add_sat(pyr, i, c);
            dust_atomic_add_sat(pyr, i + 1u, cw.r);
            dust_atomic_add_sat(pyr, i + 2u, cw.g);
            dust_atomic_add_sat(pyr, i + 3u, cw.b);
        }
        rest -= float(c);
    }
    if (acc) {
        dust_vacc_depth = dust_vacc_depth == 0u ? depthBits : min(dust_vacc_depth, depthBits);
        return true;
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
        DUST_PROF_BEGIN(vdeposit)
        any_ = dust_volume_deposit(pyr, tauPx, area, invUnit, color, depthBits, off, lw, ty, tx, noWrites, carry, peak)
            || any_;
        DUST_PROF_END(vdeposit)
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
        DUST_PROF_BEGIN(vdeposit)
        any_ = dust_volume_deposit(pyr, tauPx, area, invUnit, color, depthBits, off, lw, ty, tx, noWrites, carry, peak)
            || any_;
        DUST_PROF_END(vdeposit)
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
        DUST_PROF_BEGIN(vdeposit)
        any_ = dust_volume_deposit(pyr, tauPx, area, invUnit, color, depthBits, off, lw, ty, tx, noWrites, carry, peak)
            || any_;
        DUST_PROF_END(vdeposit)
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
const uint DUST_QUEUE_HEADER_WORDS = 144u;
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
uint dust_queue_base_o(uvec2 pyr) {
    return DustUintBuffer(pyr).v[DUST_PYR_QUEUE_BASE];
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
// PYR_FLAGS bit (a test hook through Device::dust_set_vqueue_mode, dust::DUST_FRAME_TOP_PASS_A):
// the volume consumer runs pass A in full even when every piece is known to land on the 1 x 1
// last level (the top-level path's word-identical reference)
const uint DUST_FRAME_TOP_PASS_A = 1u << 2u;

uint dust_vqueue_base(DustUintBuffer pyr) { return dust_queue_base(pyr) + DUST_QUEUE_WORDS; }
uint dust_vqueue_base_o(uvec2 pyr) { return dust_queue_base_o(pyr) + DUST_QUEUE_WORDS; }
uint dust_vqueue_capacity(DustUintBuffer pyr, uint base) {
    uint c = pyr.v[base + 4u];
    return c == 0u ? DUST_VQUEUE_CAPACITY : min(c, DUST_VQUEUE_CAPACITY);
}
uint dust_vqueue_capacity_o(uvec2 pyr, uint base) {
    uint c = DustUintBuffer(pyr).v[base + 4u];
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
uint dust_vqueue_append_o(uvec2 pyr, uint base) {
    uint r = atomicAdd(DustUintBuffer(pyr).v[base + 3u], 1u);
    if (r >= dust_vqueue_capacity_o(pyr, base)) return 0xFFFFFFFFu;
    atomicMax(DustUintBuffer(pyr).v[base], (r >> 6u) + 1u);
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
uint dust_vspill_base_o(uvec2 pyr) {
    uint vbase = dust_vqueue_base_o(pyr);
    return vbase + DustUintBuffer(pyr).v[vbase + DUST_VQUEUE_SPILL_OFFSET];
}
// A volume record, structure-of-arrays: word w of record r at base + DUST_VQUEUE_HEADER_WORDS +
// w * capacity + r, so a warp's 32 consecutive records load and store each word from 128
// contiguous bytes (one transaction a word; the former 128-B stride made 32). Words: 0-2 mean,
// 3-8 S, 9-11 seg, 12 flux, 13-14 rho, 15-17 segT, 18 seed, 19 exposure, 20-22 colour, 23-26
// the support plane, 27-29 the Sun direction, 30 r_h (AU), 31 open ends (bit 0 start, bit 1
// end). Macros: a struct argument would be copied per call.
#define DUST_VQUEUE_WORD(base, cap, r, w) ((base) + DUST_VQUEUE_HEADER_WORDS + (w) * (cap) + (r))

// ── Track 9A (t9a2d): the deferred edge pieces (dust::DUST_VEDGE_*) ───────────────────────────────
// Two regions after the volume spill's records (dust::dust_queue_offsets' vedge), each a header in
// the volume queue's format (words 0-2 the edge pass's indirect dispatch, 3 the raw append counter,
// 4 the capacity: dust_vqueue_append and dust_vqueue_capacity serve it; 5 the frame's pieces drawn,
// 6 those refused) and dust::dust_vedge_records records of DUST_VEDGE_RECORD_WORDS words,
// array-of-structures (word w of record i at the records' base + i * DUST_VEDGE_RECORD_WORDS + w:
// the edge pass reads a piece's words together; perf9_O). A chunk's consumer (both drains) appends its edge
// small pieces to the region of the chunk's parity (DUST_VIEW_EDGE_ODD), and dust_draw_small_edge.comp
// draws them beside the next chunk's producer (Device::cmd_dust_splat), so the pass costs no extra
// serialisation; the deposits are the consumer's order-free atomics, so the pyramid is the same
// wherever a piece is drawn. A piece the region refuses is dropped and counted (word 6, the pass's
// raw - held; logged by the host); word 7 counts the pieces the pass demoted (t9a2e: one projection
// a slice with the predicted terms, DUST_VEDGE_DEMOTED). Words: 0-2 mean, 3-5 g, 6-11 x1 x2 y1 y2 w1 w2, 12 mass, 13-15
// the slices' weights, 16 the range's x0 | x1 << 16, 17 its y0 | y1 << 16 (texels of its level,
// each < 2^16), 18 the level, 19-21 colour, 22 the depth bits, 23 the piece's rounding seed.
const uint DUST_VEDGE_RECORD_WORDS = 24u;
// the edge pass's workgroup takes 1 << DUST_VEDGE_GROUP_SHIFT pieces (dust_draw_small_edge.comp)
const uint DUST_VEDGE_GROUP_SHIFT = 4u;
const uint DUST_VEDGE_DRAWN = 5u;
const uint DUST_VEDGE_LOST = 6u;
const uint DUST_VEDGE_DEMOTED = 7u;
// a region's records for a view of n volume records (dust::dust_vedge_records), and the volume
// spill's (dust::dust_vspill_records: the regions' place)
uint dust_vedge_records_of(uint n) { return max(n / 2u, 64u); }
uint dust_vspill_records_of(uint n) { return max(n / 4u, 64u); }
uint dust_vedge_base(DustUintBuffer pyr, bool odd) {
    uint vbase = dust_vqueue_base(pyr);
    uint n = pyr.v[vbase + DUST_VQUEUE_RECORDS];
    n = n == 0u ? DUST_VQUEUE_CAPACITY : min(n, DUST_VQUEUE_CAPACITY);
    uint e0 = dust_vspill_base(pyr) + DUST_VQUEUE_HEADER_WORDS + dust_vspill_records_of(n) * DUST_VQUEUE_RECORD_WORDS;
    return odd ? e0 + DUST_VQUEUE_HEADER_WORDS + dust_vedge_records_of(n) * DUST_VEDGE_RECORD_WORDS : e0;
}
// the consumer's append of an edge piece: its inputs, its range r at level lev, its colour, depth
// bits and rounding seed (refused when the region is full: dropped, counted by the pass)
void dust_small_edge_put(DustUintBuffer pyr, bool odd, DustSmallEdge ed, ivec4 r, uint lev, vec3 color,
                         uint depthBits, uint seed) {
    uint ebase = dust_vedge_base(pyr, odd);
    // dust_vqueue_append's, the dispatch in the edge pass's workgroups (DUST_VEDGE_GROUP_SHIFT)
    uint i = atomicAdd(pyr.v[ebase + 3u], 1u);
    if (i >= dust_vqueue_capacity(pyr, ebase)) return;
    atomicMax(pyr.v[ebase], (i >> DUST_VEDGE_GROUP_SHIFT) + 1u);
    uint a = ebase + DUST_VQUEUE_HEADER_WORDS + i * DUST_VEDGE_RECORD_WORDS;
    pyr.v[a + 0u] = floatBitsToUint(ed.mean.x);
    pyr.v[a + 1u] = floatBitsToUint(ed.mean.y);
    pyr.v[a + 2u] = floatBitsToUint(ed.mean.z);
    pyr.v[a + 3u] = floatBitsToUint(ed.g.x);
    pyr.v[a + 4u] = floatBitsToUint(ed.g.y);
    pyr.v[a + 5u] = floatBitsToUint(ed.g.z);
    pyr.v[a + 6u] = floatBitsToUint(ed.x1);
    pyr.v[a + 7u] = floatBitsToUint(ed.x2);
    pyr.v[a + 8u] = floatBitsToUint(ed.y1);
    pyr.v[a + 9u] = floatBitsToUint(ed.y2);
    pyr.v[a + 10u] = floatBitsToUint(ed.w1);
    pyr.v[a + 11u] = floatBitsToUint(ed.w2);
    pyr.v[a + 12u] = floatBitsToUint(ed.massW);
    pyr.v[a + 13u] = floatBitsToUint(ed.wr0);
    pyr.v[a + 14u] = floatBitsToUint(ed.wr1);
    pyr.v[a + 15u] = floatBitsToUint(ed.wr2);
    pyr.v[a + 16u] = uint(r.x) | (uint(r.y) << 16u);
    pyr.v[a + 17u] = uint(r.z) | (uint(r.w) << 16u);
    pyr.v[a + 18u] = lev;
    pyr.v[a + 19u] = floatBitsToUint(color.x);
    pyr.v[a + 20u] = floatBitsToUint(color.y);
    pyr.v[a + 21u] = floatBitsToUint(color.z);
    pyr.v[a + 22u] = depthBits;
    pyr.v[a + 23u] = seed;
}
// the edge pass's read of record i of the region at ebase (capacity cap): the inputs into ed, the
// range, level, colour, depth bits and seed
void dust_small_edge_get_ed(DustUintBuffer pyr, uint ebase, uint cap, uint i, out DustSmallEdge ed) {
    uint a = ebase + DUST_VQUEUE_HEADER_WORDS + i * DUST_VEDGE_RECORD_WORDS;
    ed.mean = vec3(uintBitsToFloat(pyr.v[a + 0u]), uintBitsToFloat(pyr.v[a + 1u]),
                   uintBitsToFloat(pyr.v[a + 2u]));
    ed.g = vec3(uintBitsToFloat(pyr.v[a + 3u]), uintBitsToFloat(pyr.v[a + 4u]),
                uintBitsToFloat(pyr.v[a + 5u]));
    ed.x1 = uintBitsToFloat(pyr.v[a + 6u]);
    ed.x2 = uintBitsToFloat(pyr.v[a + 7u]);
    ed.y1 = uintBitsToFloat(pyr.v[a + 8u]);
    ed.y2 = uintBitsToFloat(pyr.v[a + 9u]);
    ed.w1 = uintBitsToFloat(pyr.v[a + 10u]);
    ed.w2 = uintBitsToFloat(pyr.v[a + 11u]);
    ed.massW = uintBitsToFloat(pyr.v[a + 12u]);
    ed.wr0 = uintBitsToFloat(pyr.v[a + 13u]);
    ed.wr1 = uintBitsToFloat(pyr.v[a + 14u]);
    ed.wr2 = uintBitsToFloat(pyr.v[a + 15u]);
}
void dust_small_edge_get(DustUintBuffer pyr, uint ebase, uint cap, uint i, out DustSmallEdge ed, out ivec4 r,
                         out uint lev, out vec3 color, out uint depthBits, out uint seed) {
    uint a = ebase + DUST_VQUEUE_HEADER_WORDS + i * DUST_VEDGE_RECORD_WORDS;
    ed.mean = vec3(uintBitsToFloat(pyr.v[a + 0u]), uintBitsToFloat(pyr.v[a + 1u]),
                   uintBitsToFloat(pyr.v[a + 2u]));
    ed.g = vec3(uintBitsToFloat(pyr.v[a + 3u]), uintBitsToFloat(pyr.v[a + 4u]),
                uintBitsToFloat(pyr.v[a + 5u]));
    ed.x1 = uintBitsToFloat(pyr.v[a + 6u]);
    ed.x2 = uintBitsToFloat(pyr.v[a + 7u]);
    ed.y1 = uintBitsToFloat(pyr.v[a + 8u]);
    ed.y2 = uintBitsToFloat(pyr.v[a + 9u]);
    ed.w1 = uintBitsToFloat(pyr.v[a + 10u]);
    ed.w2 = uintBitsToFloat(pyr.v[a + 11u]);
    ed.massW = uintBitsToFloat(pyr.v[a + 12u]);
    ed.wr0 = uintBitsToFloat(pyr.v[a + 13u]);
    ed.wr1 = uintBitsToFloat(pyr.v[a + 14u]);
    ed.wr2 = uintBitsToFloat(pyr.v[a + 15u]);
    uint rxy = pyr.v[a + 16u], rzw = pyr.v[a + 17u];
    r = ivec4(int(rxy & 0xFFFFu), int(rxy >> 16u), int(rzw & 0xFFFFu), int(rzw >> 16u));
    lev = pyr.v[a + 18u];
    color = vec3(uintBitsToFloat(pyr.v[a + 19u]), uintBitsToFloat(pyr.v[a + 20u]),
                 uintBitsToFloat(pyr.v[a + 21u]));
    depthBits = pyr.v[a + 22u];
    seed = pyr.v[a + 23u];
}

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
const uint DUST_SQUEUE_RECORD_WORDS = 68u;
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
uint dust_squeue_base_o(uvec2 pyr) {
    uint vbase = dust_vqueue_base_o(pyr);
    uint n = DustUintBuffer(pyr).v[vbase + DUST_VQUEUE_RECORDS];
    n = n == 0u ? DUST_VQUEUE_CAPACITY : min(n, DUST_VQUEUE_CAPACITY);
    return vbase + DUST_VQUEUE_HEADER_WORDS + n * DUST_VQUEUE_RECORD_WORDS;
}
uint dust_squeue_table(uint base) { return base + DUST_SQUEUE_HEADER_WORDS; }
// the speck walk's spill after the sub-packet queue (dust::DUST_SPILL_*): a header (word 0 the
// slots claimed this pass, word 1 the slots the view has), then the slots of DUST_SPILL_NODES
// nodes of DUST_SPILL_NODE_WORDS words each
// the splat's plan rows after the records (perf9_O: dust_splat.comp's stages, a row per chunk cluster)
const uint DUST_SPLAN_WORDS = 384u;
const uint DUST_SQUEUE_WORDS = DUST_SQUEUE_HEADER_WORDS + DUST_SQUEUE_CLUSTERS * DUST_SQUEUE_TABLE_WORDS + DUST_SQUEUE_CAPACITY * DUST_SQUEUE_RECORD_WORDS
                             + DUST_SQUEUE_CLUSTERS * DUST_SPLAN_WORDS;
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
uint dust_splan_base(uint base) { return dust_squeue_records(base) + DUST_SQUEUE_CAPACITY * DUST_SQUEUE_RECORD_WORDS; }
// A sub-packet record: 0-2 mean, 3-8 S, 9-11 the segment d, 12 flux, 13 exposure, 14 flags
// (bit 0 openStart, 1 openEnd, 2 the support has an origin), 15-16 rho, 17-19 segT, 20-22
// colour, 23-26 the support plane, 27 its origin, 28-30 the Sun direction, 31 r_h, 32 the
// cluster's index (its seed); 33-44 the cone words (below). Macros: a struct argument would be
// copied per call.
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
// the record's cone words (round 9 step Z; written by the splitter's prep stage, dust_split.comp
// -DDUST_SPLIT_PREP, dust_leaf_prep, after the enumerator): 33 the kind (DUST_SQUEUE_KIND_*: the
// packet drawn whole, its truncation's moments, its cone of grains, or whole by the volume path),
// 34-36 the support plane's covariance column S n, 37 v0 = n.S n, 38 the cone's apex depth dC,
// 39 its cut flag, 40 the kept shares' renormalisation, 41-44 the truncation bounds (za, zb,
// before, inner). The walk stage reads them with every piece it builds.
const uint DUST_SQUEUE_KIND_PLAIN  = 0u;
const uint DUST_SQUEUE_KIND_TRUNC  = 1u;
const uint DUST_SQUEUE_KIND_CONIC  = 2u;
const uint DUST_SQUEUE_KIND_VOLUME = 3u;
#define DUST_SQUEUE_PUT_CONE(pyr, recs, r, kind_, sn_, v0_, dC_, cut_, renorm_, tb_) { \
    uint a_ = (recs) + (r); const uint c_ = DUST_SQUEUE_CAPACITY; \
    pyr.v[a_ + 33u * c_] = (kind_); \
    pyr.v[a_ + 34u * c_] = floatBitsToUint((sn_).x); pyr.v[a_ + 35u * c_] = floatBitsToUint((sn_).y); pyr.v[a_ + 36u * c_] = floatBitsToUint((sn_).z); \
    pyr.v[a_ + 37u * c_] = floatBitsToUint(v0_); \
    pyr.v[a_ + 38u * c_] = floatBitsToUint(dC_); \
    pyr.v[a_ + 39u * c_] = (cut_) ? 1u : 0u; \
    pyr.v[a_ + 40u * c_] = floatBitsToUint(renorm_); \
    pyr.v[a_ + 41u * c_] = floatBitsToUint((tb_).za); pyr.v[a_ + 42u * c_] = floatBitsToUint((tb_).zb); \
    pyr.v[a_ + 43u * c_] = (tb_).before; pyr.v[a_ + 44u * c_] = (tb_).inner; \
}
#define DUST_SQUEUE_GET_CONE(pyr, recs, r, sn_, v0_, dC_, cut_, tb_) { \
    uint a_ = (recs) + (r); const uint c_ = DUST_SQUEUE_CAPACITY; \
    sn_ = vec3(uintBitsToFloat(pyr.v[a_ + 34u * c_]), uintBitsToFloat(pyr.v[a_ + 35u * c_]), uintBitsToFloat(pyr.v[a_ + 36u * c_])); \
    v0_ = uintBitsToFloat(pyr.v[a_ + 37u * c_]); \
    dC_ = uintBitsToFloat(pyr.v[a_ + 38u * c_]); \
    cut_ = pyr.v[a_ + 39u * c_] != 0u; \
    (tb_).za = uintBitsToFloat(pyr.v[a_ + 41u * c_]); (tb_).zb = uintBitsToFloat(pyr.v[a_ + 42u * c_]); \
    (tb_).before = pyr.v[a_ + 43u * c_]; (tb_).inner = pyr.v[a_ + 44u * c_]; \
}
uint dust_squeue_kind(DustUintBuffer pyr, uint recs, uint r) { return pyr.v[recs + r + 33u * DUST_SQUEUE_CAPACITY]; }
uint dust_squeue_kind_o(uvec2 pyr, uint recs, uint r) { return DustUintBuffer(pyr).v[recs + r + 33u * DUST_SQUEUE_CAPACITY]; }
float dust_squeue_renorm(DustUintBuffer pyr, uint recs, uint r) { return uintBitsToFloat(pyr.v[recs + r + 40u * DUST_SQUEUE_CAPACITY]); }
// the record's packet (its root: dust::Packet), support plane and origin, read where they are
// used (the walk stage re-reads the root at every piece load instead of keeping it live)
DustPacket dust_squeue_packet(DustUintBuffer pyr, uint recs, uint r) {
    uint a_ = recs + r; const uint c_ = DUST_SQUEUE_CAPACITY;
    DustPacket p;
    p.mean = vec3(uintBitsToFloat(pyr.v[a_ + 0u * c_]), uintBitsToFloat(pyr.v[a_ + 1u * c_]), uintBitsToFloat(pyr.v[a_ + 2u * c_]));
    p.S[0] = uintBitsToFloat(pyr.v[a_ + 3u * c_]); p.S[1] = uintBitsToFloat(pyr.v[a_ + 4u * c_]); p.S[2] = uintBitsToFloat(pyr.v[a_ + 5u * c_]);
    p.S[3] = uintBitsToFloat(pyr.v[a_ + 6u * c_]); p.S[4] = uintBitsToFloat(pyr.v[a_ + 7u * c_]); p.S[5] = uintBitsToFloat(pyr.v[a_ + 8u * c_]);
    p.seg = vec3(uintBitsToFloat(pyr.v[a_ + 9u * c_]), uintBitsToFloat(pyr.v[a_ + 10u * c_]), uintBitsToFloat(pyr.v[a_ + 11u * c_]));
    p.flux = uintBitsToFloat(pyr.v[a_ + 12u * c_]);
    uint f_ = pyr.v[a_ + 14u * c_];
    p.openStart = (f_ & 1u) != 0u; p.openEnd = (f_ & 2u) != 0u;
    p.rho = vec2(uintBitsToFloat(pyr.v[a_ + 15u * c_]), uintBitsToFloat(pyr.v[a_ + 16u * c_]));
    p.segT = vec3(uintBitsToFloat(pyr.v[a_ + 17u * c_]), uintBitsToFloat(pyr.v[a_ + 18u * c_]), uintBitsToFloat(pyr.v[a_ + 19u * c_]));
    return p;
}
vec4 dust_squeue_support(DustUintBuffer pyr, uint recs, uint r) {
    uint a_ = recs + r; const uint c_ = DUST_SQUEUE_CAPACITY;
    return vec4(uintBitsToFloat(pyr.v[a_ + 23u * c_]), uintBitsToFloat(pyr.v[a_ + 24u * c_]), uintBitsToFloat(pyr.v[a_ + 25u * c_]), uintBitsToFloat(pyr.v[a_ + 26u * c_]));
}
float dust_squeue_origin(DustUintBuffer pyr, uint recs, uint r) { return uintBitsToFloat(pyr.v[recs + r + 27u * DUST_SQUEUE_CAPACITY]); }
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

// ── Round 9 step Z: the leaf queue ───────────────────────────────────────────────────────────────
// The splitter draws no texel and stores no screen packet (both were the lanes' spill on Mali):
// every EWA leaf of the split walk goes to the leaf queue as the leaf packet itself — the split's
// q (16 words) and its sub-packet record's index (the consumer re-reads the exposure, colour and
// photometry there), its rounding seed (drawn by the splitter in the per-cluster thread's order,
// where it was: a leaf found on its band after the projection), the level the splitter chose and
// the level its per-packet valve alone gave (the gate and the texel-count atomics stay in the
// splitter, so the in-frame gate's order is the splitter's). The consumer dust_draw_leaf.comp projects
// the leaf again (dust_project_packet: the same bits), widens it past the valve as the splitter
// did, and draws the band's rows with dust_scatter_rows — the arithmetic the splitter's texel loop
// had, segment for segment. Its plain blob is a lane per record (the small leaves: a band of at
// most DUST_QUEUE_MIN_TEXELS texels; the large skipped), its workgroup blob a
// workgroup per entry of the large leaves' index list (a word per large leaf, appended when the
// band is known: bit 31 the spill, bits 0-30 the slot; dust_scatter_rows_wg; a leaf has at most
// DUST_SPLAT_MAX_TEXELS texels, the valve's). A full queue sends the leaf to the leaf spill, a second queue
// of the same format (the lane blob drains it in a second dispatch, DUST_VIEW_LEAF_SPILL); a leaf
// the spill refuses too is dropped and counted (DUST_LQUEUE_LOST, word 5 of the queue's header;
// the host logs it). Headers as the volume queue's (words 0-2 the indirect dispatch, 3 the raw
// append counter, 4 the capacity), the records structure-of-arrays at the capacity's stride. The
// three lie where the EWA header's offset words say (host-written per frame, words from the EWA
// queue base: dust::dust_queue_offsets).
const uint DUST_QUEUE_LQUEUE_OFFSET = 5u;   // the leaf queue
const uint DUST_QUEUE_LSPILL_OFFSET = 6u;   // the leaf spill
const uint DUST_QUEUE_BLIST_OFFSET  = 7u;   // the large leaves' index list
const uint DUST_LQUEUE_LOST = 5u;
const uint DUST_LQUEUE_RECORD_WORDS = 5u;
const uint DUST_LQUEUE_CAPACITY = 262144u;
const uint DUST_LQUEUE_SPILL_BIT = 0x80000000u;
// DustSplatPushConstants::flags bit, host-internal: dust_draw_leaf.comp (the lane blob) drains
// the leaf spill instead of the leaf queue
const uint DUST_VIEW_LEAF_SPILL = 1u << 15u;  // dust::DUST_VIEW_LEAF_SPILL (bits 20-27 are the debug value)
// the leaf queue header's frame-wide words (zero per frame, never reset per chunk: the host sizes
// the chunks from them; dust::DUST_LQUEUE_FRAME_*): the leaves (Σ the queue's word 3), the
// candidates (Σ the candidate header's word 3), the large leaves (Σ the list's word 3), the
// spilled leaves (Σ the spill's word 3); each added once per chunk by one lane of its consumer
const uint DUST_LQUEUE_FRAME_LEAVES = 8u;
const uint DUST_LQUEUE_FRAME_CANDIDATES = 9u;
const uint DUST_LQUEUE_FRAME_LARGE = 10u;
const uint DUST_LQUEUE_FRAME_SPILLED = 11u;
// the candidates the candidate buffer refused (Σ its word 5) and the largest chunk density
// (max ceil(word 3 / the chunk's clusters)); the SEQ order stage's workgroup 0
const uint DUST_LQUEUE_FRAME_REFUSED = 12u;
const uint DUST_LQUEUE_FRAME_CPC_MAX = 13u;
// a workgroup-per-record list's dispatch: x at most this, the rows in y (dust::DUST_LQUEUE_DISPATCH_MAX_X)
const uint DUST_LQUEUE_DISPATCH_MAX_X = 65535u;
uint dust_lqueue_base(DustUintBuffer pyr) { uint b = dust_queue_base(pyr); return b + pyr.v[b + DUST_QUEUE_LQUEUE_OFFSET]; }
uint dust_lqueue_base_o(uvec2 pyr) { uint b = dust_queue_base_o(pyr); return b + DustUintBuffer(pyr).v[b + DUST_QUEUE_LQUEUE_OFFSET]; }
uint dust_lspill_base(DustUintBuffer pyr) { uint b = dust_queue_base(pyr); return b + pyr.v[b + DUST_QUEUE_LSPILL_OFFSET]; }
uint dust_lspill_base_o(uvec2 pyr) { uint b = dust_queue_base_o(pyr); return b + DustUintBuffer(pyr).v[b + DUST_QUEUE_LSPILL_OFFSET]; }
uint dust_blist_base(DustUintBuffer pyr) { uint b = dust_queue_base(pyr); return b + pyr.v[b + DUST_QUEUE_BLIST_OFFSET]; }
uint dust_blist_base_o(uvec2 pyr) { uint b = dust_queue_base_o(pyr); return b + DustUintBuffer(pyr).v[b + DUST_QUEUE_BLIST_OFFSET]; }
uint dust_lqueue_capacity(DustUintBuffer pyr, uint base) {
    uint c = pyr.v[base + 4u];
    return c == 0u ? DUST_LQUEUE_CAPACITY : min(c, DUST_LQUEUE_CAPACITY);
}
uint dust_lqueue_capacity_o(uvec2 pyr, uint base) {
    uint c = DustUintBuffer(pyr).v[base + 4u];
    return c == 0u ? DUST_LQUEUE_CAPACITY : min(c, DUST_LQUEUE_CAPACITY);
}
// a record index, or ~0u when the queue is full; the dispatch a lane per record (`lanes` 64) or a
// workgroup per record (`lanes` 1)
uint dust_lqueue_append(DustUintBuffer pyr, uint base, uint lanes) {
    uint r = atomicAdd(pyr.v[base + 3u], 1u);
    if (r >= dust_lqueue_capacity(pyr, base)) return 0xFFFFFFFFu;
    // (a workgroup per record: x at most DUST_LQUEUE_DISPATCH_MAX_X, the rows in y, word 1 from 1)
    if (lanes == 1u) {
        atomicMax(pyr.v[base], min(r + 1u, DUST_LQUEUE_DISPATCH_MAX_X));
        if (r >= DUST_LQUEUE_DISPATCH_MAX_X) atomicMax(pyr.v[base + 1u], r / DUST_LQUEUE_DISPATCH_MAX_X + 1u);
    } else atomicMax(pyr.v[base], r / lanes + 1u);
    return r;
}
uint dust_lqueue_append_o(uvec2 pyr, uint base, uint lanes) {
    uint r = atomicAdd(DustUintBuffer(pyr).v[base + 3u], 1u);
    if (r >= dust_lqueue_capacity_o(pyr, base)) return 0xFFFFFFFFu;
    // (a workgroup per record: x at most DUST_LQUEUE_DISPATCH_MAX_X, the rows in y, word 1 from 1)
    if (lanes == 1u) {
        atomicMax(DustUintBuffer(pyr).v[base], min(r + 1u, DUST_LQUEUE_DISPATCH_MAX_X));
        if (r >= DUST_LQUEUE_DISPATCH_MAX_X) atomicMax(DustUintBuffer(pyr).v[base + 1u], r / DUST_LQUEUE_DISPATCH_MAX_X + 1u);
    } else atomicMax(DustUintBuffer(pyr).v[base], r / lanes + 1u);
    return r;
}
uint dust_lqueue_held(DustUintBuffer pyr, uint base) { return min(pyr.v[base + 3u], dust_lqueue_capacity(pyr, base)); }
uint dust_lqueue_held_o(uvec2 pyr, uint base) { return min(DustUintBuffer(pyr).v[base + 3u], dust_lqueue_capacity_o(pyr, base)); }
// words: 0 the seed, 1 the sub-packet record (recs + r), 2 the level (bit 31: a large leaf), 3 the
// valve's level, 4 the candidate slot (its screen packet: DUST_PQUEUE_*)
#define DUST_LQUEUE_PUT(pyr, base, cap, r, rec, seed, level, levelValve, cslot) { \
    uint b_ = (base) + DUST_VQUEUE_HEADER_WORDS + (r); uint c_ = (cap); \
    pyr.v[b_ + 0u * c_] = (seed); pyr.v[b_ + 1u * c_] = (rec); pyr.v[b_ + 2u * c_] = (level); pyr.v[b_ + 3u * c_] = (levelValve); pyr.v[b_ + 4u * c_] = (cslot); \
}
#define DUST_LQUEUE_GET(pyr, base, cap, r, seed, rec, level, levelValve, cslot) { \
    uint b_ = (base) + DUST_VQUEUE_HEADER_WORDS + (r); uint c_ = (cap); \
    seed = pyr.v[b_ + 0u * c_]; rec = pyr.v[b_ + 1u * c_]; level = pyr.v[b_ + 2u * c_]; levelValve = pyr.v[b_ + 3u * c_]; cslot = pyr.v[b_ + 4u * c_]; \
}
uint dust_leaf_slot_base(DustUintBuffer pyr, uint slot) { return (slot & DUST_LQUEUE_SPILL_BIT) != 0u ? dust_lspill_base(pyr) : dust_lqueue_base(pyr); }
uint dust_leaf_slot_base_o(uvec2 pyr, uint slot) { return (slot & DUST_LQUEUE_SPILL_BIT) != 0u ? dust_lspill_base_o(pyr) : dust_lqueue_base_o(pyr); }
// the leaf to the queue, else to the spill (a leaf neither takes is counted lost); a large one
// listed for the workgroup blob
void dust_leaf_emit(DustUintBuffer pyr, uint cslot, uint rec, uint seed, uint level, uint levelValve, bool large) {
    uint base = dust_lqueue_base(pyr);
    uint slot = dust_lqueue_append(pyr, base, 64u);
    if (slot == 0xFFFFFFFFu) {
        slot = dust_lqueue_append(pyr, dust_lspill_base(pyr), 64u);
        if (slot == 0xFFFFFFFFu) { atomicAdd(pyr.v[base + DUST_LQUEUE_LOST], 1u); return; }
        slot |= DUST_LQUEUE_SPILL_BIT;
    }
    base = dust_leaf_slot_base(pyr, slot);
    DUST_LQUEUE_PUT(pyr, base, dust_lqueue_capacity(pyr, base), slot & ~DUST_LQUEUE_SPILL_BIT, rec, seed, level | (large ? DUST_LQUEUE_SPILL_BIT : 0u), levelValve, cslot)
    if (large) {
        uint lb = dust_blist_base(pyr);
        uint i = dust_lqueue_append(pyr, lb, 1u);
        // (the list holds as many entries as the queue and the spill hold records: never full)
        if (i != 0xFFFFFFFFu) pyr.v[lb + DUST_VQUEUE_HEADER_WORDS + i] = slot;
    }
}
void dust_leaf_emit_o(uvec2 pyr, uint cslot, uint rec, uint seed, uint level, uint levelValve, bool large) {
    uint base = dust_lqueue_base_o(pyr);
    uint slot = dust_lqueue_append_o(pyr, base, 64u);
    if (slot == 0xFFFFFFFFu) {
        slot = dust_lqueue_append_o(pyr, dust_lspill_base_o(pyr), 64u);
        if (slot == 0xFFFFFFFFu) { atomicAdd(DustUintBuffer(pyr).v[base + DUST_LQUEUE_LOST], 1u); return; }
        slot |= DUST_LQUEUE_SPILL_BIT;
    }
    base = dust_leaf_slot_base_o(pyr, slot);
    DUST_LQUEUE_PUT(DustUintBuffer(pyr), base, dust_lqueue_capacity_o(pyr, base), slot & ~DUST_LQUEUE_SPILL_BIT, rec, seed, level | (large ? DUST_LQUEUE_SPILL_BIT : 0u), levelValve, cslot)
    if (large) {
        uint lb = dust_blist_base_o(pyr);
        uint i = dust_lqueue_append_o(pyr, lb, 1u);
        // (the list holds as many entries as the queue and the spill hold records: never full)
        if (i != 0xFFFFFFFFu) DustUintBuffer(pyr).v[lb + DUST_VQUEUE_HEADER_WORDS + i] = slot;
    }
}

// ── The candidate leaves (round 9 step Z2: the walk stage's output, the sequencer's input) ───
// The walk stage (dust_split.comp) draws nothing and seeds nothing: every leaf its walk reaches —
// a volumetric packet or an EWA leaf still unprojected — is written as a candidate, the record's
// candidates contiguous in the walk's order (the walk's counting pass sizes the record, the
// cluster's prefix places it, lane 0 reserves the cluster's run with one atomicAdd; the record's
// run in its words 45-46). The sequencer blob (dust_split.comp -DDUST_SPLIT_SEQ, dispatched like
// the splitter after it) walks every record's run in order through today's leaf path — the
// projection, the band and its rows, the seed (dust_next_packet_seed: its counting pass and the
// prefix are the sequencer's, so every seed is the per-cluster thread's) — and emits to the
// volume queue (dust_volume_emit) or the leaf queue (dust_leaf_emit). A cluster the buffer
// cannot hold whole is refused whole and counted (word 5, DUST_CQUEUE_LOST: the host logs it and
// sizes the chunk; the sequencer sees a run of 0). Header as the leaf queue's (word 3 the raw
// reservation counter, 4 a capacity override), then DUST_CQUEUE_RECORD_WORDS-word records
// as consecutive words, a record per DUST_CQUEUE_RECORD_WORDS (perf9_O: the walk writes through one
// pointer to the slot, the project stage reads it through one; the capacity no longer strides the
// words, it only bounds the slots). The buffer lies where word 6 of the leaf queue's
// header says (host-written per frame, words from the EWA queue base).
const uint DUST_LQUEUE_CQUEUE_OFFSET = 6u;
const uint DUST_CQUEUE_LOST = 5u;
const uint DUST_CQUEUE_RECORD_WORDS = 17u;
const uint DUST_CQUEUE_CAPACITY = 262144u;
// sub-packet record words 45-46: the record's run of candidates (its first, its count)
const uint DUST_SQUEUE_CAND_BASE = 45u;
const uint DUST_SQUEUE_CAND_COUNT = 46u;
uint dust_cqueue_base(DustUintBuffer pyr) { uint b = dust_queue_base(pyr); return b + pyr.v[dust_lqueue_base(pyr) + DUST_LQUEUE_CQUEUE_OFFSET]; }
uint dust_cqueue_base_o(uvec2 pyr) { uint b = dust_queue_base_o(pyr); return b + DustUintBuffer(pyr).v[dust_lqueue_base_o(pyr) + DUST_LQUEUE_CQUEUE_OFFSET]; }
uint dust_cqueue_capacity(DustUintBuffer pyr, uint base) {
    uint c = pyr.v[base + 4u];
    return c == 0u ? DUST_CQUEUE_CAPACITY : min(c, DUST_CQUEUE_CAPACITY);
}
uint dust_cqueue_capacity_o(uvec2 pyr, uint base) {
    uint c = DustUintBuffer(pyr).v[base + 4u];
    return c == 0u ? DUST_CQUEUE_CAPACITY : min(c, DUST_CQUEUE_CAPACITY);
}
// a run of n candidates, or ~0u (counted lost) when the buffer cannot hold it
uint dust_cqueue_reserve(DustUintBuffer pyr, uint base, uint n) {
    if (n == 0u) return 0u;
    uint r = atomicAdd(pyr.v[base + 3u], n);
    if (r + n > dust_cqueue_capacity(pyr, base)) { atomicAdd(pyr.v[base + DUST_CQUEUE_LOST], n); return 0xFFFFFFFFu; }
    // (word 0: the project stage's indirect dispatch, a lane per candidate)
    atomicMax(pyr.v[base], (r + n + 63u) / 64u);
    return r;
}
uint dust_cqueue_reserve_o(uvec2 pyr, uint base, uint n) {
    if (n == 0u) return 0u;
    uint r = atomicAdd(DustUintBuffer(pyr).v[base + 3u], n);
    if (r + n > dust_cqueue_capacity_o(pyr, base)) { atomicAdd(DustUintBuffer(pyr).v[base + DUST_CQUEUE_LOST], n); return 0xFFFFFFFFu; }
    // (word 0: the project stage's indirect dispatch, a lane per candidate)
    atomicMax(DustUintBuffer(pyr).v[base], (r + n + 63u) / 64u);
    return r;
}
// words: 0-2 mean, 3-8 S, 9-11 seg, 12 flux, 13-14 rho, 15 open ends (bit 0 start, 1 end) and
// bit 2 a volumetric packet, 16 the sub-packet record (recs + r)
#define DUST_CQUEUE_PUT(pyr, base, cap, r, q, volume, rec) { \
    uint b_ = (base) + DUST_VQUEUE_HEADER_WORDS + (r) * DUST_CQUEUE_RECORD_WORDS; const uint c_ = 1u; \
    pyr.v[b_ + 0u * c_] = floatBitsToUint((q).mean.x); pyr.v[b_ + 1u * c_] = floatBitsToUint((q).mean.y); pyr.v[b_ + 2u * c_] = floatBitsToUint((q).mean.z); \
    pyr.v[b_ + 3u * c_] = floatBitsToUint((q).S[0]); pyr.v[b_ + 4u * c_] = floatBitsToUint((q).S[1]); pyr.v[b_ + 5u * c_] = floatBitsToUint((q).S[2]); \
    pyr.v[b_ + 6u * c_] = floatBitsToUint((q).S[3]); pyr.v[b_ + 7u * c_] = floatBitsToUint((q).S[4]); pyr.v[b_ + 8u * c_] = floatBitsToUint((q).S[5]); \
    pyr.v[b_ + 9u * c_] = floatBitsToUint((q).seg.x); pyr.v[b_ + 10u * c_] = floatBitsToUint((q).seg.y); pyr.v[b_ + 11u * c_] = floatBitsToUint((q).seg.z); \
    pyr.v[b_ + 12u * c_] = floatBitsToUint((q).flux); \
    pyr.v[b_ + 13u * c_] = floatBitsToUint((q).rho.x); pyr.v[b_ + 14u * c_] = floatBitsToUint((q).rho.y); \
    pyr.v[b_ + 15u * c_] = ((q).openStart ? 1u : 0u) | ((q).openEnd ? 2u : 0u) | ((volume) ? 4u : 0u); \
    pyr.v[b_ + 16u * c_] = (rec); \
}
#define DUST_CQUEUE_GET(pyr, base, cap, r, q, volume, rec) { \
    uint b_ = (base) + DUST_VQUEUE_HEADER_WORDS + (r) * DUST_CQUEUE_RECORD_WORDS; const uint c_ = 1u; \
    (q).mean = vec3(uintBitsToFloat(pyr.v[b_ + 0u * c_]), uintBitsToFloat(pyr.v[b_ + 1u * c_]), uintBitsToFloat(pyr.v[b_ + 2u * c_])); \
    (q).S[0] = uintBitsToFloat(pyr.v[b_ + 3u * c_]); (q).S[1] = uintBitsToFloat(pyr.v[b_ + 4u * c_]); (q).S[2] = uintBitsToFloat(pyr.v[b_ + 5u * c_]); \
    (q).S[3] = uintBitsToFloat(pyr.v[b_ + 6u * c_]); (q).S[4] = uintBitsToFloat(pyr.v[b_ + 7u * c_]); (q).S[5] = uintBitsToFloat(pyr.v[b_ + 8u * c_]); \
    (q).seg = vec3(uintBitsToFloat(pyr.v[b_ + 9u * c_]), uintBitsToFloat(pyr.v[b_ + 10u * c_]), uintBitsToFloat(pyr.v[b_ + 11u * c_])); \
    (q).flux = uintBitsToFloat(pyr.v[b_ + 12u * c_]); \
    (q).rho = vec2(uintBitsToFloat(pyr.v[b_ + 13u * c_]), uintBitsToFloat(pyr.v[b_ + 14u * c_])); \
    (q).openStart = (pyr.v[b_ + 15u * c_] & 1u) != 0u; (q).openEnd = (pyr.v[b_ + 15u * c_] & 2u) != 0u; \
    (q).segT = vec3(0.0); \
    volume = (pyr.v[b_ + 15u * c_] & 4u) != 0u; \
    rec = pyr.v[b_ + 16u * c_]; \
}

// ── The screen packets (round 9 step Z3: the project stage's output, a slot per candidate) ───
// dust_project_leaf.comp (a lane per candidate, indirect from the candidate header) projects every
// EWA candidate (dust_project_packet) and writes its screen packet here, at the candidate's slot
// (consecutive words, DUST_PQUEUE_RECORD_WORDS a slot); the sequencer reads its fields at each use (never
// the packet whole), writes back the five the widening changes and the closed open ends, and
// dust_draw_leaf.comp draws from it, a pass of the segment at a time. Words: 0-1 mu, 2-4 cov, 5-6
// seg, 7 amp, 8 sigmaMin, 9 det, 10 depthAu, 11 open ends (bit 0 start, 1 end) and bit 2 on
// screen, 12-13 rho, 14-16 mean, 17-19 kx, 20-22 ky, 23-25 sun, 26 rhAu, 27-30 eye, 31 albedo, 32
// wMean; (perf9_O) 33-42 the leaf consumer's walk words (dust_draw_leaf.comp writes and reads them,
// nothing else does): 33 the record, 34 the seed, 35 the level's pyramid offset, 36 its width, 37 the
// lane walk's carry, 38 its drawn flag, 39 the level, 40-41 the open row's columns x0, x1, 42 the
// last row y1; (round 4) 43-46 the lane walk's staged texel (dust_l_t1/t2/t3, dust_l_ph_weight /
// _deposit; each texel in passes, its carried state here): 43 tau, 44 a, 45 b / a, 46 i0.
// Lies where word 7 of the leaf queue's header says (host-written, words from the EWA
// queue base); no header of its own.
const uint DUST_LQUEUE_PQUEUE_OFFSET = 7u;
const uint DUST_PQUEUE_RECORD_WORDS = 47u;
uint dust_pqueue_base(DustUintBuffer pyr) { uint b = dust_queue_base(pyr); return b + pyr.v[dust_lqueue_base(pyr) + DUST_LQUEUE_PQUEUE_OFFSET]; }
uint dust_pqueue_base_o(uvec2 pyr) { uint b = dust_queue_base_o(pyr); return b + DustUintBuffer(pyr).v[dust_lqueue_base_o(pyr) + DUST_LQUEUE_PQUEUE_OFFSET]; }
// (records are consecutive words — not structure-of-arrays at a run-time stride: Mali builds one
// 64-bit address per word of such a record and keeps them all live, 92 B of spill in the consumer;
// consecutive words are offsets from one address)
float dust_pq_f(DustUintBuffer pyr, uint base, uint cap, uint slot, uint k) { return uintBitsToFloat(pyr.v[base + slot * DUST_PQUEUE_RECORD_WORDS + k]); }
uint dust_pq_u(DustUintBuffer pyr, uint base, uint cap, uint slot, uint k) { return pyr.v[base + slot * DUST_PQUEUE_RECORD_WORDS + k]; }
uint dust_pq_u_o(uvec2 pyr, uint base, uint cap, uint slot, uint k) { return DustUintBuffer(pyr).v[base + slot * DUST_PQUEUE_RECORD_WORDS + k]; }
#define DUST_PQ_SET(pyr, base, cap, slot, k, f) pyr.v[(base) + (slot) * DUST_PQUEUE_RECORD_WORDS + (k)] = floatBitsToUint(f);
#define DUST_PQUEUE_PUT(pyr, base, cap, slot, P, on) { \
    DUST_PQ_SET(pyr, base, cap, slot, 0u, (P).mu.x) DUST_PQ_SET(pyr, base, cap, slot, 1u, (P).mu.y) \
    DUST_PQ_SET(pyr, base, cap, slot, 2u, (P).cov.x) DUST_PQ_SET(pyr, base, cap, slot, 3u, (P).cov.y) DUST_PQ_SET(pyr, base, cap, slot, 4u, (P).cov.z) \
    DUST_PQ_SET(pyr, base, cap, slot, 5u, (P).seg.x) DUST_PQ_SET(pyr, base, cap, slot, 6u, (P).seg.y) \
    DUST_PQ_SET(pyr, base, cap, slot, 7u, (P).amp) DUST_PQ_SET(pyr, base, cap, slot, 8u, (P).sigmaMin) DUST_PQ_SET(pyr, base, cap, slot, 9u, (P).det) DUST_PQ_SET(pyr, base, cap, slot, 10u, (P).depthAu) \
    pyr.v[(base) + (slot) * DUST_PQUEUE_RECORD_WORDS + 11u] = ((P).openStart ? 1u : 0u) | ((P).openEnd ? 2u : 0u) | ((on) ? 4u : 0u); \
    DUST_PQ_SET(pyr, base, cap, slot, 12u, (P).rho.x) DUST_PQ_SET(pyr, base, cap, slot, 13u, (P).rho.y) \
    DUST_PQ_SET(pyr, base, cap, slot, 14u, (P).mean.x) DUST_PQ_SET(pyr, base, cap, slot, 15u, (P).mean.y) DUST_PQ_SET(pyr, base, cap, slot, 16u, (P).mean.z) \
    DUST_PQ_SET(pyr, base, cap, slot, 17u, (P).kx.x) DUST_PQ_SET(pyr, base, cap, slot, 18u, (P).kx.y) DUST_PQ_SET(pyr, base, cap, slot, 19u, (P).kx.z) \
    DUST_PQ_SET(pyr, base, cap, slot, 20u, (P).ky.x) DUST_PQ_SET(pyr, base, cap, slot, 21u, (P).ky.y) DUST_PQ_SET(pyr, base, cap, slot, 22u, (P).ky.z) \
    DUST_PQ_SET(pyr, base, cap, slot, 23u, (P).sun.x) DUST_PQ_SET(pyr, base, cap, slot, 24u, (P).sun.y) DUST_PQ_SET(pyr, base, cap, slot, 25u, (P).sun.z) \
    DUST_PQ_SET(pyr, base, cap, slot, 26u, (P).rhAu) \
    DUST_PQ_SET(pyr, base, cap, slot, 27u, (P).eye.x) DUST_PQ_SET(pyr, base, cap, slot, 28u, (P).eye.y) DUST_PQ_SET(pyr, base, cap, slot, 29u, (P).eye.z) DUST_PQ_SET(pyr, base, cap, slot, 30u, (P).eye.w) \
    DUST_PQ_SET(pyr, base, cap, slot, 31u, (P).albedo) DUST_PQ_SET(pyr, base, cap, slot, 32u, (P).wMean) \
}
// the five words the widening changes (dust_widen_screen_packet)
#define DUST_PQUEUE_PUT_WIDEN(pyr, base, cap, slot, P) { \
    DUST_PQ_SET(pyr, base, cap, slot, 2u, (P).cov.x) DUST_PQ_SET(pyr, base, cap, slot, 4u, (P).cov.z) \
    DUST_PQ_SET(pyr, base, cap, slot, 7u, (P).amp) DUST_PQ_SET(pyr, base, cap, slot, 8u, (P).sigmaMin) DUST_PQ_SET(pyr, base, cap, slot, 9u, (P).det) \
}
// the packet whole (a use reads the fields it needs: the rest are dead loads)
DustScreenPacket dust_pq_get(DustUintBuffer pyr, uint base, uint cap, uint slot) {
    DustScreenPacket P;
    P.mu = vec2(dust_pq_f(pyr, base, cap, slot, 0u), dust_pq_f(pyr, base, cap, slot, 1u));
    P.cov = vec3(dust_pq_f(pyr, base, cap, slot, 2u), dust_pq_f(pyr, base, cap, slot, 3u), dust_pq_f(pyr, base, cap, slot, 4u));
    P.seg = vec2(dust_pq_f(pyr, base, cap, slot, 5u), dust_pq_f(pyr, base, cap, slot, 6u));
    P.amp = dust_pq_f(pyr, base, cap, slot, 7u);
    P.sigmaMin = dust_pq_f(pyr, base, cap, slot, 8u);
    P.det = dust_pq_f(pyr, base, cap, slot, 9u);
    P.depthAu = dust_pq_f(pyr, base, cap, slot, 10u);
    uint open = dust_pq_u_o(uvec2(pyr), base, cap, slot, 11u);
    P.openStart = (open & 1u) != 0u;
    P.openEnd = (open & 2u) != 0u;
    P.rho = vec2(dust_pq_f(pyr, base, cap, slot, 12u), dust_pq_f(pyr, base, cap, slot, 13u));
    P.mean = vec3(dust_pq_f(pyr, base, cap, slot, 14u), dust_pq_f(pyr, base, cap, slot, 15u), dust_pq_f(pyr, base, cap, slot, 16u));
    P.kx = vec3(dust_pq_f(pyr, base, cap, slot, 17u), dust_pq_f(pyr, base, cap, slot, 18u), dust_pq_f(pyr, base, cap, slot, 19u));
    P.ky = vec3(dust_pq_f(pyr, base, cap, slot, 20u), dust_pq_f(pyr, base, cap, slot, 21u), dust_pq_f(pyr, base, cap, slot, 22u));
    P.sun = vec3(dust_pq_f(pyr, base, cap, slot, 23u), dust_pq_f(pyr, base, cap, slot, 24u), dust_pq_f(pyr, base, cap, slot, 25u));
    P.rhAu = dust_pq_f(pyr, base, cap, slot, 26u);
    P.eye = vec4(dust_pq_f(pyr, base, cap, slot, 27u), dust_pq_f(pyr, base, cap, slot, 28u), dust_pq_f(pyr, base, cap, slot, 29u), dust_pq_f(pyr, base, cap, slot, 30u));
    P.albedo = dust_pq_f(pyr, base, cap, slot, 31u);
    P.wMean = dust_pq_f(pyr, base, cap, slot, 32u);
    return P;
}
bool dust_pq_on(DustUintBuffer pyr, uint base, uint cap, uint slot) { return (dust_pq_u(pyr, base, cap, slot, 11u) & 4u) != 0u; }
bool dust_pq_on_o(uvec2 pyr, uint base, uint cap, uint slot) { return (dust_pq_u_o(pyr, base, cap, slot, 11u) & 4u) != 0u; }

#ifdef DUST_DRAW_LEAF
// (perf9_O) The leaf consumer (dust_draw_leaf.comp, which defines DUST_DRAW_LEAF): a leaf's texels
// drawn as one flat walk, dust_scatter_segment's texel arithmetic in its order (capsule x area x
// invUnit, x the weight when lit, the rounding with the carry, the colour words, the depth
// minimum) on the texels of dust_scatter_rows in its order (rows y0..y1, texels x0..x1, the carry
// re-seeded at each segment's first texel x0 + j * DUST_SEGMENT_TEXELS). Mali-G52 holds a loop's
// live-through values over its whole body: the walk keeps in registers only its position (ty, tx
// for a lane; nothing for the workgroup's lanes), every other value of the leaf is in its slot's
// walk words (DUST_PQUEUE_RECORD_WORDS) or in shared words and read where it is used.
//
// The leaf's screen packet through one per-lane pointer to its slot's word 0, the fields at
// constant offsets (one address, not one per field)
// (the workgroup blob: volatile, which glslang also decorates Coherent; the slot's address is
// uniform, its loads scalar loads on AMD RDNA, which amdvlk served stale without it; see
// DustLeafListC in dust_draw_leaf.comp)
#ifdef DUST_LEAF_WG
layout(buffer_reference, std430, buffer_reference_align = 4) volatile buffer DustLRow { uint v[]; };
#else
layout(buffer_reference, std430, buffer_reference_align = 4) buffer DustLRow { uint v[]; };
#endif
uvec2 dust_l_slot;
uvec2 dust_l_addr(DustUintBuffer pyr, uint word) { uvec2 a = uvec2(pyr); uint o = word * 4u; uint lo = a.x + o; return uvec2(lo, a.y + (lo < o ? 1u : 0u)); }
uvec2 dust_l_addr_o(uvec2 pyr, uint word) { uvec2 a = uvec2(pyr); uint o = word * 4u; uint lo = a.x + o; return uvec2(lo, a.y + (lo < o ? 1u : 0u)); }
#define LQF(k) uintBitsToFloat(DustLRow(dust_l_slot).v[k])
DustScreenPacket dust_l_get() {
    DustScreenPacket P;
    P.mu = vec2(LQF(0u), LQF(1u));
    P.cov = vec3(LQF(2u), LQF(3u), LQF(4u));
    P.seg = vec2(LQF(5u), LQF(6u));
    P.amp = LQF(7u);
    P.sigmaMin = LQF(8u);
    P.det = LQF(9u);
    P.depthAu = LQF(10u);
    uint open = DustLRow(dust_l_slot).v[11u];
    P.openStart = (open & 1u) != 0u;
    P.openEnd = (open & 2u) != 0u;
    P.rho = vec2(LQF(12u), LQF(13u));
    P.mean = vec3(LQF(14u), LQF(15u), LQF(16u));
    P.kx = vec3(LQF(17u), LQF(18u), LQF(19u));
    P.ky = vec3(LQF(20u), LQF(21u), LQF(22u));
    P.sun = vec3(LQF(23u), LQF(24u), LQF(25u));
    P.rhAu = LQF(26u);
    P.eye = vec4(LQF(27u), LQF(28u), LQF(29u), LQF(30u));
    P.albedo = LQF(31u);
    P.wMean = LQF(32u);
    return P;
}
// The volatile view of the slot: a field read inside the walk is re-read at its use (the same
// bits), never hoisted and held across the walk. Its loads and stores go through functions: a
// buffer-reference access inside dust_capsule_tau's precise cone otherwise spreads NoContraction
// to every buffer-reference store of the shader (glslang's precise propagation)
layout(buffer_reference, std430, buffer_reference_align = 4) volatile buffer DustLRowV { uint v[]; };
// The workgroup blob's walk words (33-46) are shared words, not the slot's: lane 0's stores were
// read back by every lane through uniform (scalar, on AMD) loads that did not see them
#ifdef DUST_LEAF_WG
shared uint dust_l_ws[14];
uint dust_lv(uint k) { return (k >= 33u && k <= 46u) ? dust_l_ws[k - 33u] : DustLRowV(dust_l_slot).v[k]; }
#else
uint dust_lv(uint k) { return DustLRowV(dust_l_slot).v[k]; }
#endif
void dust_lv_set(uint k, uint v) { DustLRowV(dust_l_slot).v[k] = v; }
// (every call site has the push-constant buffer's address as `pyr`, a uvec2)
// (perf9_O fix 3) through a function: a store to DustUintBuffer(pyr) in the caller has no symbol
// root, so glslang's precise propagation gave every such store of the shader one object and
// spread NoContraction over the whole lane walk (the tree's set is dust_capsule_tau's cone only)
#ifdef DUST_LEAF_WG
void dust_lv_store(uvec2 pyr_, uint k_, uint v_) { dust_l_ws[k_ - 33u] = v_; }
#else
void dust_lv_store(uvec2 pyr_, uint k_, uint v_) { DustUintBuffer(pyr_).v[((dust_l_slot.x - pyr_.x) >> 2u) + k_] = v_; }
#endif
#define dust_lv_set(k_, v_) dust_lv_store(uvec2(pyr), (k_), (v_))
#define DUST_O_COL(n_) DustUintBuffer(pyr).v[DUST_LS(33u) + (n_) * DUST_SQUEUE_CAPACITY]
#define LQV(k) uintBitsToFloat(dust_lv(k))
#define DUST_LS(k) dust_lv(k)
DustScreenPacket dust_l_getv() {
    DustScreenPacket P;
    P.mu = vec2(LQV(0u), LQV(1u));
    P.cov = vec3(LQV(2u), LQV(3u), LQV(4u));
    P.seg = vec2(LQV(5u), LQV(6u));
    P.amp = LQV(7u);
    P.sigmaMin = LQV(8u);
    P.det = LQV(9u);
    P.depthAu = LQV(10u);
    uint open = dust_lv(11u);
    P.openStart = (open & 1u) != 0u;
    P.openEnd = (open & 2u) != 0u;
    P.rho = vec2(LQV(12u), LQV(13u));
    P.mean = vec3(LQV(14u), LQV(15u), LQV(16u));
    P.kx = vec3(LQV(17u), LQV(18u), LQV(19u));
    P.ky = vec3(LQV(20u), LQV(21u), LQV(22u));
    P.sun = vec3(LQV(23u), LQV(24u), LQV(25u));
    P.rhAu = LQV(26u);
    P.eye = vec4(LQV(27u), LQV(28u), LQV(29u), LQV(30u));
    P.albedo = LQV(31u);
    P.wMean = LQV(32u);
    return P;
}
// dust_capsule_tau and dust_texel_weight with each field read from the slot where the expression
// uses it (the same expressions on the same bits; dust_common.glsl's text with P.<field> replaced
// by the field's read): only the values the expression is computing are live
float dust_capsule_tau_l(vec2 x) {
#define d vec2(LQV(5u), LQV(6u))
    float det = LQV(9u);
    if (!(det > 0.0)) return 0.0;
    float m00 = LQV(4u) / det, m01 = -LQV(3u) / det, m11 = LQV(2u) / det;
    vec2 e = x - vec2(LQV(0u), LQV(1u));
    float c = m00 * e.x * e.x + 2.0 * m01 * e.x * e.y + m11 * e.y * e.y;
    float a = m00 * d.x * d.x + 2.0 * m01 * d.x * d.y + m11 * d.y * d.y;
    if (!(a > 1e-6)) return LQV(7u) * exp(-0.5 * c);
    float b = m00 * d.x * e.x + m01 * (d.x * e.y + d.y * e.x) + m11 * d.y * e.y;
    float sa = sqrt(a);
    float u0 = b / a;
    // an open end continues into the neighbouring bin: no Gaussian cap there
    float phiEnd = ((dust_lv(11u) & 2u) != 0u) ? 1.0 : 0.5 * (1.0 + dust_erf((1.0 - u0) * sa * 0.70710678));
    float phiStart = ((dust_lv(11u) & 1u) != 0u) ? 0.0 : 0.5 * (1.0 + dust_erf(-u0 * sa * 0.70710678));
    float span = phiEnd - phiStart;
    // the squared distance to the line from the residual e - u0 d, not c - b^2/a (catastrophic
    // cancellation for a chord whose start is far off screen; dust::capsule_tau)
    precise vec2 r = e - u0 * d;
    precise float q = max(m00 * r.x * r.x + 2.0 * m01 * r.x * r.y + m11 * r.y * r.y, 0.0);
    // a linear density along the segment (dust::capsule_tau): the integral of u exp(-a(u-u0)^2/2)
    // over it is u0 I0 + (E_start - E_end)/a with the Gaussian's values at the closed ends
    float i0 = sqrt(2.0 * DUST_PI_F32 / a) * span;
    float weight;
    if (LQV(13u) == LQV(12u)) {
        weight = LQV(12u) * i0;
    } else {
        float i1;
        if (!((dust_lv(11u) & 1u) != 0u) && !((dust_lv(11u) & 2u) != 0u)) {
            // both ends closed: the Gaussians' difference from expm1 where they are close (the
            // cancellation at a << 1), the plain difference otherwise (dust::capsule_tau)
            float z1 = -0.5 * a * u0 * u0;
            float z2 = -0.5 * a * (1.0 - u0) * (1.0 - u0);
            float dz = z2 - z1;
            float diff = abs(dz) < 1.0 ? -exp(z1) * dust_expm1(dz) : exp(z1) - exp(z2);
            i1 = u0 * i0 + diff / a;
        } else {
            float eStart = ((dust_lv(11u) & 1u) != 0u) ? 0.0 : exp(-0.5 * a * u0 * u0);
            float eEnd = ((dust_lv(11u) & 2u) != 0u) ? 0.0 : exp(-0.5 * a * (1.0 - u0) * (1.0 - u0));
            i1 = u0 * i0 + (eStart - eEnd) / a;
        }
        weight = max(LQV(12u) * i0 + (LQV(13u) - LQV(12u)) * i1, 0.0);
    }
    return LQV(7u) * exp(-0.5 * q) * weight;
}
#undef d
float dust_texel_weight_l(vec2 x) {
    vec2 dxy = x - vec2(LQV(0u), LQV(1u));
    vec3 d = vec3(LQV(17u), LQV(18u), LQV(19u)) * dxy.x + vec3(LQV(20u), LQV(21u), LQV(22u)) * dxy.y;
    vec3 pos = vec3(LQV(14u), LQV(15u), LQV(16u)) + d;
    vec3 rVec = vec3(LQV(23u), LQV(24u), LQV(25u)) * (-LQV(26u) * DUST_AU_M) + d;
    float rn = max(length(rVec), 1.0);
    return dust_radiance_weight_o(uvec2(dust_phot_lut), LQV(31u), rVec * (-1.0 / rn), vec4(LQV(27u), LQV(28u), LQV(29u), LQV(30u)).xyz - vec4(LQV(27u), LQV(28u), LQV(29u), LQV(30u)).w * pos, rn / DUST_AU_M);
}
// One texel of the walk: dust_scatter_segment's tau, weight and deposit on texel (tx, ty), the
// carry, the record, the level's offset and width and the drawn flag in the slot's walk words
bool dust_l_texel(uvec2 pyr, float invUnit, float s, int ty, int tx) {
    float yc = (float(ty) + 0.5) * s;
    float area = s * s;
    vec2 x = vec2((float(tx) + 0.5) * s, yc);
    float tau = dust_capsule_tau_l(x) * area * invUnit;
    if (LQF(31u) > 0.0) tau *= dust_texel_weight_l(x);
    if (!(tau >= 0.0) || isinf(tau)) return false;
    float tauC = min(tau, DUST_COUNT_MAX_PER_ADD) + uintBitsToFloat(DUST_LS(37u));
    uint counts = uint(tauC + 0.5);
    dust_lv_set(37u, floatBitsToUint(tauC - float(counts)));
    if (counts == 0u) return false;
    uint i = DUST_LS(35u) + (uint(ty) * DUST_LS(36u) + uint(tx)) * DUST_PYRAMID_TEXEL_WORDS;
    dust_atomic_add_sat_o(pyr, i, counts);
    // the colour words component by component (uvec3(float(counts) * color * DUST_COLOR_SCALE + hf)
    // is component-wise: the same three expressions), each colour word read where it is used
    float hf = float(dust_hash32(i) >> 8u) * (1.0 / 16777216.0);
    dust_atomic_add_sat_o(pyr, i + 1u, uint(float(counts) * uintBitsToFloat(DUST_O_COL(20u)) * DUST_COLOR_SCALE + hf));
    dust_atomic_add_sat_o(pyr, i + 2u, uint(float(counts) * uintBitsToFloat(DUST_O_COL(21u)) * DUST_COLOR_SCALE + hf));
    dust_atomic_add_sat_o(pyr, i + 3u, uint(float(counts) * uintBitsToFloat(DUST_O_COL(22u)) * DUST_COLOR_SCALE + hf));
    uint depthBits = floatBitsToUint(max(LQV(10u), 0.0));
    uint cur = DustUintBuffer(pyr).v[i + 4u];
    if (cur == 0u || depthBits < cur) {
        uint seen = atomicCompSwap(DustUintBuffer(pyr).v[i + 4u], 0u, depthBits);
        if (seen != 0u) atomicMin(DustUintBuffer(pyr).v[i + 4u], depthBits);
    }
    dust_lv_set(38u, 1u);
    return true;
}
#ifndef DUST_LEAF_WG
shared uint dust_l_lw[64];
#define DUST_O_LRL() { memoryBarrierShared(); dust_l_slot = dust_l_addr_o(pyr, dust_l_lw[gl_LocalInvocationID.x]); }
#define DUST_O_LRL_SET(w_) dust_l_lw[gl_LocalInvocationID.x] = (w_);
#else
#define DUST_O_LRL()
#define DUST_O_LRL_SET(w_)
#endif
// T1: 0 = tau final (word 43, the point-like or empty case), 1 = go on to T2
uint dust_l_t1(uvec2 pyr, float invUnit, float s, int ty, int tx) {
    float area = s * s;
    vec2 x = vec2((float(tx) + 0.5) * s, (float(ty) + 0.5) * s);
    vec2 d = vec2(LQV(5u), LQV(6u));
    float det = LQV(9u);
    if (!(det > 0.0)) { dust_lv_set(43u, floatBitsToUint(0.0 * area * invUnit)); return 0u; }
    float m00 = LQV(4u) / det, m01 = -LQV(3u) / det, m11 = LQV(2u) / det;
    vec2 e = x - vec2(LQV(0u), LQV(1u));
    float c = m00 * e.x * e.x + 2.0 * m01 * e.x * e.y + m11 * e.y * e.y;
    // (a and b precise: dust_capsule_tau's cone, u0 = b / a feeding r)
    precise float a = m00 * d.x * d.x + 2.0 * m01 * d.x * d.y + m11 * d.y * d.y;
    if (!(a > 1e-6)) { dust_lv_set(43u, floatBitsToUint(LQV(7u) * exp(-0.5 * c) * area * invUnit)); return 0u; }
    precise float b = m00 * d.x * e.x + m01 * (d.x * e.y + d.y * e.x) + m11 * d.y * e.y;
    dust_lv_set(44u, floatBitsToUint(a));
    dust_lv_set(45u, floatBitsToUint(b / a));
    return 1u;
}
void dust_l_t2(uvec2 pyr) {
    float a = uintBitsToFloat(dust_lv(44u)), u0 = uintBitsToFloat(dust_lv(45u));
    float sa = sqrt(a);
    float phiEnd = ((dust_lv(11u) & 2u) != 0u) ? 1.0 : 0.5 * (1.0 + dust_erf((1.0 - u0) * sa * 0.70710678));
    float phiStart = ((dust_lv(11u) & 1u) != 0u) ? 0.0 : 0.5 * (1.0 + dust_erf(-u0 * sa * 0.70710678));
    float span = phiEnd - phiStart;
    dust_lv_set(46u, floatBitsToUint(sqrt(2.0 * DUST_PI_F32 / a) * span));
}
void dust_l_t3(uvec2 pyr, float invUnit, float s, int ty, int tx) {
    float area = s * s;
    vec2 x = vec2((float(tx) + 0.5) * s, (float(ty) + 0.5) * s);
    vec2 d = vec2(LQV(5u), LQV(6u));
    float det = LQV(9u);
    float m00 = LQV(4u) / det, m01 = -LQV(3u) / det, m11 = LQV(2u) / det;
    vec2 e = x - vec2(LQV(0u), LQV(1u));
    float a = uintBitsToFloat(dust_lv(44u)), u0 = uintBitsToFloat(dust_lv(45u)), i0 = uintBitsToFloat(dust_lv(46u));
    precise vec2 r = e - u0 * d;
    precise float q = max(m00 * r.x * r.x + 2.0 * m01 * r.x * r.y + m11 * r.y * r.y, 0.0);
    float weight;
    if (LQV(13u) == LQV(12u)) {
        weight = LQV(12u) * i0;
    } else {
        float i1;
        if (!((dust_lv(11u) & 1u) != 0u) && !((dust_lv(11u) & 2u) != 0u)) {
            float z1 = -0.5 * a * u0 * u0;
            float z2 = -0.5 * a * (1.0 - u0) * (1.0 - u0);
            float dz = z2 - z1;
            float diff = abs(dz) < 1.0 ? -exp(z1) * dust_expm1(dz) : exp(z1) - exp(z2);
            i1 = u0 * i0 + diff / a;
        } else {
            float eStart = ((dust_lv(11u) & 1u) != 0u) ? 0.0 : exp(-0.5 * a * u0 * u0);
            float eEnd = ((dust_lv(11u) & 2u) != 0u) ? 0.0 : exp(-0.5 * a * (1.0 - u0) * (1.0 - u0));
            i1 = u0 * i0 + (eStart - eEnd) / a;
        }
        weight = max(LQV(12u) * i0 + (LQV(13u) - LQV(12u)) * i1, 0.0);
    }
    dust_lv_set(43u, floatBitsToUint(LQV(7u) * exp(-0.5 * q) * weight * area * invUnit));
}
uint dust_l_ph_tau(uvec2 pyr, float invUnit, float s, int ty, int tx) {
    float yc = (float(ty) + 0.5) * s;
    float area = s * s;
    vec2 x = vec2((float(tx) + 0.5) * s, yc);
    float tau = dust_capsule_tau_l(x) * area * invUnit;
    dust_lv_set(43u, floatBitsToUint(tau));
    return LQF(31u) > 0.0 ? 1u : 2u;
}
void dust_l_ph_weight(uvec2 pyr, float s, int ty, int tx) {
    float yc = (float(ty) + 0.5) * s;
    vec2 x = vec2((float(tx) + 0.5) * s, yc);
    float tau = uintBitsToFloat(dust_lv(43u));
    tau *= dust_texel_weight_l(x);
    dust_lv_set(43u, floatBitsToUint(tau));
}
bool dust_l_ph_deposit(uvec2 pyr, int ty, int tx) {
    float tau = uintBitsToFloat(dust_lv(43u));
    if (!(tau >= 0.0) || isinf(tau)) return false;
    float tauC = min(tau, DUST_COUNT_MAX_PER_ADD) + uintBitsToFloat(DUST_LS(37u));
    uint counts = uint(tauC + 0.5);
    dust_lv_set(37u, floatBitsToUint(tauC - float(counts)));
    if (counts == 0u) return false;
    uint i = DUST_LS(35u) + (uint(ty) * DUST_LS(36u) + uint(tx)) * DUST_PYRAMID_TEXEL_WORDS;
    dust_atomic_add_sat_o(pyr, i, counts);
    // the colour words component by component (uvec3(float(counts) * color * DUST_COLOR_SCALE + hf)
    // is component-wise: the same three expressions), each colour word read where it is used
    float hf = float(dust_hash32(i) >> 8u) * (1.0 / 16777216.0);
    dust_atomic_add_sat_o(pyr, i + 1u, uint(float(counts) * uintBitsToFloat(DUST_O_COL(20u)) * DUST_COLOR_SCALE + hf));
    dust_atomic_add_sat_o(pyr, i + 2u, uint(float(counts) * uintBitsToFloat(DUST_O_COL(21u)) * DUST_COLOR_SCALE + hf));
    dust_atomic_add_sat_o(pyr, i + 3u, uint(float(counts) * uintBitsToFloat(DUST_O_COL(22u)) * DUST_COLOR_SCALE + hf));
    uint depthBits = floatBitsToUint(max(LQV(10u), 0.0));
    uint cur = DustUintBuffer(pyr).v[i + 4u];
    if (cur == 0u || depthBits < cur) {
        uint seen = atomicCompSwap(DustUintBuffer(pyr).v[i + 4u], 0u, depthBits);
        if (seen != 0u) atomicMin(DustUintBuffer(pyr).v[i + 4u], depthBits);
    }
    dust_lv_set(38u, 1u);
    return true;
}
// dust_scatter_rows over the leaf's packet as one flat walk: each step opens the next row (its
// columns into the walk words) or draws the next texel
bool dust_scatter_rows_pq(uvec2 pyr, float invUnit, uint pbase, uint pcap, uint slot, uint rec, float s,
                          uint off, uint w, uint h, uint seed, int rowA, int rowB) {
    dust_l_slot = dust_l_addr_o(pyr, pbase + slot * DUST_PQUEUE_RECORD_WORDS);
    int y0, y1;
    if (!dust_band_rows(dust_band_new(dust_l_get()), s, h, y0, y1)) return false;
    y0 = max(y0, rowA);
    y1 = min(y1, rowB);
    dust_lv_set(33u, rec); dust_lv_set(34u, seed); dust_lv_set(35u, off); dust_lv_set(36u, w); dust_lv_set(38u, 0u);
    dust_lv_set(40u, 0u); dust_lv_set(41u, 0u); dust_lv_set(42u, uint(y1));
    DUST_O_LRL_SET((dust_l_slot.x - uvec2(pyr).x) >> 2u)
    int ty = y0 - 1, tx = 1;
    uint ph = 0u;
    while (true) {
        DUST_O_LRL()
        if (ph == 0u) {
            if (tx > int(DUST_LS(41u))) {
                ++ty;
                if (ty > int(DUST_LS(42u))) break;
                int c0, c1;
                float s = float(1u << DUST_LS(39u));
                if (!dust_band_columns(dust_band_new(dust_l_getv()), (float(ty) + 0.5) * s, s, DUST_LS(36u), c0, c1)) continue;
                dust_lv_set(40u, uint(c0)); dust_lv_set(41u, uint(c1)); tx = c0;
            }
            if (uint(tx - int(DUST_LS(40u))) % DUST_SEGMENT_TEXELS == 0u) dust_lv_set(37u, floatBitsToUint(dust_segment_carry(DUST_LS(34u), (uint(ty) << 16u) | uint(tx))));
            ph = dust_l_t1(pyr, invUnit, float(1u << DUST_LS(39u)), ty, tx) != 0u ? 5u : (LQF(31u) > 0.0 ? 1u : 2u);
        } else if (ph == 5u) {
            dust_l_t2(pyr);
            ph = 6u;
        } else if (ph == 6u) {
            dust_l_t3(pyr, invUnit, float(1u << DUST_LS(39u)), ty, tx);
            ph = LQF(31u) > 0.0 ? 1u : 2u;
        } else if (ph == 1u) {
            dust_l_ph_weight(pyr, float(1u << DUST_LS(39u)), ty, tx);
            ph = 2u;
        } else {
            dust_l_ph_deposit(pyr, ty, tx);
            ph = 0u;
            ++tx;
        }
    }
    return DUST_LS(38u) != 0u;
}
#ifdef DUST_QUEUE_WORKGROUP
// (perf9_O) the workgroup's carries and flags, a word per lane (the texel step reads and writes them
// where it uses them: no carry is held over the texel's tau)
shared float dust_l_carry[64];
shared uint dust_l_anyf[64];
shared uint dust_l_total;
shared uint dust_l_k[64];
shared uint dust_l_rt[64];
shared int dust_l_yc0;
bool dust_l_texel_wg(uvec2 pyr, float invUnit, float s, int ty, int tx) {
    float yc = (float(ty) + 0.5) * s;
    float area = s * s;
    vec2 x = vec2((float(tx) + 0.5) * s, yc);
    float tau = dust_capsule_tau_l(x) * area * invUnit;
    if (LQF(31u) > 0.0) tau *= dust_texel_weight_l(x);
    if (!(tau >= 0.0) || isinf(tau)) return false;
    float tauC = min(tau, DUST_COUNT_MAX_PER_ADD) + dust_l_carry[gl_LocalInvocationID.x];
    uint counts = uint(tauC + 0.5);
    dust_l_carry[gl_LocalInvocationID.x] = tauC - float(counts);
    if (counts == 0u) return false;
    uint i = DUST_LS(35u) + (uint(ty) * DUST_LS(36u) + uint(tx)) * DUST_PYRAMID_TEXEL_WORDS;
    dust_atomic_add_sat_o(pyr, i, counts);
    // the colour words component by component (uvec3(float(counts) * color * DUST_COLOR_SCALE + hf)
    // is component-wise: the same three expressions), each colour word read where it is used
    float hf = float(dust_hash32(i) >> 8u) * (1.0 / 16777216.0);
    dust_atomic_add_sat_o(pyr, i + 1u, uint(float(counts) * uintBitsToFloat(DUST_O_COL(20u)) * DUST_COLOR_SCALE + hf));
    dust_atomic_add_sat_o(pyr, i + 2u, uint(float(counts) * uintBitsToFloat(DUST_O_COL(21u)) * DUST_COLOR_SCALE + hf));
    dust_atomic_add_sat_o(pyr, i + 3u, uint(float(counts) * uintBitsToFloat(DUST_O_COL(22u)) * DUST_COLOR_SCALE + hf));
    uint depthBits = floatBitsToUint(max(LQV(10u), 0.0));
    uint cur = DustUintBuffer(pyr).v[i + 4u];
    if (cur == 0u || depthBits < cur) {
        uint seen = atomicCompSwap(DustUintBuffer(pyr).v[i + 4u], 0u, depthBits);
        if (seen != 0u) atomicMin(DustUintBuffer(pyr).v[i + 4u], depthBits);
    }
    dust_l_anyf[gl_LocalInvocationID.x] = 1u;
    return true;
}
// dust_scatter_rows_wg over a slot's screen packet (the same chunks, prefix and segments; a lane's
// segments k = lane, lane + lanes, ... drawn as one flat walk over their texels in order, the carry
// re-seeded at each segment's first texel). The leaf's words (record, seed, level, the level's
// offset and width) are lane 0's stores into the slot's walk words, read at their uses
bool dust_scatter_rows_wg_pq(uvec2 pyr, float invUnit, uint pbase, uint pcap, uint slot, uint rec, float s,
                             uint off, uint w, uint h, uint seed, int rowA, int rowB, uint lanes, uint lane) {
    dust_l_slot = dust_l_addr_o(pyr, pbase + slot * DUST_PQUEUE_RECORD_WORDS);
    int y0, y1;
    if (!dust_band_rows(dust_band_new(dust_l_get()), s, h, y0, y1)) return false;
    y0 = max(y0, rowA);
    y1 = min(y1, rowB);
    if (lane == 0u) { dust_lv_set(33u, rec); dust_lv_set(34u, seed); dust_lv_set(35u, off); dust_lv_set(36u, w); dust_lv_set(42u, uint(y1)); }
    dust_l_anyf[lane] = 0u;
    memoryBarrierBuffer();
    barrier();
    for (int yc0 = y0; yc0 <= int(DUST_LS(42u)); yc0 += int(DUST_QUEUE_ROW_CHUNK)) {
        int n = min(int(DUST_QUEUE_ROW_CHUNK), int(DUST_LS(42u)) - yc0 + 1);
        for (int r = int(lane); r < n; r += int(lanes)) {
            int x0, x1;
            float sr = float(1u << DUST_LS(39u));
            bool ok = dust_band_columns(dust_band_new(dust_l_getv()), (float(yc0 + r) + 0.5) * sr, sr, DUST_LS(36u), x0, x1);
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
            dust_l_total = acc;
            dust_l_yc0 = yc0;
        }
        barrier();
        // the lane's walk position in shared words: k (its segment), r (its row) << 16 | tx - x0[r]
        if (lane < dust_l_total) {
            int r = 0;
            while (dust_q_seg[r + 1] <= lane) r++;
            int tx = dust_q_x0[r] + int((lane - dust_q_seg[r]) * DUST_SEGMENT_TEXELS);
            dust_l_k[lane] = lane;
            dust_l_rt[lane] = (uint(r) << 16u) | uint(tx - dust_q_x0[r]);
            dust_l_carry[lane] = dust_segment_carry(DUST_LS(34u), (uint(yc0 + r) << 16u) | uint(tx));
            while (true) {
                {
                    int rr = int(dust_l_rt[gl_LocalInvocationID.x] >> 16u);
                    dust_l_texel_wg(pyr, invUnit, float(1u << DUST_LS(39u)), dust_l_yc0 + rr, dust_q_x0[rr] + int(dust_l_rt[gl_LocalInvocationID.x] & 0xFFFFu));
                }
                // (a fence: the walk's shared words below are re-read, not the values read before the texel)
                memoryBarrierShared();
                uint kk = dust_l_k[gl_LocalInvocationID.x];
                int r2 = int(dust_l_rt[gl_LocalInvocationID.x] >> 16u);
                int t2 = dust_q_x0[r2] + int(dust_l_rt[gl_LocalInvocationID.x] & 0xFFFFu) + 1;
                int xa = dust_q_x0[r2] + int((kk - dust_q_seg[r2]) * DUST_SEGMENT_TEXELS);
                if (t2 > min(xa + int(DUST_SEGMENT_TEXELS) - 1, dust_q_x1[r2])) {
                    kk += lanes;
                    if (kk >= dust_l_total) break;
                    while (dust_q_seg[r2 + 1] <= kk) r2++;
                    t2 = dust_q_x0[r2] + int((kk - dust_q_seg[r2]) * DUST_SEGMENT_TEXELS);
                    dust_l_k[gl_LocalInvocationID.x] = kk;
                    dust_l_carry[gl_LocalInvocationID.x] = dust_segment_carry(DUST_LS(34u), (uint(dust_l_yc0 + r2) << 16u) | uint(t2));
                }
                dust_l_rt[gl_LocalInvocationID.x] = (uint(r2) << 16u) | uint(t2 - dust_q_x0[r2]);
            }
        }
        barrier();
    }
    return dust_l_anyf[lane] != 0u;
}
#endif
#endif // DUST_DRAW_LEAF
