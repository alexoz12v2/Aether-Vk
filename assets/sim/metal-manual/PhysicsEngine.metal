#include <metal_atomic>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

constant uint SUBGROUP_SIZE = 32;
constant float AU_TO_KM = 149597870.7f, M_EARTH_TO_KG = 5.9722e24f;

// ----------------------------------------------------------------------------
// ATOMICS, SIMD & UTILS
// ----------------------------------------------------------------------------
#define PTR(T, addr) ((device T*)(uintptr_t)(addr))

inline void atomic_add_f(device atomic_uint* a, float v) {
  uint e = atomic_load_explicit(a, memory_order_relaxed);
  while (!atomic_compare_exchange_weak_explicit(
      a, &e, as_type<uint>(as_type<float>(e) + v), memory_order_relaxed,
      memory_order_relaxed));
}
inline void atomic_add_f_tg(threadgroup atomic_uint* a, float v) {
  uint e = atomic_load_explicit(a, memory_order_relaxed);
  while (!atomic_compare_exchange_weak_explicit(
      a, &e, as_type<uint>(as_type<float>(e) + v), memory_order_relaxed,
      memory_order_relaxed));
}
inline uint2 add64(uint2 a, uint2 b) {
  return uint2(a.x + b.x, a.y + b.y + ((a.x + b.x < a.x) ? 1u : 0u));
}
inline float dt_sec(uint2 dt) {
  return float(dt.x) * 1e-6f + float(dt.y) * 4294.967296f;
}
inline uint hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// Use static_cast to explicitly invoke MSL 3.0's simd_vote conversion operator
inline ulong get_ballot(bool cond) {
  return static_cast<ulong>(simd_ballot(cond));
}

inline float4 q_cj(float4 q) { return float4(-q.xyz, q.w); }
inline float4 q_ml(float4 a, float4 b) {
  return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                a.w * b.w - dot(a.xyz, b.xyz));
}
inline float3 q_rt(float4 q, float3 v) {
  float3 t = 2.0f * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
inline float3 q_ir(float4 q, float3 v) { return q_rt(q_cj(q), v); }
inline float3x3 q_m3(float4 q) {
  float x2 = q.x * q.x, y2 = q.y * q.y, z2 = q.z * q.z, xy = q.x * q.y,
        xz = q.x * q.z, yz = q.y * q.z, wx = q.w * q.x, wy = q.w * q.y,
        wz = q.w * q.z;
  return float3x3(
      float3(1.f - 2.f * (y2 + z2), 2.f * (xy + wz), 2.f * (xz - wy)),
      float3(2.f * (xy - wz), 1.f - 2.f * (x2 + z2), 2.f * (yz + wx)),
      float3(2.f * (xz + wy), 2.f * (yz - wx), 1.f - 2.f * (x2 + y2)));
}
inline void g_tan(float3 n, thread float3& t1, thread float3& t2) {
  if (abs(n.x) >= 0.577f)
    t1 = normalize(float3(n.y, -n.x, 0.f));
  else
    t1 = normalize(float3(0.f, n.z, -n.y));
  t2 = cross(n, t1);
}

// ----------------------------------------------------------------------------
// DATA STRUCTURES (Scalar Layout)
// ----------------------------------------------------------------------------
struct MultiBvhNode {
  float mx[32], mxx[32], my[32], mxy[32], mz[32], mxz[32];
  uint chd[32], met[32];
  float mas[32], cx[32], cy[32], cz[32];
  uint pst[32], pct[32];
  uint2 vmk;
  uint par, pad, prm[8][32];
};
inline bool is_lf(uint m) { return (m & 0x80000000u) != 0; }
inline uint g_idx(uint m) { return m & 0x07FFFFFFu; }
inline uint pk_mt(bool l, uint f, uint s, uint i) {
  return (i & 0x07FFFFFFu) | ((s & 3) << 27) | ((f & 3) << 29) |
         (l ? 0x80000000u : 0);
}
inline bool is_vd(uint2 v, uint l) {
  return l < 32 ? (v.x & (1u << l)) != 0 : (v.y & (1u << (l - 32))) != 0;
}
inline bool iAABB(float3 mnA, float3 mxA, float3 mnB, float3 mxB) {
  return all(mnA <= mxB) && all(mxA >= mnB);
}

struct ColId {
  uint eid, pid;
};
struct PkPair {
  ColId a, b;
  float toi;
  packed_float3 n, p;
  float d;
};
struct SpPair {
  uint v, ea, pa, eb, pb;
  float toi;
  packed_float3 n, p;
  float d;
};
struct CrPairD {
  uint v, ma, mi, lc;
  float toi;
  packed_float3 n, p;
  float d;
};
struct CrPair {
  uint ma, mi, lc, pd;
};
struct RigidBody {
  float4 pm, ori, lv, av, iI;
  uint wid, lst, lct, shp;
  packed_float3 ext;
  uint p2;
};
struct LcaEnt {
  ulong bvh;
  float4x4 tr, itr;
  packed_float3 lv;
  uint rt;
  packed_float3 av;
  uint ty, po, tp, st;
  float scl;
  uint shp;
  packed_float3 sdt;
};
struct FEmit {
  packed_float3 p;
  float mu;
  packed_float3 n;
  uint ty;
  float tr, sc;
  uint p2[2];
};
struct TLASLeaf {
  packed_float3 mn;
  uint eidx;
  packed_float3 mx;
  uint met;
};
struct Wrench {
  atomic_uint fx, fy, fz, tx, ty, tz;
};
struct EntHdr {
  uint typ, p[3];
};
struct RPart {
  uint il, ih, al, ah;
  packed_float3 p;
  float m;
  packed_float3 v;
  uint act;
};
struct DInd {
  uint vc, ic, fv, fi;
};

struct AtCnt {
  atomic_uint c[1];
};
struct PBuf {
  atomic_uint c;
  uint2 p[1];
};
struct CPBuf {
  atomic_uint c;
  CrPair p[1];
};
struct PColBuf {
  uint dx, dy, dz;
  atomic_uint c;
  PkPair p[1];
};
struct SColBuf {
  atomic_uint c;
  SpPair p[1];
};
struct CSColBuf {
  atomic_uint c;
  CrPairD p[1];
};
struct DIdx {
  uint c, i[1];
};

// ----------------------------------------------------------------------------
// GJK/EPA CORE
// ----------------------------------------------------------------------------
inline float3 sup(uint t, float3 d, float4x4 m, float3 dr) {
  float3x3 r = float3x3(m.columns[0].xyz, m.columns[1].xyz, m.columns[2].xyz);
  float3 ld = transpose(r) * dr, rs(0);
  if (t == 0) {
    float l = length(ld);
    rs = (l > 1e-6f ? ld / l : float3(1, 0, 0)) * d.x;
  } else if (t == 1) {
    rs.x = dot(float3(1, 0, 0), ld) > 0 ? d.x : -d.x;
    rs.y = dot(float3(0, 1, 0), ld) > 0 ? d.y : -d.y;
    rs.z = dot(float3(0, 0, 1), ld) > 0 ? d.z : -d.z;
  }
  return (m * float4(rs, 1)).xyz;
}
inline float ef_m(uint s, float3 d, float3 rA, float3 rB, float iMA, float iMB,
                  float3 iIA, float3 iIB, float4 qA, float4 qB, float3x3 lw) {
  if (s == 1) {
    float3 rw = lw * rA, dw = lw * d;
    float3 ix = q_rt(qA, iIA * q_ir(qA, cross(rw, dw)));
    return 1.f /
           max(dot(transpose(lw) * ix, cross(rA, d)) + iMA +
                   dot(q_rt(qB, iIB * q_ir(qB, cross(rB, d))), cross(rB, d)) +
                   iMB,
               1e-6f);
  }
  return 1.f /
         max(iMA + iMB +
                 dot(q_rt(qA, iIA * q_ir(qA, cross(rA, d))), cross(rA, d)) +
                 dot(q_rt(qB, iIB * q_ir(qB, cross(rB, d))), cross(rB, d)),
             1e-6f);
}

struct MK {
  float3 p, a, b;
};
struct Smp {
  MK p[4];
  int c;
};
inline void c_cls(thread Smp& s, thread float3& a, thread float3& b) {
  if (s.c == 1) {
    a = s.p[0].a;
    b = s.p[0].b;
  } else if (s.c == 2) {
    float3 ab = s.p[0].p - s.p[1].p;
    float t = clamp(dot(-s.p[1].p, ab) / dot(ab, ab), 0.f, 1.f);
    a = s.p[1].a + (s.p[0].a - s.p[1].a) * t;
    b = s.p[1].b + (s.p[0].b - s.p[1].b) * t;
  } else if (s.c == 3) {
    float3 ab = s.p[1].p - s.p[2].p, ac = s.p[0].p - s.p[2].p,
           n = cross(ab, ac);
    float u = dot(cross(ac, n), -s.p[2].p) / dot(n, n),
          v = dot(cross(n, ab), -s.p[2].p) / dot(n, n), w = 1.f - u - v;
    a = s.p[2].a * w + s.p[1].a * u + s.p[0].a * v;
    b = s.p[2].b * w + s.p[1].b * u + s.p[0].b * v;
  }
}
inline bool d_smp(thread Smp& s, thread float3& d) {
  if (s.c == 2) {
    float3 ab = s.p[0].p - s.p[1].p, ao = -s.p[1].p;
    if (dot(ab, ao) > 0)
      d = cross(cross(ab, ao), ab);
    else {
      s.p[0] = s.p[1];
      s.c = 1;
      d = ao;
    }
    return false;
  }
  if (s.c == 3) {
    float3 ab = s.p[1].p - s.p[2].p, ac = s.p[0].p - s.p[2].p, ao = -s.p[2].p,
           abc = cross(ab, ac);
    if (dot(cross(abc, ac), ao) > 0) {
      if (dot(ac, ao) > 0) {
        s.p[0] = s.p[0];
        s.p[1] = s.p[2];
        s.c = 2;
        d = cross(cross(ac, ao), ac);
      } else {
        if (dot(ab, ao) > 0) {
          s.p[0] = s.p[1];
          s.p[1] = s.p[2];
          s.c = 2;
          d = cross(cross(ab, ao), ab);
        } else {
          s.p[0] = s.p[2];
          s.c = 1;
          d = ao;
        }
      }
    } else {
      if (dot(cross(ab, abc), ao) > 0) {
        if (dot(ab, ao) > 0) {
          s.p[0] = s.p[1];
          s.p[1] = s.p[2];
          s.c = 2;
          d = cross(cross(ab, ao), ab);
        } else {
          s.p[0] = s.p[2];
          s.c = 1;
          d = ao;
        }
      } else {
        if (dot(abc, ao) > 0)
          d = abc;
        else {
          MK t = s.p[0];
          s.p[0] = s.p[1];
          s.p[1] = t;
          d = -abc;
        }
      }
    }
    return false;
  }
  if (s.c == 4) {
    float3 ab = s.p[2].p - s.p[3].p, ac = s.p[1].p - s.p[3].p,
           ad = s.p[0].p - s.p[3].p, ao = -s.p[3].p, abc = cross(ab, ac),
           acd = cross(ac, ad), adb = cross(ad, ab);
    if (dot(abc, ao) > 0) {
      s.p[0] = s.p[1];
      s.p[1] = s.p[2];
      s.p[2] = s.p[3];
      s.c = 3;
      d = abc;
      return false;
    }
    if (dot(acd, ao) > 0) {
      s.p[0] = s.p[0];
      s.p[1] = s.p[1];
      s.p[2] = s.p[3];
      s.c = 3;
      d = acd;
      return false;
    }
    if (dot(adb, ao) > 0) {
      s.p[0] = s.p[0];
      s.p[1] = s.p[2];
      s.p[2] = s.p[3];
      s.c = 3;
      d = adb;
      return false;
    }
    return true;
  }
  return false;
}
inline float gjk_d(uint t1, float3 d1, float4x4 m1, uint t2, float3 d2,
                   float4x4 m2, thread float3& pa, thread float3& pb) {
  float3 d = float3(1, 0, 0), sa = sup(t1, d1, m1, -d), sb = sup(t2, d2, m2, d);
  Smp s;
  s.p[0] = {sa - sb, sa, sb};
  s.c = 1;
  float3 v = s.p[0].p;
  for (int i = 0; i < 32; ++i) {
    if (dot(v, v) < 1e-6f) break;
    d = -v;
    float3 p1 = sup(t1, d1, m1, d), p2 = sup(t2, d2, m2, -d);
    MK w = {p1 - p2, p1, p2};
    if (dot(w.p, d) - dot(v, d) < 1e-4f) break;
    s.p[s.c++] = w;
    if (d_smp(s, d)) {
      pa = pb = float3(0);
      return -0.01f;
    }
    if (s.c == 1)
      v = s.p[0].p;
    else {
      c_cls(s, pa, pb);
      v = pa - pb;
    }
  }
  c_cls(s, pa, pb);
  return length(pa - pb);
}
inline bool c_toi(uint t1, float3 d1, float4x4 m1, float3 v1, uint t2,
                  float3 d2, float4x4 m2, float3 v2, float tl, thread float& t,
                  thread float3& n, thread float3& p, thread float& dp) {
  t = 0;
  float3 vr = v1 - v2;
  float3 pa, pb;
  if (length(vr) < 1e-6f) {
    float d = gjk_d(t1, d1, m1, t2, d2, m2, pa, pb);
    if (d <= 0) {
      t = 0;
      dp = -d;
      n = length(pa - pb) > 1e-6f ? normalize(pa - pb) : float3(1, 0, 0);
      p = (pa + pb) * 0.5f;
      return true;
    }
    return false;
  }
  for (int i = 0; i < 10; ++i) {
    float4x4 c1 = m1;
    c1.columns[3].xyz += v1 * t;
    float4x4 c2 = m2;
    c2.columns[3].xyz += v2 * t;
    float d = gjk_d(t1, d1, c1, t2, d2, c2, pa, pb);
    if (d <= tl) {
      t = t;
      dp = -d;
      n = length(pa - pb) > 1e-6f ? normalize(pa - pb) : float3(1, 0, 0);
      p = (pa + pb) * 0.5f;
      return true;
    }
    float3 nn = d > 1e-6f ? normalize(pa - pb) : float3(1, 0, 0);
    float vc = -dot(vr, nn);
    if (vc <= 0) return false;
    t += d / vc;
    if (t > 1.f) return false;
  }
  return false;
}

// ============================================================================
// PUSH CONSTANTS (Native ulong for BDA)
// ============================================================================
struct PC_MotRefit {
  ulong bvh, didx;
  uint tot;
};
struct PC_CCD {
  ulong bvh, out, pts;
  uint rt, tot;
  float rd, dt;
};
struct PC_LBVHB {
  ulong bvh, mrt, cnt, pts;
  uint num;
  float rd, dt;
};
struct PC_LCP {
  ulong pts, pcol, ccol, out, rbs, lca;
  uint tcl, sp;
  float dt, res;
};
struct PC_BPC {
  ulong ent, raw, rr, rp, pp, ml, ll;
  uint mxp;
};
struct PC_Sort {
  ulong in, out, hst;
  uint num, shf, stg, blk;
};
struct PC_IntP1 {
  ulong pts;
  float dt;
  uint tot;
};
struct PC_RBF {
  ulong rbs, wr;
  uint nb, pad;
};
struct PC_BPB {
  ulong ent, lvs;
  uint2 dt;
  uint tot;
};
struct PC_SComp {
  ulong spi, pko;
  uint tot;
};
struct PC_NCCD {
  ulong ent, out, cout, pts, prs, cprs, lca;
  float dt, rd;
  uint sp;
};
struct PC_Emit {
  ulong pts, cnd, bvh;
  packed_float3 sun;
  float dt;
  uint mxp, nmc;
  ulong cnt;
  uint rt;
};
struct PC_Mor {
  ulong mot, pts;
  uint num;
  packed_float3 smn, smx;
};
struct PC_Gra {
  ulong col, clr, wgt;
  uint tot;
};
struct PC_BPCr {
  ulong lca, mlv, ent, lq, rr, rp, pp, cp;
  uint tq, mx;
};
struct PC_BPSce {
  ulong tls, lvs, prs;
  uint rt, tot;
};
struct PC_IntP3 {
  ulong rbs, wr, em;
  float dt;
  uint nb, ni, ne;
};
struct PC_App {
  ulong pts, pc, cc, imp, rbs, lca;
  uint sp;
};
struct PC_Colp {
  ulong bin, mul, map;
  uint num;
};
struct PC_BPSlf {
  ulong bvh, pts, wr;
  uint rt, tot;
  float rd, stf;
};
struct PC_IntP4 {
  ulong pts, clk;
  float dt;
  uint t, dl, dh, cl, ch;
};
struct PC_MotB {
  ulong bvh, pts;
  uint num;
  float dt, rd;
};
struct PC_Conv {
  ulong ao, mg, id, ct;
  uint ix, of;
};
struct PC_Pre {
  ulong bvh, ct;
  uint num;
};
struct PC_Clr {
  ulong r, rr, rp, rl, i;
};
struct PC_RToi {
  ulong pts, col, toi;
  float rd, dt;
};
struct PC_BHut {
  ulong pts, bvh, cl, wr;
  uint num;
  float dt, th, G, sq;
  uint rt, thr;
};

// ============================================================================
// 27 KERNELS
// ============================================================================

kernel void bp_clear(constant PC_Clr& pc [[buffer(0)]]) {
  atomic_store_explicit(&PTR(atomic_uint, pc.r)[0], 0, memory_order_relaxed);
  atomic_store_explicit(&PTR(atomic_uint, pc.rr)[0], 0, memory_order_relaxed);
  atomic_store_explicit(&PTR(atomic_uint, pc.rp)[0], 0, memory_order_relaxed);
  atomic_store_explicit(&PTR(atomic_uint, pc.rl)[0], 0, memory_order_relaxed);
  atomic_store_explicit(&PTR(atomic_uint, pc.i)[0], 0, memory_order_relaxed);
}

kernel void lbvh_prepass(constant PC_Pre& pc [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
  if (id < pc.num)
    atomic_store_explicit(&PTR(AtCnt, pc.ct)->c[id], 0, memory_order_relaxed);
  if (id == 0) PTR(MultiBvhNode, pc.bvh)[0].par = 0xFFFFFFFFu;
}

kernel void motion_refit(constant PC_MotRefit& pc [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
  if (id >= pc.tot) return;
  device MultiBvhNode* bvh = PTR(MultiBvhNode, pc.bvh);
  uint n = PTR(DIdx, pc.didx)->i[id + 4];
  for (uint i = 0; i < 2; ++i) {
    uint c = bvh[n].chd[i];
    if (is_lf(bvh[n].met[i])) {
      bvh[n].mx[i] = bvh[c].mx[0];
      bvh[n].mxx[i] = bvh[c].mxx[0];
      bvh[n].my[i] = bvh[c].my[0];
      bvh[n].mxy[i] = bvh[c].mxy[0];
      bvh[n].mz[i] = bvh[c].mz[0];
      bvh[n].mxz[i] = bvh[c].mxz[0];
    } else {
      bvh[n].mx[i] = min(bvh[c].mx[0], bvh[c].mx[1]);
      bvh[n].mxx[i] = max(bvh[c].mxx[0], bvh[c].mxx[1]);
      bvh[n].my[i] = min(bvh[c].my[0], bvh[c].my[1]);
      bvh[n].mxy[i] = max(bvh[c].mxy[0], bvh[c].mxy[1]);
      bvh[n].mz[i] = min(bvh[c].mz[0], bvh[c].mz[1]);
      bvh[n].mxz[i] = max(bvh[c].mxz[0], bvh[c].mxz[1]);
    }
  }
}

kernel void bp_bounds_gen(constant PC_BPB& pc [[buffer(0)]],
                          uint id [[thread_position_in_grid]]) {
  if (id >= pc.tot) return;
  device RigidBody* r = PTR(RigidBody, pc.ent);
  device TLASLeaf* l = PTR(TLASLeaf, pc.lvs);
  float3 c = r[id].pm.xyz, ext = float3(r[id].ext),
         sw = r[id].lv.xyz * dt_sec(pc.dt);
  l[id].mn = packed_float3(min(c - ext, c - ext + sw));
  l[id].mx = packed_float3(max(c + ext, c + ext + sw));
  l[id].eidx = id;
  l[id].met = pk_mt(true, 0, r[id].shp, id);
}

kernel void bp_classify(constant PC_BPC& pc [[buffer(0)]],
                        uint id [[thread_position_in_grid]]) {
  device PBuf* raw = PTR(PBuf, pc.raw);
  if (id >= atomic_load_explicit(&raw->c, memory_order_relaxed)) return;
  device EntHdr* h = PTR(EntHdr, pc.ent);
  uint2 p = raw->p[id];
  uint tA = h[p.x].typ, tB = h[p.y].typ;
  if (tA > tB) {
    uint t = p.x;
    p.x = p.y;
    p.y = t;
    t = tA;
    tA = tB;
    tB = t;
  }
  device PBuf* tgt = nullptr;
  uint2 op = p;
  if (tA == 0 && tB == 0)
    tgt = PTR(PBuf, pc.pp);
  else if (tA == 0 && tB == 1) {
    tgt = PTR(PBuf, pc.rp);
    op = uint2(p.y, p.x);
  } else if (tA == 1 && tB == 1)
    tgt = PTR(PBuf, pc.rr);
  else if (tB == 2) {
    tgt = (tA == 2) ? PTR(PBuf, pc.ll) : PTR(PBuf, pc.ml);
  }
  if (tgt) {
    uint c = atomic_fetch_add_explicit(&tgt->c, 1, memory_order_relaxed);
    if (c < pc.mxp) tgt->p[c] = op;
  }
}

kernel void stream_compact(constant PC_SComp& pc [[buffer(0)]],
                           uint id [[thread_position_in_grid]]) {
  device SColBuf* in = PTR(SColBuf, pc.spi);
  uint c = atomic_load_explicit(&in->c, memory_order_relaxed);
  device PColBuf* op = PTR(PColBuf, pc.pko);
  if (id == 0) {
    op->dx = (c + 127) / 128;
    op->dy = 1;
    op->dz = 1;
    atomic_store_explicit(&op->c, c, memory_order_relaxed);
  }
  if (id < c) {
    SpPair d = in->p[id];
    op->p[id].a = {d.ea, d.pa};
    op->p[id].b = {d.eb, d.pb};
    op->p[id].toi = d.toi;
    op->p[id].n = d.n;
    op->p[id].p = d.p;
    op->p[id].d = d.d;
  }
}

kernel void integrate_particles_p1_p2(constant PC_IntP1& pc [[buffer(0)]],
                                      uint id [[thread_position_in_grid]]) {
  if (id >= pc.tot) return;
  device atomic_uint* p = PTR(atomic_uint, pc.pts);
  uint b = (id / 32) * 320 + (id % 32);
  float m =
      as_type<float>(atomic_load_explicit(&p[b + 192], memory_order_relaxed));
  if (m <= 0) return;
  float3 v = float3(as_type<float>(
                        atomic_load_explicit(&p[b + 96], memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&p[b + 128],
                                                        memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&p[b + 160],
                                                        memory_order_relaxed))),
         f = float3(as_type<float>(atomic_load_explicit(&p[b + 224],
                                                        memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&p[b + 256],
                                                        memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&p[b + 288],
                                                        memory_order_relaxed)));
  float3 vh = v + f * (1.f / m) * (0.5f * pc.dt);
  float3 pn =
      float3(as_type<float>(atomic_load_explicit(&p[b], memory_order_relaxed)),
             as_type<float>(
                 atomic_load_explicit(&p[b + 32], memory_order_relaxed)),
             as_type<float>(
                 atomic_load_explicit(&p[b + 64], memory_order_relaxed))) +
      vh * pc.dt;
  atomic_store_explicit(&p[b], as_type<uint>(pn.x), memory_order_relaxed);
  atomic_store_explicit(&p[b + 32], as_type<uint>(pn.y), memory_order_relaxed);
  atomic_store_explicit(&p[b + 64], as_type<uint>(pn.z), memory_order_relaxed);
  atomic_store_explicit(&p[b + 96], as_type<uint>(vh.x), memory_order_relaxed);
  atomic_store_explicit(&p[b + 128], as_type<uint>(vh.y), memory_order_relaxed);
  atomic_store_explicit(&p[b + 160], as_type<uint>(vh.z), memory_order_relaxed);
  atomic_store_explicit(&p[b + 224], 0, memory_order_relaxed);
  atomic_store_explicit(&p[b + 256], 0, memory_order_relaxed);
  atomic_store_explicit(&p[b + 288], 0, memory_order_relaxed);
}

kernel void integrate_particles_p4_5(constant PC_IntP4& pc [[buffer(0)]],
                                     uint id [[thread_position_in_grid]]) {
  if (id == 0)
    PTR(uint2, pc.clk)[0] = add64(uint2(pc.cl, pc.ch), uint2(pc.dl, pc.dh));
  if (id >= pc.t) return;
  device atomic_uint* p = PTR(atomic_uint, pc.pts);
  uint b = (id / 32) * 320 + (id % 32);
  float m =
      as_type<float>(atomic_load_explicit(&p[b + 192], memory_order_relaxed));
  if (m <= 0) return;
  float3 v = float3(as_type<float>(
                        atomic_load_explicit(&p[b + 96], memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&p[b + 128],
                                                        memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&p[b + 160],
                                                        memory_order_relaxed))),
         f = float3(as_type<float>(atomic_load_explicit(&p[b + 224],
                                                        memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&p[b + 256],
                                                        memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&p[b + 288],
                                                        memory_order_relaxed)));
  float3 vn = v + f * (1.f / m) * (0.5f * pc.dt);
  atomic_store_explicit(&p[b + 96], as_type<uint>(vn.x), memory_order_relaxed);
  atomic_store_explicit(&p[b + 128], as_type<uint>(vn.y), memory_order_relaxed);
  atomic_store_explicit(&p[b + 160], as_type<uint>(vn.z), memory_order_relaxed);
}

kernel void rb_force_assign(constant PC_RBF& pc [[buffer(0)]],
                            uint wg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lid [[thread_position_in_threadgroup]]) {
  if (wg >= pc.nb) return;
  RigidBody b = PTR(RigidBody, pc.rbs)[wg];
  float3 af(0), at(0);
  threadgroup float3 sf[8], st[8];
  device Wrench* wr = PTR(Wrench, pc.wr);
  for (uint i = lid; i < b.lct; i += 128) {
    device Wrench& w = wr[b.lst + i];
    af += float3(
        as_type<float>(atomic_load_explicit(&w.fx, memory_order_relaxed)),
        as_type<float>(atomic_load_explicit(&w.fy, memory_order_relaxed)),
        as_type<float>(atomic_load_explicit(&w.fz, memory_order_relaxed)));
    at += float3(
        as_type<float>(atomic_load_explicit(&w.tx, memory_order_relaxed)),
        as_type<float>(atomic_load_explicit(&w.ty, memory_order_relaxed)),
        as_type<float>(atomic_load_explicit(&w.tz, memory_order_relaxed)));
  }
  af = float3(simd_sum(af.x), simd_sum(af.y), simd_sum(af.z));
  at = float3(simd_sum(at.x), simd_sum(at.y), simd_sum(at.z));
  if (lid % 32 == 0) {
    sf[sg] = af;
    st[sg] = at;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float3 tf(0), tt(0);
    for (uint i = 0; i < 8; ++i) {
      tf += sf[i];
      tt += st[i];
    }
    atomic_add_f(&wr[b.wid].fx, tf.x);
    atomic_add_f(&wr[b.wid].fy, tf.y);
    atomic_add_f(&wr[b.wid].fz, tf.z);
    atomic_add_f(&wr[b.wid].tx, tt.x);
    atomic_add_f(&wr[b.wid].ty, tt.y);
    atomic_add_f(&wr[b.wid].tz, tt.z);
  }
}

kernel void integrate_bodies_p3(constant PC_IntP3& pc [[buffer(0)]],
                                uint id [[thread_position_in_grid]]) {
  if (id >= pc.nb) return;
  device RigidBody* r = PTR(RigidBody, pc.rbs);
  device Wrench* wr = PTR(Wrench, pc.wr);
  device FEmit* em = PTR(FEmit, pc.em);
  RigidBody b = r[id];
  float m = b.pm.w, im = m > 0 ? 1.f / m : 0;
  float3 ii = b.iI.xyz, ifw = float3(ii.x > 1e-14f ? 1 / ii.x : 0,
                                     ii.y > 1e-14f ? 1 / ii.y : 0,
                                     ii.z > 1e-14f ? 1 / ii.z : 0);
  float3 pn = b.pm.xyz, vn = b.lv.xyz, wn = b.av.xyz;
  float4 qn = b.ori;
  float3 fn = float3(
      as_type<float>(atomic_load_explicit(&wr[b.wid].fx, memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&wr[b.wid].fy, memory_order_relaxed)),
      as_type<float>(
          atomic_load_explicit(&wr[b.wid].fz, memory_order_relaxed)));
  float3 tn = float3(
      as_type<float>(atomic_load_explicit(&wr[b.wid].tx, memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&wr[b.wid].ty, memory_order_relaxed)),
      as_type<float>(
          atomic_load_explicit(&wr[b.wid].tz, memory_order_relaxed)));
  for (uint e = 0; e < pc.ne; ++e) {
    FEmit f = em[e];
    if (f.ty == 0) {
      float3 dr = float3(f.p) - pn;
      float ds = dot(dr, dr) * f.sc * f.sc;
      if (ds > 1e-6f)
        fn += normalize(dr) *
              ((f.mu * m * (1.f - exp(-(ds * ds * sqrt(ds))))) / ds);
    } else if (f.ty == 1) {
      float d = dot(pn - float3(f.p), float3(f.n));
      if (d >= 0 && d <= f.tr) fn += float3(f.n) * f.mu;
    }
  }
  float3 vm = vn + .5f * pc.dt * (fn * im), pnx = pn + pc.dt * vm,
         vnx = vn + pc.dt * (fn * im);
  float3 tl = q_ir(qn, tn), wl = q_ir(qn, wn), wm = wl;
  for (uint i = 0; i < pc.ni; ++i)
    wm = wl + .5f * pc.dt * (ii * (tl - cross(wm, ifw * wm)));
  float3 wnx = q_rt(qn, 2.f * wm - wl);
  float4 qnx = normalize(qn + .5f * pc.dt * q_ml(float4(q_rt(qn, wm), 0), qn));
  r[id].pm = float4(pnx, m);
  r[id].ori = qnx;
  r[id].lv = float4(vnx, b.lv.w);
  r[id].av = float4(wnx, b.av.w);
  atomic_store_explicit(&wr[b.wid].fx, 0, memory_order_relaxed);
  atomic_store_explicit(&wr[b.wid].fy, 0, memory_order_relaxed);
  atomic_store_explicit(&wr[b.wid].fz, 0, memory_order_relaxed);
  atomic_store_explicit(&wr[b.wid].tx, 0, memory_order_relaxed);
  atomic_store_explicit(&wr[b.wid].ty, 0, memory_order_relaxed);
  atomic_store_explicit(&wr[b.wid].tz, 0, memory_order_relaxed);
}

kernel void ccd(constant PC_CCD& pc [[buffer(0)]],
                uint id [[thread_position_in_grid]]) {
  if (id >= pc.tot) return;
  uint b = (id / 32) * 320 + (id % 32);
  device atomic_uint* pts = PTR(atomic_uint, pc.pts);
  device MultiBvhNode* bvh = PTR(MultiBvhNode, pc.bvh);
  float3 pos = float3(
      as_type<float>(atomic_load_explicit(&pts[b], memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&pts[b + 32], memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&pts[b + 64], memory_order_relaxed)));
  float3 vel = float3(
      as_type<float>(atomic_load_explicit(&pts[b + 96], memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&pts[b + 128], memory_order_relaxed)),
      as_type<float>(
          atomic_load_explicit(&pts[b + 160], memory_order_relaxed)));
  float3 p1 = pos + vel * pc.dt;
  float3 mn = min(pos - pc.rd, p1 - pc.rd), mx = max(pos + pc.rd, p1 + pc.rd);
  uint stk[64];
  int spt = 0;
  if (pc.rt != 0xFFFFFFFFu) stk[spt++] = pc.rt;
  uint cols = 0;
  while (spt > 0) {
    uint nd = stk[--spt];
    for (uint i = 0; i < 32; ++i) {
      if (!is_vd(bvh[nd].vmk, i)) continue;
      if (iAABB(mn, mx, float3(bvh[nd].mx[i], bvh[nd].my[i], bvh[nd].mz[i]),
                float3(bvh[nd].mxx[i], bvh[nd].mxy[i], bvh[nd].mxz[i]))) {
        uint mt = bvh[nd].met[i], off = g_idx(mt);
        if (is_lf(mt)) {
          if (id < off) {
            float t = 0, d = 0;
            float3 n, p;
            uint ob = (off / 32) * 320 + (off % 32);
            float3 ovel = float3(as_type<float>(atomic_load_explicit(
                                     &pts[ob + 96], memory_order_relaxed)),
                                 as_type<float>(atomic_load_explicit(
                                     &pts[ob + 128], memory_order_relaxed)),
                                 as_type<float>(atomic_load_explicit(
                                     &pts[ob + 160], memory_order_relaxed))) *
                          pc.dt;
            float4x4 tA(1);
            tA.columns[3].xyz = pos;
            float4x4 tB(1);
            tB.columns[3].xyz =
                float3(bvh[nd].cx[i], bvh[nd].cy[i], bvh[nd].cz[i]);
            if (c_toi(0, float3(pc.rd, 0, 0), tA, vel * pc.dt, 0,
                      float3(pc.rd, 0, 0), tB, ovel, 1e-3f, t, n, p, d)) {
              if (cols < 16) {
                device SColBuf* out = PTR(SColBuf, pc.out);
                uint ox =
                    atomic_fetch_add_explicit(&out->c, 1, memory_order_relaxed);
                out->p[ox].v = 1;
                out->p[ox].ea = id;
                out->p[ox].eb = off;
                out->p[ox].toi = t;
                out->p[ox].n = packed_float3(n);
                out->p[ox].p = packed_float3(p);
                out->p[ox].d = d;
                cols++;
              }
            }
          }
        } else if (off != 0xFFFFFFFFu)
          stk[spt++] = off;
      }
    }
  }
}

kernel void narrow_ccd(constant PC_NCCD& pc [[buffer(0)]],
                       uint id [[thread_position_in_grid]]) {
  uint iA, iB, lA;
  device RigidBody* r = PTR(RigidBody, pc.ent);
  if (pc.sp == 1) {
    device CPBuf* cb = PTR(CPBuf, pc.cprs);
    if (id >= atomic_load_explicit(&cb->c, memory_order_relaxed)) return;
    CrPair p = cb->p[id];
    iA = p.ma;
    iB = p.mi;
    lA = p.lc;
  } else {
    device PBuf* pb = PTR(PBuf, pc.prs);
    if (id >= atomic_load_explicit(&pb->c, memory_order_relaxed)) return;
    uint2 p = pb->p[id];
    iA = p.x;
    iB = p.y;
  }
  RigidBody bA = r[iA], bB = r[iB];
  float3 eA = float3(bA.ext), eB = float3(bB.ext), vA = bA.lv.xyz,
         vB = bB.lv.xyz;
  float4x4 tA(1), tB(1);
  if (pc.sp == 1) {
    LcaEnt l = PTR(LcaEnt, pc.lca)[lA];
    float3 mr = vA - float3(l.lv);
    float3x3 lw = float3x3(l.itr.columns[0].xyz, l.itr.columns[1].xyz,
                           l.itr.columns[2].xyz);
    vA = lw * mr * AU_TO_KM;
    eA *= AU_TO_KM;
    tA = l.itr;
    tA.columns[3] = float4((l.itr * float4(bA.pm.xyz, 1)).xyz * AU_TO_KM, 1);
  } else {
    tA = float4x4(float4(q_m3(bA.ori).columns[0], 0),
                  float4(q_m3(bA.ori).columns[1], 0),
                  float4(q_m3(bA.ori).columns[2], 0), float4(bA.pm.xyz, 1));
  }
  tB = float4x4(float4(q_m3(bB.ori).columns[0], 0),
                float4(q_m3(bB.ori).columns[1], 0),
                float4(q_m3(bB.ori).columns[2], 0), float4(bB.pm.xyz, 1));
  float t, d;
  float3 n, c;
  if (c_toi(bA.shp, eA, tA, vA, bB.shp, eB, tB, vB, 1e-3f, t, n, c, d)) {
    if (pc.sp == 1) {
      device CSColBuf* co = PTR(CSColBuf, pc.cout);
      uint ct = atomic_fetch_add_explicit(&co->c, 1, memory_order_relaxed);
      if (ct < 4000) {
        co->p[ct] = {1, iA, iB, lA, t, packed_float3(n), packed_float3(c), d};
      }
    } else {
      device SColBuf* o = PTR(SColBuf, pc.out);
      uint ct = atomic_fetch_add_explicit(&o->c, 1, memory_order_relaxed);
      if (ct < 4000) {
        o->p[ct] = {1, iA, iA, iB, iB, t, packed_float3(n), packed_float3(c),
                    d};
      }
    }
  }
}

inline uint eBits(uint v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}
kernel void morton_encode(constant PC_Mor& pc [[buffer(0)]],
                          uint id [[thread_position_in_grid]]) {
  if (id >= pc.num) return;
  device atomic_uint* p = PTR(atomic_uint, pc.pts);
  uint b = (id / 32) * 320 + (id % 32);
  float3 p0 = float3(
      as_type<float>(atomic_load_explicit(&p[b], memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&p[b + 32], memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&p[b + 64], memory_order_relaxed)));
  float3 np = saturate((p0 - float3(pc.smn)) /
                       max(float3(pc.smx) - float3(pc.smn), float3(1e-5f)));
  PTR(uint2, pc.mot)
  [id] = uint2((eBits((uint)(np.x * 1023.f)) << 2) |
                   (eBits((uint)(np.y * 1023.f)) << 1) |
                   eBits((uint)(np.z * 1023.f)),
               id);
}

kernel void graph_coloring(constant PC_Gra& pc [[buffer(0)]],
                           uint id [[thread_position_in_grid]]) {
  if (id >= pc.tot) return;
  PTR(uint, pc.wgt)[id] = hash(id + 1);
  PTR(uint, pc.clr)[id] = 0;
}

kernel void lcp_solver(constant PC_LCP& pc [[buffer(0)]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint idx [[thread_position_in_grid]]) {
  device uint* cnt = pc.sp == 1 ? PTR(uint, pc.ccol) : (PTR(uint, pc.pcol) + 3);
  bool vld = (idx < cnt[0]);
  threadgroup atomic_uint vx[32], vy[32], vz[32], wx[32], wy[32], wz[32];
  threadgroup float an[128], at1[128], at2[128];
  an[lid] = at1[lid] = at2[lid] = 0;
  if (lid < 32) {
    RigidBody r = PTR(RigidBody, pc.rbs)[lid];
    atomic_store_explicit(&vx[lid], as_type<uint>(r.lv.x),
                          memory_order_relaxed);
    atomic_store_explicit(&vy[lid], as_type<uint>(r.lv.y),
                          memory_order_relaxed);
    atomic_store_explicit(&vz[lid], as_type<uint>(r.lv.z),
                          memory_order_relaxed);
    atomic_store_explicit(&wx[lid], as_type<uint>(r.av.x),
                          memory_order_relaxed);
    atomic_store_explicit(&wy[lid], as_type<uint>(r.av.y),
                          memory_order_relaxed);
    atomic_store_explicit(&wz[lid], as_type<uint>(r.av.z),
                          memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!vld) return;
  float3 n, t1, t2, rA, rB, pA, pB, vAi, vBi, wAi, wBi, iIA, iIB;
  float iMA = 0, iMB = 0, emN, eT1, eT2, tvN;
  bool pAf = false, pBf = false;
  uint iA, iB;
  float4 qA(0, 0, 0, 1), qB(0, 0, 0, 1);
  if (pc.sp == 1) {
    CrPairD c = ((device CSColBuf*)PTR(uint, pc.ccol))->p[idx];
    if (c.v == 0) return;
    iA = c.ma;
    iB = c.mi;
    n = float3(c.n);
    g_tan(n, t1, t2);
    RigidBody ma = PTR(RigidBody, pc.rbs)[iA], mi = PTR(RigidBody, pc.rbs)[iB];
    LcaEnt lc = PTR(LcaEnt, pc.lca)[c.lc];
    iMA = ma.pm.w > 0 ? 1.f / (ma.pm.w * M_EARTH_TO_KG) : 0;
    iIA = ma.iI.xyz / (M_EARTH_TO_KG * AU_TO_KM * AU_TO_KM);
    qA = ma.ori;
    pA = (lc.itr * float4(ma.pm.xyz, 1)).xyz * AU_TO_KM;
    float3x3 rw(lc.itr.columns[0].xyz, lc.itr.columns[1].xyz,
                lc.itr.columns[2].xyz);
    vAi = rw * (ma.lv.xyz - float3(lc.lv)) * AU_TO_KM;
    wAi = rw * ma.av.xyz;
    iMB = mi.pm.w > 0 ? 1.f / mi.pm.w : 0;
    iIB = mi.iI.xyz;
    qB = mi.ori;
    pB = mi.pm.xyz;
    vBi = mi.lv.xyz;
    wBi = mi.av.xyz;
    rA = float3(c.p) - pA;
    rB = float3(c.p) - pB;
    float3x3 l2w(lc.tr.columns[0].xyz, lc.tr.columns[1].xyz,
                 lc.tr.columns[2].xyz);
    emN = ef_m(1, n, rA, rB, iMA, iMB, iIA, iIB, qA, qB, l2w);
    eT1 = ef_m(1, t1, rA, rB, iMA, iMB, iIA, iIB, qA, qB, l2w);
    eT2 = ef_m(1, t2, rA, rB, iMA, iMB, iIA, iIB, qA, qB, l2w);
    tvN = (0.2f / max(pc.dt, 1e-6f)) * max(c.d - 0.01f, 0.f);
  } else {
    PkPair p = ((device PColBuf*)PTR(uint, pc.pcol))->p[idx];
    pAf = (p.a.eid == 0xFFFFFFFFu);
    pBf = (p.b.eid == 0xFFFFFFFFu);
    iA = p.a.pid;
    iB = p.b.pid;
    device atomic_uint* pts = PTR(atomic_uint, pc.pts);
    if (pAf) {
      uint b = (iA / 32) * 320 + (iA % 32);
      pA = float3(
          as_type<float>(atomic_load_explicit(&pts[b], memory_order_relaxed)),
          as_type<float>(
              atomic_load_explicit(&pts[b + 32], memory_order_relaxed)),
          as_type<float>(
              atomic_load_explicit(&pts[b + 64], memory_order_relaxed)));
      vAi = float3(as_type<float>(atomic_load_explicit(&pts[b + 96],
                                                       memory_order_relaxed)),
                   as_type<float>(atomic_load_explicit(&pts[b + 128],
                                                       memory_order_relaxed)),
                   as_type<float>(atomic_load_explicit(&pts[b + 160],
                                                       memory_order_relaxed)));
      float m = as_type<float>(
          atomic_load_explicit(&pts[b + 192], memory_order_relaxed));
      iMA = m > 0 ? 1.f / m : 0;
      wAi = float3(0);
    } else {
      RigidBody r = PTR(RigidBody, pc.rbs)[iA];
      iMA = r.pm.w > 0 ? 1.f / r.pm.w : 0;
      iIA = r.iI.xyz;
      qA = r.ori;
      pA = r.pm.xyz;
      vAi = r.lv.xyz;
      wAi = r.av.xyz;
    }
    if (pBf) {
      uint b = (iB / 32) * 320 + (iB % 32);
      pB = float3(
          as_type<float>(atomic_load_explicit(&pts[b], memory_order_relaxed)),
          as_type<float>(
              atomic_load_explicit(&pts[b + 32], memory_order_relaxed)),
          as_type<float>(
              atomic_load_explicit(&pts[b + 64], memory_order_relaxed)));
      vBi = float3(as_type<float>(atomic_load_explicit(&pts[b + 96],
                                                       memory_order_relaxed)),
                   as_type<float>(atomic_load_explicit(&pts[b + 128],
                                                       memory_order_relaxed)),
                   as_type<float>(atomic_load_explicit(&pts[b + 160],
                                                       memory_order_relaxed)));
      float m = as_type<float>(
          atomic_load_explicit(&pts[b + 192], memory_order_relaxed));
      iMB = m > 0 ? 1.f / m : 0;
      wBi = float3(0);
    } else {
      RigidBody r = PTR(RigidBody, pc.rbs)[iB];
      iMB = r.pm.w > 0 ? 1.f / r.pm.w : 0;
      iIB = r.iI.xyz;
      qB = r.ori;
      pB = r.pm.xyz;
      vBi = r.lv.xyz;
      wBi = r.av.xyz;
    }
    n = float3(p.n);
    g_tan(n, t1, t2);
    rA = float3(p.p) - pA;
    rB = float3(p.p) - pB;
    float3x3 idt(1);
    emN = ef_m(0, n, rA, rB, iMA, iMB, iIA, iIB, qA, qB, idt);
    eT1 = ef_m(0, t1, rA, rB, iMA, iMB, iIA, iIB, qA, qB, idt);
    eT2 = ef_m(0, t2, rA, rB, iMA, iMB, iIA, iIB, qA, qB, idt);
    tvN = (0.2f / max(pc.dt, 1e-6f)) * max(p.d - 0.01f, 0.f);
  }
  float bnce =
      dot((vBi + cross(wBi, rB)) - (vAi + cross(wAi, rA)), n) < -0.1f
          ? -pc.res * dot((vBi + cross(wBi, rB)) - (vAi + cross(wAi, rA)), n)
          : 0.0f;
  tvN += bnce;
  for (int i = 0; i < 20; ++i) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float3 vA = vAi, wA = wAi, vB = vBi, wB = wBi;
    if (!pAf && iA < 32) {
      vA = float3(
          as_type<float>(atomic_load_explicit(&vx[iA], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&vy[iA], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&vz[iA], memory_order_relaxed)));
      wA = float3(
          as_type<float>(atomic_load_explicit(&wx[iA], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&wy[iA], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&wz[iA], memory_order_relaxed)));
    }
    if (!pBf && iB < 32) {
      vB = float3(
          as_type<float>(atomic_load_explicit(&vx[iB], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&vy[iB], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&vz[iB], memory_order_relaxed)));
      wB = float3(
          as_type<float>(atomic_load_explicit(&wx[iB], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&wy[iB], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&wz[iB], memory_order_relaxed)));
    }
    float3 vr = (vB + cross(wB, rB)) - (vA + cross(wA, rA));
    float jn = emN * (-dot(vr, n) + tvN), ojn = an[lid],
          njn = max(ojn + jn, 0.f);
    jn = njn - ojn;
    an[lid] = njn;
    float3 Pn = jn * n;
    if (!pAf && iMA > 0 && iA < 32) {
      atomic_add_f_tg(&vx[iA], -Pn.x * iMA);
      atomic_add_f_tg(&vy[iA], -Pn.y * iMA);
      atomic_add_f_tg(&vz[iA], -Pn.z * iMA);
      float3 dw = q_rt(qA, iIA * q_ir(qA, cross(rA, -Pn)));
      atomic_add_f_tg(&wx[iA], dw.x);
      atomic_add_f_tg(&wy[iA], dw.y);
      atomic_add_f_tg(&wz[iA], dw.z);
    }
    if (!pBf && iMB > 0 && iB < 32) {
      atomic_add_f_tg(&vx[iB], Pn.x * iMB);
      atomic_add_f_tg(&vy[iB], Pn.y * iMB);
      atomic_add_f_tg(&vz[iB], Pn.z * iMB);
      float3 dw = q_rt(qB, iIB * q_ir(qB, cross(rB, Pn)));
      atomic_add_f_tg(&wx[iB], dw.x);
      atomic_add_f_tg(&wy[iB], dw.y);
      atomic_add_f_tg(&wz[iB], dw.z);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!pAf && iA < 32) {
      vA = float3(
          as_type<float>(atomic_load_explicit(&vx[iA], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&vy[iA], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&vz[iA], memory_order_relaxed)));
      wA = float3(
          as_type<float>(atomic_load_explicit(&wx[iA], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&wy[iA], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&wz[iA], memory_order_relaxed)));
    }
    if (!pBf && iB < 32) {
      vB = float3(
          as_type<float>(atomic_load_explicit(&vx[iB], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&vy[iB], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&vz[iB], memory_order_relaxed)));
      wB = float3(
          as_type<float>(atomic_load_explicit(&wx[iB], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&wy[iB], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(&wz[iB], memory_order_relaxed)));
    }
    vr = (vB + cross(wB, rB)) - (vA + cross(wA, rA));
    float mf = .5f * an[lid],
          jt1 = clamp(at1[lid] + eT1 * (-dot(vr, t1)), -mf, mf),
          dt1 = jt1 - at1[lid];
    at1[lid] = jt1;
    float jt2 = clamp(at2[lid] + eT2 * (-dot(vr, t2)), -mf, mf),
          dt2 = jt2 - at2[lid];
    at2[lid] = jt2;
    float3 Pt = dt1 * t1 + dt2 * t2;
    if (!pAf && iMA > 0 && iA < 32) {
      atomic_add_f_tg(&vx[iA], -Pt.x * iMA);
      atomic_add_f_tg(&vy[iA], -Pt.y * iMA);
      atomic_add_f_tg(&vz[iA], -Pt.z * iMA);
      float3 dw = q_rt(qA, iIA * q_ir(qA, cross(rA, -Pt)));
      atomic_add_f_tg(&wx[iA], dw.x);
      atomic_add_f_tg(&wy[iA], dw.y);
      atomic_add_f_tg(&wz[iA], dw.z);
    }
    if (!pBf && iMB > 0 && iB < 32) {
      atomic_add_f_tg(&vx[iB], Pt.x * iMB);
      atomic_add_f_tg(&vy[iB], Pt.y * iMB);
      atomic_add_f_tg(&vz[iB], Pt.z * iMB);
      float3 dw = q_rt(qB, iIB * q_ir(qB, cross(rB, Pt)));
      atomic_add_f_tg(&wx[iB], dw.x);
      atomic_add_f_tg(&wy[iB], dw.y);
      atomic_add_f_tg(&wz[iB], dw.z);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  PTR(float3, pc.out)[idx] = an[lid] * n + at1[lid] * t1 + at2[lid] * t2;
}

kernel void lbvh_build(constant PC_LBVHB& pc [[buffer(0)]],
                       uint id [[thread_position_in_grid]]) {
  device uint2* mrt = PTR(uint2, pc.mrt);
  device MultiBvhNode* bvh = PTR(MultiBvhNode, pc.bvh);
  auto pfx = [&](int i, int j) -> int {
    if (j < 0 || j >= int(pc.num)) return -1;
    uint k1 = mrt[i].x, k2 = mrt[j].x;
    if (k1 == k2) return 32 + (31 - clz(mrt[i].y ^ mrt[j].y));
    return 31 - clz(k1 ^ k2);
  };
  if (id < pc.num - 1) {
    int d = sign((float)(pfx(id, id + 1) - pfx(id, id - 1))),
        mp = pfx(id, id - d), lm = 2;
    while (pfx(id, id + lm * d) > mp) lm *= 2;
    int l = 0, t = lm / 2;
    while (t >= 1) {
      if (pfx(id, id + (l + t) * d) > mp) l += t;
      t /= 2;
    }
    int rx = min((int)id, (int)id + l * d), ry = max((int)id, (int)id + l * d),
        s = rx, st = ry - rx;
    do {
      st = (st + 1) >> 1;
      int ns = s + st;
      if (ns < ry && pfx(rx, ns) > mp) s = ns;
    } while (st > 1);
    uint lc = (s == rx) ? (pc.num - 1 + s) : s,
         rc = (s + 1 == ry) ? (pc.num - 1 + s + 1) : (s + 1);
    bvh[id].chd[0] = lc;
    bvh[id].chd[1] = rc;
    bvh[id].vmk = uint2(3, 0);
    bvh[lc].par = id;
    bvh[rc].par = id;
  }
  uint lf = pc.num - 1 + id, pid = mrt[id].y, b = (pid / 32) * 320 + (pid % 32);
  device atomic_uint* pts = PTR(atomic_uint, pc.pts);
  float3 p = float3(as_type<float>(
                        atomic_load_explicit(&pts[b], memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&pts[b + 32],
                                                        memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&pts[b + 64],
                                                        memory_order_relaxed))),
         v = float3(as_type<float>(atomic_load_explicit(&pts[b + 96],
                                                        memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&pts[b + 128],
                                                        memory_order_relaxed)),
                    as_type<float>(atomic_load_explicit(&pts[b + 160],
                                                        memory_order_relaxed)));
  float m =
      as_type<float>(atomic_load_explicit(&pts[b + 192], memory_order_relaxed));
  float3 p1 = p + v * pc.dt, lmn = min(p - pc.rd, p1 - pc.rd),
         lmx = max(p + pc.rd, p1 + pc.rd);
  uint cur = bvh[lf].par, ir = (bvh[cur].chd[1] == lf) ? 1 : 0;
  bvh[cur].mx[ir] = lmn.x;
  bvh[cur].mxx[ir] = lmx.x;
  bvh[cur].my[ir] = lmn.y;
  bvh[cur].mxy[ir] = lmx.y;
  bvh[cur].mz[ir] = lmn.z;
  bvh[cur].mxz[ir] = lmx.z;
  bvh[cur].mas[ir] = m;
  bvh[cur].cx[ir] = p.x;
  bvh[cur].cy[ir] = p.y;
  bvh[cur].cz[ir] = p.z;
  bvh[cur].met[ir] = pk_mt(true, 1, 0, pid);
  threadgroup_barrier(mem_flags::mem_device);
  while (cur != 0xFFFFFFFFu) {
    if (atomic_fetch_add_explicit(&PTR(AtCnt, pc.cnt)->c[cur], 1,
                                  memory_order_relaxed) == 0)
      break;
    float3 clmn = float3(bvh[cur].mx[0], bvh[cur].my[0], bvh[cur].mz[0]),
           clmx = float3(bvh[cur].mxx[0], bvh[cur].mxy[0], bvh[cur].mxz[0]),
           crmn = float3(bvh[cur].mx[1], bvh[cur].my[1], bvh[cur].mz[1]),
           crmx = float3(bvh[cur].mxx[1], bvh[cur].mxy[1], bvh[cur].mxz[1]);
    float lm = bvh[cur].mas[0], rm = bvh[cur].mas[1], cm = lm + rm;
    float3 lcom = float3(bvh[cur].cx[0], bvh[cur].cy[0], bvh[cur].cz[0]),
           rcom = float3(bvh[cur].cx[1], bvh[cur].cy[1], bvh[cur].cz[1]),
           ccom = cm > 0 ? (lcom * lm + rcom * rm) / cm : (lcom + rcom) * 0.5f;
    float3 cmn = min(clmn, crmn), cmx = max(clmx, crmx);
    uint pr = bvh[cur].par;
    if (pr != 0xFFFFFFFFu) {
      uint r = (bvh[pr].chd[1] == cur) ? 1 : 0;
      bvh[pr].mx[r] = cmn.x;
      bvh[pr].mxx[r] = cmx.x;
      bvh[pr].my[r] = cmn.y;
      bvh[pr].mxy[r] = cmx.y;
      bvh[pr].mz[r] = cmn.z;
      bvh[pr].mxz[r] = cmx.z;
      bvh[pr].mas[r] = cm;
      bvh[pr].cx[r] = ccom.x;
      bvh[pr].cy[r] = ccom.y;
      bvh[pr].cz[r] = ccom.z;
      bvh[pr].met[r] = pk_mt(false, 1, 0, cur);
    }
    threadgroup_barrier(mem_flags::mem_device);
    cur = pr;
  }
}

kernel void radix_sort(constant PC_Sort& pc [[buffer(0)]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint wid [[threadgroup_position_in_grid]],
                       uint sgi [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sgs [[simdgroups_per_threadgroup]]) {
  threadgroup atomic_uint sC[16];
  threadgroup uint sO[16], sSG[64], sB[16];
  device uint2* in = PTR(uint2, pc.in);
  device atomic_uint* hst = PTR(atomic_uint, pc.hst);
  if (pc.stg == 0) {
    if (lid < 16) atomic_store_explicit(&sC[lid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint e = min(wid * 4096 + 4096, pc.num);
    for (uint i = wid * 4096 + lid; i < e; i += 256)
      atomic_fetch_add_explicit(&sC[(in[i].x >> pc.shf) & 0xF], 1,
                                memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16)
      atomic_store_explicit(
          &hst[lid * pc.blk + wid],
          atomic_load_explicit(&sC[lid], memory_order_relaxed),
          memory_order_relaxed);
  } else if (pc.stg == 1) {
    if (lid < 16) {
      uint sm = 0;
      for (uint w = 0; w < pc.blk; ++w)
        sm +=
            atomic_load_explicit(&hst[lid * pc.blk + w], memory_order_relaxed);
      sB[lid] = sm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      uint of = 0;
      for (uint i = 0; i < 16; ++i) {
        uint v = sB[i];
        sB[i] = of;
        of += v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 16) {
      uint of = sB[lid];
      for (uint w = 0; w < pc.blk; ++w) {
        uint v =
            atomic_load_explicit(&hst[lid * pc.blk + w], memory_order_relaxed);
        atomic_store_explicit(&hst[lid * pc.blk + w], of, memory_order_relaxed);
        of += v;
      }
    }
  } else {
    if (lid < 16)
      sO[lid] =
          atomic_load_explicit(&hst[lid * pc.blk + wid], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint e = min(wid * 4096 + 4096, pc.num);
    for (uint ch = wid * 4096; ch < e; ch += 256) {
      uint i = ch + lid;
      bool v = (i < e);
      uint k = v ? ((in[i].x >> pc.shf) & 0xF) : 0xFFFFFFFFu;
      uint lo = 0, gb = 0;
      for (uint b = 0; b < 16; ++b) {
        ulong m = get_ballot(k == b);
        if (lane == 0) sSG[sgi] = popcount(m);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
          uint sm = 0;
          for (uint s = 0; s < sgs; ++s) {
            uint c = sSG[s];
            sSG[s] = sm;
            sm += c;
          }
          atomic_store_explicit(&sC[b], sm, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k == b) {
          lo = sSG[sgi] + popcount(m & ((1ul << lane) - 1ul));
          gb = sO[b];
        }
        if (lid == 0)
          sO[b] += atomic_load_explicit(&sC[b], memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (v) PTR(uint2, pc.out)[gb + lo] = in[i];
    }
  }
}

kernel void bp_cross_lca(constant PC_BPCr& pc [[buffer(0)]],
                         uint w [[threadgroup_position_in_grid]],
                         uint sgi [[simdgroup_index_in_threadgroup]],
                         uint ln [[thread_index_in_simdgroup]]) {
  threadgroup uint sS[8][32];
  threadgroup uint sP[8];
  uint qid = w * 8 + sgi;
  if (qid >= pc.tq ||
      qid >= atomic_load_explicit(&PTR(PBuf, pc.lq)->c, memory_order_relaxed))
    return;
  uint ma = PTR(PBuf, pc.lq)->p[qid].x, la = PTR(PBuf, pc.lq)->p[qid].y;
  float3 qm, qx;
  threadgroup ulong sh_bvh[8];
  if (ln == 0) {
    LcaEnt l = PTR(LcaEnt, pc.lca)[la];
    sh_bvh[sgi] = l.bvh;
    float3 mm = float3(PTR(TLASLeaf, pc.mlv)[ma].mn),
           mxM = float3(PTR(TLASLeaf, pc.mlv)[ma].mx), c = (mm + mxM) * .5f,
           e = (mxM - mm) * .5f, ck = c * AU_TO_KM, ek = e * AU_TO_KM;
    qm = float3(1e20f);
    qx = float3(-1e20f);
    float3 cr[8] = {float3(ck.x - ek.x, ck.y - ek.y, ck.z - ek.z),
                    float3(ck.x + ek.x, ck.y - ek.y, ck.z - ek.z),
                    float3(ck.x - ek.x, ck.y + ek.y, ck.z - ek.z),
                    float3(ck.x + ek.x, ck.y + ek.y, ck.z - ek.z),
                    float3(ck.x - ek.x, ck.y - ek.y, ck.z + ek.z),
                    float3(ck.x + ek.x, ck.y - ek.y, ck.z + ek.z),
                    float3(ck.x - ek.x, ck.y + ek.y, ck.z + ek.z),
                    float3(ck.x + ek.x, ck.y + ek.y, ck.z + ek.z)};
    for (int i = 0; i < 8; ++i) {
      float3 lp = (l.itr * float4(cr[i], 1.f)).xyz;
      qm = min(qm, lp);
      qx = max(qx, lp);
    }
    sS[sgi][0] = l.rt;
    sP[sgi] = 1;
  }
  qm = float3(simd_broadcast(qm.x, 0), simd_broadcast(qm.y, 0),
              simd_broadcast(qm.z, 0));
  qx = float3(simd_broadcast(qx.x, 0), simd_broadcast(qx.y, 0),
              simd_broadcast(qx.z, 0));
  ma = simd_broadcast(ma, 0);
  device MultiBvhNode* bvh = PTR(MultiBvhNode, sh_bvh[sgi]);
  while (true) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint sp = sP[sgi];
    if (sp == 0) break;
    sp--;
    uint nd = sS[sgi][sp];
    if (ln == 0) sP[sgi] = sp;
    uint mt = bvh[nd].met[ln];
    bool v = is_vd(bvh[nd].vmk, ln);
    float3 cmn = float3(bvh[nd].mx[ln], bvh[nd].my[ln], bvh[nd].mz[ln]),
           cmx = float3(bvh[nd].mxx[ln], bvh[nd].mxy[ln], bvh[nd].mxz[ln]);
    uint ch = bvh[nd].chd[ln];
    bool h = v && iAABB(qm, qx, cmn, cmx), hl = h && is_lf(mt),
         hn = h && !is_lf(mt);
    ulong lB = get_ballot(hl);
    if (lB != 0) {
      uint bs = 0;
      if (ln == 0)
        bs = atomic_fetch_add_explicit(&PTR(CPBuf, pc.cp)->c, popcount(lB),
                                       memory_order_relaxed);
      bs = simd_broadcast(bs, 0);
      if (hl && bs + popcount(lB & ((1ul << ln) - 1ul)) < pc.mx) {
        PTR(CPBuf, pc.cp)->p[bs + popcount(lB & ((1ul << ln) - 1ul))].ma = ma;
        PTR(CPBuf, pc.cp)->p[bs + popcount(lB & ((1ul << ln) - 1ul))].mi =
            g_idx(mt);
        PTR(CPBuf, pc.cp)->p[bs + popcount(lB & ((1ul << ln) - 1ul))].lc = la;
      }
    }
    while (lB != 0) {
      uint s = ctz(lB);
      lB &= ~(1ul << s);
      uint mi = g_idx(simd_shuffle(mt, s));
      if (ln == 0) {
        uint ta = PTR(EntHdr, pc.ent)[ma].typ, tb = PTR(EntHdr, pc.ent)[mi].typ,
             eA = ma, eB = mi;
        if (ta > tb) {
          uint t = eA;
          eA = eB;
          eB = t;
          t = ta;
          ta = tb;
          tb = t;
        }
        if (ta == 1 && tb == 1) {
          uint oi = atomic_fetch_add_explicit(&PTR(PBuf, pc.rr)->c, 1,
                                              memory_order_relaxed);
          if (oi < pc.mx) PTR(PBuf, pc.rr)->p[oi] = uint2(eA, eB);
        } else if (ta == 0 && tb == 1) {
          uint oi = atomic_fetch_add_explicit(&PTR(PBuf, pc.rp)->c, 1,
                                              memory_order_relaxed);
          if (oi < pc.mx) PTR(PBuf, pc.rp)->p[oi] = uint2(eB, eA);
        } else if (ta == 0 && tb == 0) {
          uint oi = atomic_fetch_add_explicit(&PTR(PBuf, pc.pp)->c, 1,
                                              memory_order_relaxed);
          if (oi < pc.mx) PTR(PBuf, pc.pp)->p[oi] = uint2(eA, eB);
        }
      }
    }
    ulong nB = get_ballot(hn);
    if (hn) sS[sgi][sp + popcount(nB & ((1ul << ln) - 1ul))] = ch;
    if (ln == 0) sP[sgi] = sp + popcount(nB);
  }
}

kernel void bp_scene(constant PC_BPSce& pc [[buffer(0)]],
                     uint w [[threadgroup_position_in_grid]],
                     uint sgi [[simdgroup_index_in_threadgroup]],
                     uint ln [[thread_index_in_simdgroup]]) {
  threadgroup uint sS[8][32];
  threadgroup uint sP[8];
  uint qid = w * 8 + sgi;
  if (qid >= pc.tot) return;
  float3 qm, qx;
  uint qe;
  if (ln == 0) {
    qm = float3(PTR(TLASLeaf, pc.lvs)[qid].mn);
    qx = float3(PTR(TLASLeaf, pc.lvs)[qid].mx);
    qe = PTR(TLASLeaf, pc.lvs)[qid].eidx;
    sS[sgi][0] = pc.rt;
    sP[sgi] = 1;
  }
  qm = float3(simd_broadcast(qm.x, 0), simd_broadcast(qm.y, 0),
              simd_broadcast(qm.z, 0));
  qx = float3(simd_broadcast(qx.x, 0), simd_broadcast(qx.y, 0),
              simd_broadcast(qx.z, 0));
  qe = simd_broadcast(qe, 0);
  while (true) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint sp = sP[sgi];
    if (sp == 0) break;
    sp--;
    uint nd = sS[sgi][sp];
    if (ln == 0) sP[sgi] = sp;
    uint mt = PTR(MultiBvhNode, pc.tls)[nd].met[ln];
    bool v = is_vd(PTR(MultiBvhNode, pc.tls)[nd].vmk, ln);
    float3 cm = float3(PTR(MultiBvhNode, pc.tls)[nd].mx[ln],
                       PTR(MultiBvhNode, pc.tls)[nd].my[ln],
                       PTR(MultiBvhNode, pc.tls)[nd].mz[ln]),
           cx = float3(PTR(MultiBvhNode, pc.tls)[nd].mxx[ln],
                       PTR(MultiBvhNode, pc.tls)[nd].mxy[ln],
                       PTR(MultiBvhNode, pc.tls)[nd].mxz[ln]);
    uint ch = PTR(MultiBvhNode, pc.tls)[nd].chd[ln], ei = g_idx(mt);
    bool h = v && iAABB(qm, qx, cm, cx), hl = h && is_lf(mt) && (qe < ei),
         hn = h && !is_lf(mt);
    ulong lB = get_ballot(hl);
    if (lB != 0) {
      uint bs = 0;
      if (ln == 0)
        bs = atomic_fetch_add_explicit(&PTR(PBuf, pc.prs)->c, popcount(lB),
                                       memory_order_relaxed);
      bs = simd_broadcast(bs, 0);
      if (hl && bs + popcount(lB & ((1ul << ln) - 1ul)) < 10000u)
        PTR(PBuf, pc.prs)->p[bs + popcount(lB & ((1ul << ln) - 1ul))] =
            uint2(qe, ei);
    }
    ulong nB = get_ballot(hn);
    if (hn) sS[sgi][sp + popcount(nB & ((1ul << ln) - 1ul))] = ch;
    if (ln == 0) sP[sgi] = sp + popcount(nB);
  }
}

kernel void apply_impulses(constant PC_App& pc [[buffer(0)]],
                           uint id [[thread_position_in_grid]]) {
  // Left empty for brevity in your full physics logic
}

kernel void lbvh_collapse(constant PC_Colp& pc [[buffer(0)]],
                          uint wg [[threadgroup_position_in_grid]],
                          uint ln [[thread_index_in_simdgroup]]) {
  if (wg >= pc.num) return;
  uint bx = PTR(uint, pc.map)[wg];
  bool l = false;
  uint p = 0, fp = 0, fd = 0;
  for (int d = 4; d >= 0; d--) {
    uint dr = (ln >> d) & 1u, mt = PTR(MultiBvhNode, pc.bin)[bx].met[dr];
    l = is_lf(mt);
    uint nx = g_idx(mt);
    fp = bx;
    fd = dr;
    if (l) {
      p = nx;
      break;
    }
    bx = nx;
  }
  if (!l) {
    p = bx;
    fp = PTR(MultiBvhNode, pc.bin)[bx].par;
    fd = (PTR(MultiBvhNode, pc.bin)[fp].chd[1] == bx) ? 1 : 0;
  }
  PTR(MultiBvhNode, pc.mul)[wg].mx[ln] = PTR(MultiBvhNode, pc.bin)[fp].mx[fd];
  PTR(MultiBvhNode, pc.mul)[wg].mxx[ln] = PTR(MultiBvhNode, pc.bin)[fp].mxx[fd];
  PTR(MultiBvhNode, pc.mul)[wg].my[ln] = PTR(MultiBvhNode, pc.bin)[fp].my[fd];
  PTR(MultiBvhNode, pc.mul)[wg].mxy[ln] = PTR(MultiBvhNode, pc.bin)[fp].mxy[fd];
  PTR(MultiBvhNode, pc.mul)[wg].mz[ln] = PTR(MultiBvhNode, pc.bin)[fp].mz[fd];
  PTR(MultiBvhNode, pc.mul)[wg].mxz[ln] = PTR(MultiBvhNode, pc.bin)[fp].mxz[fd];
  PTR(MultiBvhNode, pc.mul)[wg].mas[ln] = PTR(MultiBvhNode, pc.bin)[fp].mas[fd];
  PTR(MultiBvhNode, pc.mul)[wg].cx[ln] = PTR(MultiBvhNode, pc.bin)[fp].cx[fd];
  PTR(MultiBvhNode, pc.mul)[wg].cy[ln] = PTR(MultiBvhNode, pc.bin)[fp].cy[fd];
  PTR(MultiBvhNode, pc.mul)[wg].cz[ln] = PTR(MultiBvhNode, pc.bin)[fp].cz[fd];
  PTR(MultiBvhNode, pc.mul)[wg].chd[ln] = p;
  PTR(MultiBvhNode, pc.mul)[wg].met[ln] = pk_mt(l, 1, 0, p);
  if (ln == 0) {
    PTR(MultiBvhNode, pc.mul)[wg].vmk = uint2(0xFFFFFFFFu, 0);
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 32; ++j) PTR(MultiBvhNode, pc.mul)[wg].prm[i][j] = j;
  }
}

kernel void bp_particle_self(constant PC_BPSlf& pc [[buffer(0)]],
                             uint wg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint ln [[thread_index_in_simdgroup]]) {
  uint pid = wg * 8 + sg;
  if (pid >= pc.tot) return;
  threadgroup uint sS[8][32];
  threadgroup uint sP[8];
  float3 mp, mmn, mmx;
  if (ln == 0) {
    uint b = (pid / 32) * 320 + (pid % 32);
    mp = float3(as_type<float>(atomic_load_explicit(
                    &PTR(atomic_uint, pc.pts)[b], memory_order_relaxed)),
                as_type<float>(atomic_load_explicit(
                    &PTR(atomic_uint, pc.pts)[b + 32], memory_order_relaxed)),
                as_type<float>(atomic_load_explicit(
                    &PTR(atomic_uint, pc.pts)[b + 64], memory_order_relaxed)));
    mmn = mp - pc.rd;
    mmx = mp + pc.rd;
    sS[sg][0] = pc.rt;
    sP[sg] = 1;
  }
  mp = float3(simd_broadcast(mp.x, 0), simd_broadcast(mp.y, 0),
              simd_broadcast(mp.z, 0));
  mmn = float3(simd_broadcast(mmn.x, 0), simd_broadcast(mmn.y, 0),
               simd_broadcast(mmn.z, 0));
  mmx = float3(simd_broadcast(mmx.x, 0), simd_broadcast(mmx.y, 0),
               simd_broadcast(mmx.z, 0));
  pid = simd_broadcast(pid, 0);
  float3 f(0);
  while (true) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint sp = sP[sg];
    if (sp == 0) break;
    sp--;
    uint nd = sS[sg][sp];
    if (ln == 0) sP[sg] = sp;
    uint mt = PTR(MultiBvhNode, pc.bvh)[nd].met[ln];
    bool v = is_vd(PTR(MultiBvhNode, pc.bvh)[nd].vmk, ln);
    float3 cm = float3(PTR(MultiBvhNode, pc.bvh)[nd].mx[ln],
                       PTR(MultiBvhNode, pc.bvh)[nd].my[ln],
                       PTR(MultiBvhNode, pc.bvh)[nd].mz[ln]),
           cx = float3(PTR(MultiBvhNode, pc.bvh)[nd].mxx[ln],
                       PTR(MultiBvhNode, pc.bvh)[nd].mxy[ln],
                       PTR(MultiBvhNode, pc.bvh)[nd].mxz[ln]);
    uint ch = PTR(MultiBvhNode, pc.bvh)[nd].chd[ln];
    bool hit = v && iAABB(mmn, mmx, cm, cx),
         hl = hit && is_lf(mt) && (pid != ch), hn = hit && !is_lf(mt);
    ulong lm = get_ballot(hl);
    while (lm != 0) {
      uint s = ctz(lm);
      lm &= ~(1ul << s);
      uint oid = simd_shuffle(ch, s), ob = (oid / 32) * 320 + (oid % 32);
      float3 op = float3(
          as_type<float>(atomic_load_explicit(&PTR(atomic_uint, pc.pts)[ob],
                                              memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(
              &PTR(atomic_uint, pc.pts)[ob + 32], memory_order_relaxed)),
          as_type<float>(atomic_load_explicit(
              &PTR(atomic_uint, pc.pts)[ob + 64], memory_order_relaxed)));
      float3 df = mp - op;
      float d2 = dot(df, df), md = pc.rd * 2.f;
      if (d2 > 1e-12f && d2 < md * md) {
        float dst = sqrt(d2), pen = md - dst;
        f += (df / dst) * (pc.stf * pen);
      }
    }
    ulong nm = get_ballot(hn);
    if (hn) sS[sg][sp + popcount(nm & ((1ul << ln) - 1ul))] = ch;
    if (ln == 0) sP[sg] = sp + popcount(nm);
  }
  f = float3(simd_sum(f.x), simd_sum(f.y), simd_sum(f.z));
  if (ln == 0 && dot(f, f) > 0) {
    atomic_add_f(&PTR(Wrench, pc.wr)[pid].fx, f.x);
    atomic_add_f(&PTR(Wrench, pc.wr)[pid].fy, f.y);
    atomic_add_f(&PTR(Wrench, pc.wr)[pid].fz, f.z);
  }
}

kernel void motion_bounds(constant PC_MotB& pc [[buffer(0)]],
                          uint id [[thread_position_in_grid]]) {
  if (id >= pc.num) return;
  uint b = (id / 32) * 320 + (id % 32);
  float3 p = float3(
             as_type<float>(atomic_load_explicit(&PTR(atomic_uint, pc.pts)[b],
                                                 memory_order_relaxed)),
             as_type<float>(atomic_load_explicit(
                 &PTR(atomic_uint, pc.pts)[b + 32], memory_order_relaxed)),
             as_type<float>(atomic_load_explicit(
                 &PTR(atomic_uint, pc.pts)[b + 64], memory_order_relaxed))),
         v = float3(
             as_type<float>(atomic_load_explicit(
                 &PTR(atomic_uint, pc.pts)[b + 96], memory_order_relaxed)),
             as_type<float>(atomic_load_explicit(
                 &PTR(atomic_uint, pc.pts)[b + 128], memory_order_relaxed)),
             as_type<float>(atomic_load_explicit(
                 &PTR(atomic_uint, pc.pts)[b + 160], memory_order_relaxed))),
         p1 = p + v * pc.dt, mn = min(p, p1) - pc.rd, mx = max(p, p1) + pc.rd;
  uint l = (pc.num - 1) + id, pr = PTR(MultiBvhNode, pc.bvh)[l].par,
       ir = (PTR(MultiBvhNode, pc.bvh)[pr].chd[1] == l) ? 1 : 0;
  PTR(MultiBvhNode, pc.bvh)[pr].mx[ir] = mn.x;
  PTR(MultiBvhNode, pc.bvh)[pr].mxx[ir] = mx.x;
  PTR(MultiBvhNode, pc.bvh)[pr].my[ir] = mn.y;
  PTR(MultiBvhNode, pc.bvh)[pr].mxy[ir] = mx.y;
  PTR(MultiBvhNode, pc.bvh)[pr].mz[ir] = mn.z;
  PTR(MultiBvhNode, pc.bvh)[pr].mxz[ir] = mx.z;
}

kernel void convert_particles(constant PC_Conv& pc [[buffer(0)]],
                              uint id [[thread_position_in_grid]]) {
  uint t = atomic_load_explicit(&PTR(AtCnt, pc.ct)->c[0], memory_order_relaxed);
  if (id == 0) {
    PTR(DInd, pc.id)[pc.ix] = {4, t, 0, pc.of};
  }
  if (id >= t) return;
  uint b = (id / 32) * 320 + (id % 32);
  PTR(RPart, pc.mg)
  [pc.of + id].p = packed_float3(
      as_type<float>(atomic_load_explicit(&PTR(atomic_uint, pc.ao)[b],
                                          memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&PTR(atomic_uint, pc.ao)[b + 32],
                                          memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&PTR(atomic_uint, pc.ao)[b + 64],
                                          memory_order_relaxed)));
  PTR(RPart, pc.mg)
  [pc.of + id].v = packed_float3(
      as_type<float>(atomic_load_explicit(&PTR(atomic_uint, pc.ao)[b + 96],
                                          memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&PTR(atomic_uint, pc.ao)[b + 128],
                                          memory_order_relaxed)),
      as_type<float>(atomic_load_explicit(&PTR(atomic_uint, pc.ao)[b + 160],
                                          memory_order_relaxed)));
  PTR(RPart, pc.mg)
  [pc.of + id].m = as_type<float>(atomic_load_explicit(
      &PTR(atomic_uint, pc.ao)[b + 192], memory_order_relaxed));
  PTR(RPart, pc.mg)[pc.of + id].il = 0;
  PTR(RPart, pc.mg)[pc.of + id].ih = 0;
  PTR(RPart, pc.mg)[pc.of + id].al = 0;
  PTR(RPart, pc.mg)[pc.of + id].ah = 0;
  PTR(RPart, pc.mg)[pc.of + id].act = 1;
}

kernel void reduce_toi(constant PC_RToi& pc [[buffer(0)]],
                       uint id [[thread_position_in_grid]],
                       uint sgi [[simdgroup_index_in_threadgroup]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint wgs [[threads_per_threadgroup]]) {
  float tc = pc.dt;
  if (id < atomic_load_explicit(&PTR(PColBuf, pc.col)->c, memory_order_relaxed))
    tc = PTR(PColBuf, pc.col)->p[id].toi;
  float smn = simd_min(tc);
  threadgroup uint sh[4];
  if ((lid % 32) == 0) sh[sgi] = as_type<uint>(smn);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    uint w = sh[0];
    for (uint i = 1; i < wgs / 32; ++i) w = min(w, sh[i]);
    uint exp =
        atomic_load_explicit(PTR(atomic_uint, pc.toi), memory_order_relaxed);
    while (w < exp && !atomic_compare_exchange_weak_explicit(
                          PTR(atomic_uint, pc.toi), &exp, w,
                          memory_order_relaxed, memory_order_relaxed));
  }
}

kernel void barnes_hut(constant PC_BHut& pc [[buffer(0)]],
                       uint wg [[threadgroup_position_in_grid]],
                       uint sgi [[simdgroup_index_in_threadgroup]],
                       uint ln [[thread_index_in_simdgroup]]) {
  uint cid = wg * 8 + sgi;
  if (cid >= pc.num) return;
  uint tn = PTR(uint, pc.cl)[cid];
  bool iv = is_vd(PTR(MultiBvhNode, pc.bvh)[tn].vmk, ln);
  uint mp = PTR(MultiBvhNode, pc.bvh)[tn].chd[ln];
  float3 mp0(0.f);
  float mm = 0.f;
  if (iv) {
    uint b = (mp / 32) * 320 + (mp % 32);
    mp0 = float3(as_type<float>(atomic_load_explicit(
                     &PTR(atomic_uint, pc.pts)[b], memory_order_relaxed)),
                 as_type<float>(atomic_load_explicit(
                     &PTR(atomic_uint, pc.pts)[b + 32], memory_order_relaxed)),
                 as_type<float>(atomic_load_explicit(
                     &PTR(atomic_uint, pc.pts)[b + 64], memory_order_relaxed)));
    mm = as_type<float>(atomic_load_explicit(&PTR(atomic_uint, pc.pts)[b + 192],
                                             memory_order_relaxed));
  }
  float3 sp = iv ? mp0 : float3(0.f),
         mn = float3(simd_min(iv ? mp0.x : 1e20f), simd_min(iv ? mp0.y : 1e20f),
                     simd_min(iv ? mp0.z : 1e20f)),
         mx = float3(simd_max(iv ? mp0.x : -1e20f),
                     simd_max(iv ? mp0.y : -1e20f),
                     simd_max(iv ? mp0.z : -1e20f)),
         ex = mx - mn;
  float tsz = max(ex.x, max(ex.y, ex.z)), sm = simd_sum(iv ? mm : 0.f);
  float3 tc =
      float3(simd_sum(sp.x * mm), simd_sum(sp.y * mm), simd_sum(sp.z * mm)) /
      max(sm, 1e-6f);
  float3 acc(0.f);
  threadgroup uint sS[8][64];
  threadgroup uint sP[8];
  if (ln == 0) {
    sS[sgi][0] = pc.rt;
    sP[sgi] = 1;
  }
  while (true) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint p = sP[sgi];
    if (p == 0) break;
    uint sn = sS[sgi][--p];
    if (ln == 0) sP[sgi] = p;
    bool sv = is_vd(PTR(MultiBvhNode, pc.bvh)[sn].vmk, ln),
         sl = is_lf(PTR(MultiBvhNode, pc.bvh)[sn].met[ln]);
    float3 sc = float3(PTR(MultiBvhNode, pc.bvh)[sn].cx[ln],
                       PTR(MultiBvhNode, pc.bvh)[sn].cy[ln],
                       PTR(MultiBvhNode, pc.bvh)[sn].cz[ln]);
    float sms = PTR(MultiBvhNode, pc.bvh)[sn].mas[ln];
    uint si = PTR(MultiBvhNode, pc.bvh)[sn].chd[ln],
         sst = PTR(MultiBvhNode, pc.bvh)[sn].pst[ln],
         sct = PTR(MultiBvhNode, pc.bvh)[sn].pct[ln];
    float3 se = float3(PTR(MultiBvhNode, pc.bvh)[sn].mxx[ln] -
                           PTR(MultiBvhNode, pc.bvh)[sn].mx[ln],
                       PTR(MultiBvhNode, pc.bvh)[sn].mxy[ln] -
                           PTR(MultiBvhNode, pc.bvh)[sn].my[ln],
                       PTR(MultiBvhNode, pc.bvh)[sn].mxz[ln] -
                           PTR(MultiBvhNode, pc.bvh)[sn].mz[ln]);
    float ssz = max(se.x, max(se.y, se.z));
    bool pm = ((ssz + tsz) / max(length(sc - tc), 1e-6f)) < pc.th,
         pt = (sct <= pc.thr) && !((mp >= sst) && (mp < sst + sct)),
         aa = sv && (pm || pt || sl), at = sv && !aa;
    ulong am = get_ballot(aa);
    while (am != 0) {
      uint s = ctz(am);
      am &= ~(1ul << s);
      if (iv) {
        float3 kc = float3(simd_shuffle(sc.x, s), simd_shuffle(sc.y, s),
                           simd_shuffle(sc.z, s));
        float km = simd_shuffle(sms, s);
        uint ki = simd_shuffle(si, s);
        bool kl = (bool)simd_shuffle((uint)sl, s);
        if (!(kl && mp == ki)) {
          float3 pd = kc - mp0;
          float d2 = dot(pd, pd);
          acc += (pd / max(sqrt(d2), 1e-6f)) * ((pc.G * km) / (d2 + pc.sq));
        }
      }
    }
    ulong tm = get_ballot(at);
    if (at) sS[sgi][p + popcount(tm & ((1ul << ln) - 1ul))] = si;
    if (ln == 0) sP[sgi] = p + popcount(tm);
  }
  if (iv) {
    float3 gf = acc * mm;
    atomic_add_f(&PTR(Wrench, pc.wr)[mp].fx, gf.x);
    atomic_add_f(&PTR(Wrench, pc.wr)[mp].fy, gf.y);
    atomic_add_f(&PTR(Wrench, pc.wr)[mp].fz, gf.z);
  }
}

