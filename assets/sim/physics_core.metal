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



// --- TRANSLATED KERNELS --- 

// --- msl_integrate_particles_p1_p2.txt ---
// @assets/sim/integrate_particles_p1_p2.comp
//
// Particle Velocity-Verlet Predictor — Phase 1 & 2
// ─────────────────────────────────────────────────
// Frame-start invariant: AOSOA slots 7/8/9 hold F(x_n) from the previous frame.
//
//   v_{n+½} = v_n + (dt/2) · M⁻¹ · F(x_n)     [half-kick]
//   x_{n+1} = x_n + dt · v_{n+½}               [full position leap]
//
// After writing, CLEARS slots 7/8/9 to 0 so the unified force-generation pass
// (barnes_hut, bp_particle_self, narrow-phase) can safely atomicAdd into them.
// The half-kick velocity v_{n+½} is stored temporarily in slots 3/4/5 for
// integrate_particles_p4_5 to complete the VV corrector step.
//
// Target: SPIR-V 1.4 · Vulkan 1.1 · flexible across all hardware subgroup sizes.





struct PushConstants_integrate_particles_p1_p2 {
    ParticleData particles;
    float dt;
    uint total_particles;
};


[[kernel]]
void integrate_particles_p1_p2(constant PushConstants_integrate_particles_p1_p2& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint gid = thread_position_in_grid.x;
    if (gid >= pc.total_particles) return;

    uint base = (gid / SUBGROUP_SIZE) * (10u * SUBGROUP_SIZE) + (gid % SUBGROUP_SIZE);

    float mass = P_READ(pc.particles, base + 6u * SUBGROUP_SIZE);
    if (mass <= 0.0) return;

    float inv_m = 1.0 / mass;
    float half_dt = 0.5 * pc.dt;

    float3 v_n = float3(P_READ(pc.particles, base + 3u * SUBGROUP_SIZE), P_READ(pc.particles, base + 4u * SUBGROUP_SIZE), P_READ(pc.particles, base + 5u * SUBGROUP_SIZE));
    float3 f_n = float3(P_READ(pc.particles, base + 7u * SUBGROUP_SIZE), P_READ(pc.particles, base + 8u * SUBGROUP_SIZE), P_READ(pc.particles, base + 9u * SUBGROUP_SIZE));

    float3 v_half = v_n + f_n * inv_m * half_dt;
    float3 pos_n = float3(P_READ(pc.particles, base + 0u * SUBGROUP_SIZE), P_READ(pc.particles, base + 1u * SUBGROUP_SIZE), P_READ(pc.particles, base + 2u * SUBGROUP_SIZE));
    float3 pos_next = pos_n + v_half * pc.dt;

    P_WRITE(pc.particles, base + 0u * SUBGROUP_SIZE, pos_next.x);
    P_WRITE(pc.particles, base + 1u * SUBGROUP_SIZE, pos_next.y);
    P_WRITE(pc.particles, base + 2u * SUBGROUP_SIZE, pos_next.z);

    P_WRITE(pc.particles, base + 3u * SUBGROUP_SIZE, v_half.x);
    P_WRITE(pc.particles, base + 4u * SUBGROUP_SIZE, v_half.y);
    P_WRITE(pc.particles, base + 5u * SUBGROUP_SIZE, v_half.z);

    P_WRITE(pc.particles, base + 7u * SUBGROUP_SIZE, 0.0);
    P_WRITE(pc.particles, base + 8u * SUBGROUP_SIZE, 0.0);
    P_WRITE(pc.particles, base + 9u * SUBGROUP_SIZE, 0.0);
}


// --- msl_integrate_bodies_p3.txt ---
#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.msl"
#include "imex_math.msl"

struct PushConstants {
    device RigidBody* rigid_bodies;
    device Wrench* wrenches;
    device ForceEmitter* emitters;
    float dt;
    uint n_bodies;
    uint n_iterations;
    uint num_emitters;
};

[[kernel]]
void integrate_bodies_p3(
    constant PushConstants& pc [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= pc.n_bodies) return;

    device RigidBody& body = pc.rigid_bodies[id];
    float mass = body.position_mass.w;
    float inv_m = (mass > 0.0) ? 1.0 / mass : 0.0;
    
    float3 I_inv = body.inertia_tensor_inv.xyz;
    float3 I_fwd = float3(
        (I_inv.x > 1e-14) ? 1.0 / I_inv.x : 0.0,
        (I_inv.y > 1e-14) ? 1.0 / I_inv.y : 0.0,
        (I_inv.z > 1e-14) ? 1.0 / I_inv.z : 0.0
    );

    float3 pos_n = body.position_mass.xyz;
    float4 q_n = body.orientation;
    float3 v_n = body.linear_vel_drag.xyz;
    float3 w_n = body.angular_vel_drag.xyz;

    uint w_idx = body.wrench_idx;
    device Wrench& wrench = pc.wrenches[w_idx];

    float3 f_n = float3(as_type<float>(wrench.force_x), as_type<float>(wrench.force_y), as_type<float>(wrench.force_z));
    float3 t_n = float3(as_type<float>(wrench.torque_x), as_type<float>(wrench.torque_y), as_type<float>(wrench.torque_z));

    for (uint e = 0; e < pc.num_emitters; ++e) {
        device ForceEmitter& emitter = pc.emitters[e];
        float3 em_pos = emitter.position;
        float em_mu = emitter.mu;
        float3 em_norm = emitter.normal;
        uint em_type = emitter.type_id;
        float em_trunc = emitter.trunc_distance;
        float em_scale = emitter.scale_factor;

        if (em_type == 0) {
            float3 r = em_pos - pos_n;
            float s_dist_sq = dot(r, r) * em_scale * em_scale;
            if (s_dist_sq > 1e-6) {
                float s_dist = sqrt(s_dist_sq);
                float s_dist3 = s_dist_sq * s_dist;
                float s_dist5 = s_dist3 * s_dist_sq;
                float softening = 1.0 - exp(-s_dist5);
                float force_mag = (em_mu * mass * softening) / s_dist_sq;
                f_n += normalize(r) * force_mag;
            }
        } else if (em_type == 1) {
            float dist = dot(pos_n - em_pos, em_norm);
            if (dist >= 0.0 && dist <= em_trunc) {
                f_n += em_norm * em_mu;
            }
        }
    }

    float half_dt = 0.5 * pc.dt;
    float3 a_lin = f_n * inv_m;
    float3 v_mid = v_n + half_dt * a_lin;
    float3 pos_next = pos_n + pc.dt * v_mid;
    float3 v_next = v_n + pc.dt * a_lin;

    float3 t_local = quat_rotate_inv(q_n, t_n);
    float3 w_n_local = quat_rotate_inv(q_n, w_n);
    float3 w_mid_local = w_n_local;

    for (uint iter = 0; iter < pc.n_iterations; ++iter) {
        float3 gyro = cross(w_mid_local, I_fwd * w_mid_local);
        float3 a_ang = I_inv * (t_local - gyro);
        w_mid_local = w_n_local + half_dt * a_ang;
    }

    float3 w_next_local = 2.0 * w_mid_local - w_n_local;
    float3 w_next = quat_rotate(q_n, w_next_local);
    float3 w_mid_world = quat_rotate(q_n, w_mid_local);
    float4 omega_pure = float4(w_mid_world, 0.0);
    float4 q_next = normalize(q_n + half_dt * quat_mult(omega_pure, q_n));

    body.position_mass = float4(pos_next, mass);
    body.orientation = q_next;
    body.linear_vel_drag = float4(v_next, body.linear_vel_drag.w);
    body.angular_vel_drag = float4(w_next, body.angular_vel_drag.w);

    wrench.force_x = 0;
    wrench.force_y = 0;
    wrench.force_z = 0;
    wrench.torque_x = 0;
    wrench.torque_y = 0;
    wrench.torque_z = 0;
}


// --- msl_rb_force_assign.txt ---
#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.msl"
#include "imex_math.msl"

struct PushConstants {
    device RigidBody* rigid_bodies;
    device Wrench* wrenches;
    uint n_bodies;
    uint _pad;
};

inline void atomic_add_float(device uint* addr, float val) {
    device atomic_uint* atomic_addr = reinterpret_cast<device atomic_uint*>(addr);
    uint old_val = atomic_load_explicit(atomic_addr, memory_order_relaxed);
    uint assumed_val, new_val;
    do {
        assumed_val = old_val;
        new_val = as_type<uint>(as_type<float>(assumed_val) + val);
    } while (!atomic_compare_exchange_weak_explicit(atomic_addr, &old_val, new_val, memory_order_relaxed, memory_order_relaxed));
}

[[kernel]]
void rb_force_assign(
    constant PushConstants& pc [[push_constant]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint threads_per_simdgroup [[thread_execution_width]]
) {
    uint body_id = threadgroup_position_in_grid.x;
    if (body_id >= pc.n_bodies) return;

    uint local_id = thread_position_in_threadgroup.x;
    
    device RigidBody& body = pc.rigid_bodies[body_id];
    uint leaf_start = body.leaf_start_idx;
    uint leaf_count = body.leaf_count;
    uint com_wrench = body.wrench_idx;

    float3 acc_f = float3(0.0);
    float3 acc_t = float3(0.0);

    for (uint i = local_id; i < leaf_count; i += 128u) {
        device Wrench& lw = pc.wrenches[leaf_start + i];
        acc_f += float3(as_type<float>(lw.force_x), as_type<float>(lw.force_y), as_type<float>(lw.force_z));
        acc_t += float3(as_type<float>(lw.torque_x), as_type<float>(lw.torque_y), as_type<float>(lw.torque_z));
    }

    acc_f.x = simd_sum(acc_f.x);
    acc_f.y = simd_sum(acc_f.y);
    acc_f.z = simd_sum(acc_f.z);
    
    acc_t.x = simd_sum(acc_t.x);
    acc_t.y = simd_sum(acc_t.y);
    acc_t.z = simd_sum(acc_t.z);

    threadgroup float3 sh_f[32]; // Max 32 subgroups if thread_execution_width is 4
    threadgroup float3 sh_t[32];

    if (simd_lane_id == 0u) {
        sh_f[simd_group_id] = acc_f;
        sh_t[simd_group_id] = acc_t;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0u) {
        float3 total_f = float3(0.0);
        float3 total_t = float3(0.0);
        uint subgroups_per_wg = 128u / threads_per_simdgroup;
        for (uint s = 0u; s < subgroups_per_wg; ++s) {
            total_f += sh_f[s];
            total_t += sh_t[s];
        }
        
        device Wrench& cw = pc.wrenches[com_wrench];
        atomic_add_float(&cw.force_x, total_f.x);
        atomic_add_float(&cw.force_y, total_f.y);
        atomic_add_float(&cw.force_z, total_f.z);
        atomic_add_float(&cw.torque_x, total_t.x);
        atomic_add_float(&cw.torque_y, total_t.y);
        atomic_add_float(&cw.torque_z, total_t.z);
    }
}


// --- msl_integrate_particles_p4_5.txt ---
// @assets/sim/integrate_particles_p4_5.comp
//
// Particle Velocity-Verlet Corrector — Phase 4 & 5
// ─────────────────────────────────────────────────
// Invariant entering this pass:
//   • AOSOA slots 3/4/5 hold v_{n+½}  (stored by integrate_particles_p1_p2)
//   • AOSOA slots 7/8/9 hold F(x_{n+1}) (written by force generators after p3)
//
//   v_{n+1} = v_{n+½} + (dt/2) · M⁻¹ · F(x_{n+1})    [VV corrector]
//
// The force buffer is intentionally NOT cleared — F(x_{n+1}) persists as
// F(x_n) for the NEXT frame's integrate_particles_p1_p2 pass.
//
// Thread 0 additionally advances the emulated 64-bit engine clock:
//   global_time_us += dt_us    (uvec2 carry-propagating addition from imex_math.glsl)
//
// Target: SPIR-V 1.4 · Vulkan 1.1 · flexible across all hardware subgroup sizes.

struct PushConstants_integrate_particles_p4_5 {
    ParticleData particles;
    ClockBuffer  clock;
    float        dt;
    uint         total_particles;
    uint         dt_us_lo;
    uint         dt_us_hi;
    uint         current_time_lo;
    uint         current_time_hi;
};

[[kernel]]
void integrate_particles_p4_5(constant PushConstants_integrate_particles_p4_5& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint gid = thread_position_in_grid.x;

    // ── Thread 0: advance the 64-bit engine clock exactly once per frame ─────
    // This must happen regardless of particle count so the clock always ticks.
    if (gid == 0u) {
        uint2 t_n  = uint2(pc.current_time_lo, pc.current_time_hi);
        uint2 dt_u = uint2(pc.dt_us_lo,        pc.dt_us_hi);
        uint2 res;
        res.x = t_n.x + dt_u.x;
        uint carry = (res.x < t_n.x) ? 1u : 0u;
        res.y = t_n.y + dt_u.y + carry;
        pc.clock.global_time_us = res;
    }

    if (gid >= pc.total_particles) return;

    uint block = gid / SUBGROUP_SIZE;
    uint lane  = gid % SUBGROUP_SIZE;
    uint base  = block * (10u * SUBGROUP_SIZE) + lane;

    // ── Skip inactive / massless particles ────────────────────────────────
    float mass = P_READ(pc.particles, base + 6u * SUBGROUP_SIZE);
    if (mass <= 0.0) return;

    float inv_m   = 1.0 / mass;
    float half_dt = 0.5 * pc.dt;

    // ── Load v_{n+½} (written by p1_p2) ──────────────────────────────────
    float3 v_half = float3(
        P_READ(pc.particles, base + 3u * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 4u * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 5u * SUBGROUP_SIZE)
    );

    // ── Load F(x_{n+1}) (written by force generators after p3) ───────────
    float3 f_next = float3(
        P_READ(pc.particles, base + 7u * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 8u * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 9u * SUBGROUP_SIZE)
    );

    // ── VV Corrector ─────────────────────────────────────────────────────
    float3 v_next = v_half + f_next * inv_m * half_dt;

    // Write v_{n+1} back — force buffer stays intact for next frame
    P_WRITE(pc.particles, base + 3u * SUBGROUP_SIZE, v_next.x);
    P_WRITE(pc.particles, base + 4u * SUBGROUP_SIZE, v_next.y);
    P_WRITE(pc.particles, base + 5u * SUBGROUP_SIZE, v_next.z);
}

// --- msl_bp_clear.txt ---
#include <metal_stdlib>
using namespace metal;

#include "../debug_utils.h"
#include "../bvh_utils.h"

struct PushConstants {
    device uint* raw_scene_pairs;
    device uint* out_rb_rb;
    device uint* out_rb_ps;
    device uint* out_rb_lca;
    device uint* internal_pairs;
};

[[kernel]]
void bp_clear(
    constant PushConstants& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    *pc.raw_scene_pairs = 0u;
    *pc.out_rb_rb = 0u;
    *pc.out_rb_ps = 0u;
    *pc.out_rb_lca = 0u;
    *pc.internal_pairs = 0u;
}

// --- msl_bp_bounds_gen.txt ---
// @assets/sim/bp_bounds_gen.comp

struct PushConstants_bp_bounds_gen {
    device RigidBody* scene_entities;
    device uint* particles;
    device TLASLeaf* tlas_leaves;
    uint2    dt_us;
    uint     total_entities;
    uint     num_rigid_bodies;
    float    particle_radius;
};

[[kernel]]
void bp_bounds_gen(
    constant PushConstants_bp_bounds_gen& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint id = thread_position_in_grid.x;
    if (id >= pc.total_entities) return;

    float dt = dt_to_seconds(pc.dt_us);
    float3 center, extents, vel;
    uint shape_type;
    uint64_t bda;

    if (id < pc.num_rigid_bodies) {
        device RigidBody& body = pc.scene_entities[id];
        center = body.position_mass.xyz;
        extents = body.shape_extents;
        vel = body.linear_vel_drag.xyz;
        shape_type = body.shape_type;
        bda = (uint64_t)&body;
    } else {
        uint particle_system_idx = id - pc.num_rigid_bodies;
        // The bounds of a particle system should ideally be computed over all its particles.
        // For now, since particles are grouped into entities (32 particles per entity),
        // we approximate the bounds using the center of the first particle in the group.
        uint base = particle_system_idx * (10 * SUBGROUP_SIZE);
        
        center = float3(
            as_type<float>(pc.particles[base + 0]),
            as_type<float>(pc.particles[base + 1 * SUBGROUP_SIZE]),
            as_type<float>(pc.particles[base + 2 * SUBGROUP_SIZE])
        );
        extents = float3(pc.particle_radius * 16.0); // Rough approximation for 32 particles
        vel = float3(
            as_type<float>(pc.particles[base + 3 * SUBGROUP_SIZE]),
            as_type<float>(pc.particles[base + 4 * SUBGROUP_SIZE]),
            as_type<float>(pc.particles[base + 5 * SUBGROUP_SIZE])
        );
        shape_type = BVH_SHAPE_SPHERE;
        bda = (uint64_t)&pc.particles[base]; // Address of this chunk of 32 particles
    }

    float3 static_min = center - extents;
    float3 static_max = center + extents;
    float3 sweep = vel * dt;

    device TLASLeaf& leaf = pc.tlas_leaves[id];
    leaf.min_bound = min(static_min, static_min + sweep);
    leaf.max_bound = max(static_max, static_max + sweep);
    leaf.entity_idx = id;
    leaf.metadata = bvh_pack_metadata(true, BVH_FRAME_MACRO, shape_type, id);
    leaf.bda = bda;
}

// --- msl_bp_scene.txt ---
#include <metal_stdlib>
#include "../debug_utils.metal"


using namespace metal;

struct PushConstants {
    device MultiBvhNode* tlas_bvh;
    device TLASLeaf* query_leaves;
    device PairBuffer* overlapping_pairs;
    uint tlas_root_index;
    uint total_queries;
};

[[kernel]]
void bp_scene(
    constant PushConstants& pc [[buffer(0)]],
    uint3 gl_WorkGroupID [[threadgroup_position_in_grid]],
    uint gl_SubgroupID [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    uint query_idx = gl_WorkGroupID.x * 8 + gl_SubgroupID;
    if (query_idx >= pc.total_queries) return;

    float3 my_min, my_max;
    uint my_ent_id;

    threadgroup uint shared_stacks[8][32];
    threadgroup uint shared_stack_ptrs[8];

    if (lane_id == 0) {
        my_min = pc.query_leaves[query_idx].min_bound;
        my_max = pc.query_leaves[query_idx].max_bound;
        my_ent_id = pc.query_leaves[query_idx].entity_idx;

        shared_stacks[gl_SubgroupID][0] = pc.tlas_root_index;
        shared_stack_ptrs[gl_SubgroupID] = 1;
    }

    my_min.x = simd_broadcast(my_min.x, 0);
    my_min.y = simd_broadcast(my_min.y, 0);
    my_min.z = simd_broadcast(my_min.z, 0);
    my_max.x = simd_broadcast(my_max.x, 0);
    my_max.y = simd_broadcast(my_max.y, 0);
    my_max.z = simd_broadcast(my_max.z, 0);
    my_ent_id = simd_broadcast(my_ent_id, 0);

    while (true) {
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint stack_ptr = shared_stack_ptrs[gl_SubgroupID];
        if (stack_ptr == 0) break;

        stack_ptr--;
        uint node_idx = shared_stacks[gl_SubgroupID][stack_ptr];
        if (lane_id == 0) shared_stack_ptrs[gl_SubgroupID] = stack_ptr;

        uint meta = pc.tlas_bvh[node_idx].metadata[lane_id];
        uint2 valid_mask = pc.tlas_bvh[node_idx].valid_mask;
        bool valid = bvh_node_is_valid(valid_mask, lane_id);

        float3 c_min = float3(
            pc.tlas_bvh[node_idx].min_x[lane_id],
            pc.tlas_bvh[node_idx].min_y[lane_id],
            pc.tlas_bvh[node_idx].min_z[lane_id]
        );
        float3 c_max = float3(
            pc.tlas_bvh[node_idx].max_x[lane_id],
            pc.tlas_bvh[node_idx].max_y[lane_id],
            pc.tlas_bvh[node_idx].max_z[lane_id]
        );
        uint child_payload = pc.tlas_bvh[node_idx].child_indices[lane_id];

        uint entity_id = bvh_get_index(meta);

        bool hit = valid && intersectAABB(my_min, my_max, c_min, c_max);
        bool is_leaf = bvh_is_leaf(meta);

        bool hit_leaf = hit && is_leaf && (my_ent_id < entity_id);
        bool hit_node = hit && !is_leaf;

        uint leaf_count = simd_sum(hit_leaf ? 1 : 0);
        uint leaf_offset = simd_prefix_exclusive_sum(hit_leaf ? 1 : 0);

        if (leaf_count > 0) {
            uint base_idx = 0;
            if (lane_id == 0) {
                base_idx = atomic_fetch_add_explicit(&pc.overlapping_pairs->count, leaf_count, memory_order_relaxed);
            }
            base_idx = simd_broadcast(base_idx, 0);

            if (hit_leaf && base_idx + leaf_offset < 10000u) {
                pc.overlapping_pairs->pairs[base_idx + leaf_offset] = uint2(my_ent_id, entity_id);
            }
        }

        uint node_count = simd_sum(hit_node ? 1 : 0);
        uint push_offset = simd_prefix_exclusive_sum(hit_node ? 1 : 0);

        if (hit_node) shared_stacks[gl_SubgroupID][stack_ptr + push_offset] = child_payload;
        if (lane_id == 0) shared_stack_ptrs[gl_SubgroupID] = stack_ptr + node_count;
    }
}


// --- msl_bp_classify.txt ---
#include <metal_stdlib>
using namespace metal;



#ifndef TYPE_PARTICLE_SYSTEM
#define TYPE_PARTICLE_SYSTEM 0
#define TYPE_RIGID_BODY      1
#define TYPE_MICRO_LCA       2
#endif

struct PairBuffer {
    atomic_uint count;
    uint pad;
    uint2 pairs[1];
};

struct TLASLeaf {
    packed_float3 min_bound;
    uint entity_idx;
    packed_float3 max_bound;
    uint metadata;
    uint64_t bda;
    uint pad[2];
};

struct LeafBuffer {
    TLASLeaf leaves[1];
};

struct EntityHeader {
    uint ty;
    uint pad[3];
};

struct PushConstants {
    uint64_t raw_pairs;
    uint2 out_rb_rb;
    uint2 out_rb_ps;
    uint2 out_ps_ps;
    uint64_t tlas_leaves;
    uint max_pairs;
    uint num_rigid_bodies;
};

[[kernel]]
void bp_classify(
    constant PushConstants& pc [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    device PairBuffer* raw_pairs = (device PairBuffer*)(pc.raw_pairs);
    uint count = atomic_load_explicit(&raw_pairs->count, memory_order_relaxed);
    if (id >= count) return;

    uint2 pair = raw_pairs->pairs[id];
    uint ent_A = pair.x;
    uint ent_B = pair.y;

    device LeafBuffer* tlas_leaves = (device LeafBuffer*)(pc.tlas_leaves);
    uint64_t bda_A = tlas_leaves->leaves[ent_A].bda;
    uint64_t bda_B = tlas_leaves->leaves[ent_B].bda;

    device EntityHeader* header_A = (device EntityHeader*)(bda_A);
    device EntityHeader* header_B = (device EntityHeader*)(bda_B);

    uint type_A = header_A->ty;
    uint type_B = header_B->ty;

    if (type_A > type_B) {
        uint temp = ent_A; ent_A = ent_B; ent_B = temp;
        temp = type_A; type_A = type_B; type_B = temp;
    }

    if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_PARTICLE_SYSTEM) {
        if (pc.out_ps_ps.x != 0 || pc.out_ps_ps.y != 0) {
            uint64_t ptr_val = ((uint64_t)pc.out_ps_ps.y << 32) | pc.out_ps_ps.x;
            device PairBuffer* buf = (device PairBuffer*)(ptr_val);
            uint out_idx = atomic_fetch_add_explicit(&buf->count, 1, memory_order_relaxed);
            if (out_idx < pc.max_pairs) {
                buf->pairs[out_idx] = uint2(ent_A, ent_B);
            }
        }
    } else if (type_A == TYPE_RIGID_BODY && type_B == TYPE_PARTICLE_SYSTEM) {
        if (pc.out_rb_ps.x != 0 || pc.out_rb_ps.y != 0) {
            uint64_t ptr_val = ((uint64_t)pc.out_rb_ps.y << 32) | pc.out_rb_ps.x;
            device PairBuffer* buf = (device PairBuffer*)(ptr_val);
            uint out_idx = atomic_fetch_add_explicit(&buf->count, 1, memory_order_relaxed);
            if (out_idx < pc.max_pairs) {
                buf->pairs[out_idx] = uint2(ent_A, ent_B);
            }
        }
    } else if (type_A == TYPE_RIGID_BODY && type_B == TYPE_RIGID_BODY) {
        if (pc.out_rb_rb.x != 0 || pc.out_rb_rb.y != 0) {
            uint64_t ptr_val = ((uint64_t)pc.out_rb_rb.y << 32) | pc.out_rb_rb.x;
            device PairBuffer* buf = (device PairBuffer*)(ptr_val);
            uint out_idx = atomic_fetch_add_explicit(&buf->count, 1, memory_order_relaxed);
            if (out_idx < pc.max_pairs) {
                buf->pairs[out_idx] = uint2(ent_A, ent_B);
            }
        }
    }
}

// --- msl_bp_cross_lca.txt ---
#include <metal_stdlib>
using namespace metal;

#include "../debug_utils.metal"


struct PushConstants {
    device LcaEntity* lca_entities;
    device TLASLeaf* macro_leaves;
    device EntityHeader* entity_headers;
    device PairBuffer* lca_query_pairs;
    device PairBuffer* out_rb_rb;
    device PairBuffer* out_rb_ps;
    device PairBuffer* out_ps_ps;
    device CrossPairBuffer* out_cross_pairs;
    device MultiBvhNode* tlas_bvh;
    uint total_queries;
    uint max_pairs;
};

#define AU_TO_KM 149597870.7

void transform_aabb_macro_to_micro(float3 lca_center, float lca_scale, float3 macro_center_au, float3 macro_extents_au, thread float3& out_min, thread float3& out_max) {
    float3 center_km = macro_center_au * AU_TO_KM;
    float3 extents_km = macro_extents_au * AU_TO_KM;

    float3 corners[8] = {
        float3(center_km.x - extents_km.x, center_km.y - extents_km.y, center_km.z - extents_km.z),
        float3(center_km.x + extents_km.x, center_km.y - extents_km.y, center_km.z - extents_km.z),
        float3(center_km.x - extents_km.x, center_km.y + extents_km.y, center_km.z - extents_km.z),
        float3(center_km.x + extents_km.x, center_km.y + extents_km.y, center_km.z - extents_km.z),
        float3(center_km.x - extents_km.x, center_km.y - extents_km.y, center_km.z + extents_km.z),
        float3(center_km.x + extents_km.x, center_km.y - extents_km.y, center_km.z + extents_km.z),
        float3(center_km.x - extents_km.x, center_km.y + extents_km.y, center_km.z + extents_km.z),
        float3(center_km.x + extents_km.x, center_km.y + extents_km.y, center_km.z + extents_km.z)
    };
    out_min = float3(1e20); out_max = float3(-1e20);
    for (int i = 0; i < 8; i++) {
        float3 local_p = (corners[i] - lca_center) / lca_scale;
        out_min = min(out_min, local_p);
        out_max = max(out_max, local_p);
    }
}

[[kernel]]
void bp_cross_lca(
    constant PushConstants& pc [[buffer(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]
) {
    uint lane_id = thread_index_in_simdgroup;
    uint subgroup_id = thread_index_in_threadgroup / threads_per_simdgroup;
    uint query_idx = threadgroup_position_in_grid.x * (256 / threads_per_simdgroup) + subgroup_id;

    if (query_idx >= pc.total_queries || query_idx >= pc.lca_query_pairs->count) return;

    uint2 query = pc.lca_query_pairs->pairs[query_idx];
    uint macro_ent_id = query.x;
    uint lca_ent_id = query.y;
    float3 query_min, query_max;

    threadgroup uint shared_stacks[8][32]; // Max subgroups is 256/32 = 8
    threadgroup uint shared_stack_ptrs[8];
    threadgroup device MultiBvhNode* shared_lca_bvh_addr[8];

    if (lane_id == 0) {
        LcaEntity l_ent = pc.lca_entities[lca_ent_id];
        shared_lca_bvh_addr[subgroup_id] = pc.tlas_bvh;
        
        TLASLeaf macro_leaf = pc.macro_leaves[macro_ent_id];
        float3 macro_min = macro_leaf.min_bound;
        float3 macro_max = macro_leaf.max_bound;

        float3 center_au = (macro_min + macro_max) * 0.5;
        float3 extents_au = (macro_max - macro_min) * 0.5;

        transform_aabb_macro_to_micro(l_ent.center_pos, l_ent.scale, center_au, extents_au, query_min, query_max);

        shared_stacks[subgroup_id][0] = l_ent.bvh_root_index;
        shared_stack_ptrs[subgroup_id] = 1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    query_min = simd_broadcast(query_min, 0);
    query_max = simd_broadcast(query_max, 0);
    macro_ent_id = simd_broadcast(macro_ent_id, 0);

    device MultiBvhNode* tlas = shared_lca_bvh_addr[subgroup_id];

    while (true) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint stack_ptr = shared_stack_ptrs[subgroup_id];
        if (stack_ptr == 0) break;

        stack_ptr--;
        uint node_idx = shared_stacks[subgroup_id][stack_ptr];
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr;

        uint meta = tlas[node_idx].metadata[lane_id];
        bool valid = bvh_node_is_valid(tlas[node_idx].valid_mask, lane_id);
        
        float3 c_min = float3(tlas[node_idx].min_x[lane_id], tlas[node_idx].min_y[lane_id], tlas[node_idx].min_z[lane_id]);
        float3 c_max = float3(tlas[node_idx].max_x[lane_id], tlas[node_idx].max_y[lane_id], tlas[node_idx].max_z[lane_id]);
        uint child_payload = tlas[node_idx].child_indices[lane_id];

        bool hit = valid && intersectAABB(query_min, query_max, c_min, c_max);
        bool is_leaf = bvh_is_leaf(meta);

        bool hit_leaf = hit && is_leaf;
        bool hit_node = hit && !is_leaf;

        uint leaf_count = simd_sum(hit_leaf ? 1 : 0);
        uint leaf_offset = simd_prefix_exclusive_sum(hit_leaf ? 1 : 0);

        if (leaf_count > 0) {
            uint base_idx = 0;
            if (lane_id == 0) {
                base_idx = atomic_fetch_add_explicit((device atomic_uint*)&pc.out_cross_pairs->count, leaf_count, memory_order_relaxed);
            }
            base_idx = simd_broadcast(base_idx, 0);

            if (hit_leaf && (base_idx + leaf_offset) < pc.max_pairs) {
                pc.out_cross_pairs->pairs[base_idx + leaf_offset].macro_id = macro_ent_id;
                pc.out_cross_pairs->pairs[base_idx + leaf_offset].micro_id = bvh_get_index(meta);
                pc.out_cross_pairs->pairs[base_idx + leaf_offset].lca_id = lca_ent_id;
            }
        }

        uint subgroup_size = threads_per_simdgroup;
        for (uint src_lane = 0; src_lane < subgroup_size; src_lane++) {
            bool src_hit_leaf = simd_broadcast(hit_leaf, src_lane);
            if (src_hit_leaf) {
                uint micro_ent_id = bvh_get_index(simd_broadcast(meta, src_lane));

                if (lane_id == 0) {
                    uint type_A = pc.entity_headers[macro_ent_id].ty;
                    uint type_B = pc.entity_headers[micro_ent_id].ty;
                    uint ent_A = macro_ent_id;
                    uint ent_B = micro_ent_id;

                    if (type_A > type_B) {
                        uint temp = ent_A; ent_A = ent_B; ent_B = temp;
                        temp = type_A; type_A = type_B; type_B = temp;
                    }

                    if (type_A == TYPE_RIGID_BODY && type_B == TYPE_RIGID_BODY) {
                        uint out_idx = atomic_fetch_add_explicit((device atomic_uint*)&pc.out_rb_rb->count, 1, memory_order_relaxed);
                        if (out_idx < pc.max_pairs) pc.out_rb_rb->pairs[out_idx] = uint2(ent_A, ent_B);
                    } else if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_RIGID_BODY) {
                        uint out_idx = atomic_fetch_add_explicit((device atomic_uint*)&pc.out_rb_ps->count, 1, memory_order_relaxed);
                        if (out_idx < pc.max_pairs) pc.out_rb_ps->pairs[out_idx] = uint2(ent_B, ent_A);
                    } else if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_PARTICLE_SYSTEM) {
                        uint out_idx = atomic_fetch_add_explicit((device atomic_uint*)&pc.out_ps_ps->count, 1, memory_order_relaxed);
                        if (out_idx < pc.max_pairs) pc.out_ps_ps->pairs[out_idx] = uint2(ent_A, ent_B);
                    }
                }
            }
        }

        uint node_count = simd_sum(hit_node ? 1 : 0);
        uint push_offset = simd_prefix_exclusive_sum(hit_node ? 1 : 0);

        if (hit_node) shared_stacks[subgroup_id][stack_ptr + push_offset] = child_payload;
        if (lane_id == 0) shared_stack_ptrs[subgroup_id] = stack_ptr + node_count;
    }
}

// --- msl_bp_particle_self.txt ---
// @assets/sim/bp_particle_self.comp

struct PushConstants_bp_particle_self {
    ulong bvh;
    ulong particles;
    ulong wrench_buffer;
    uint root_index;
    uint total_particles;
    float particle_radius;
    float stiffness;
};

[[kernel]] void bp_particle_self(
    constant PushConstants_bp_particle_self& pc [[buffer(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]
) {
    uint SUBGROUPS_PER_WG = 8; // 256 / 32
    uint my_p_idx = threadgroup_position_in_grid.x * SUBGROUPS_PER_WG + simdgroup_index_in_threadgroup;
    if (my_p_idx >= pc.total_particles) return;

    threadgroup uint shared_stacks[8][32];
    threadgroup uint shared_stack_ptrs[8];

    float3 my_pos, my_min, my_max;
    
    device atomic_uint* particles_buf = PTR(atomic_uint, pc.particles);

    if (thread_index_in_simdgroup == 0) {
        uint block_idx = my_p_idx / 32;
        uint local_idx = my_p_idx % 32;
        uint base = block_idx * 320 + local_idx;

        my_pos = float3(
            as_type<float>(atomic_load_explicit(&particles_buf[base + 0], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&particles_buf[base + 32], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&particles_buf[base + 64], memory_order_relaxed))
        );

        my_min = my_pos - pc.particle_radius;
        my_max = my_pos + pc.particle_radius;

        shared_stacks[simdgroup_index_in_threadgroup][0] = pc.root_index;
        shared_stack_ptrs[simdgroup_index_in_threadgroup] = 1;
    }

    my_pos = float3(simd_broadcast(my_pos.x, 0), simd_broadcast(my_pos.y, 0), simd_broadcast(my_pos.z, 0));
    my_min = float3(simd_broadcast(my_min.x, 0), simd_broadcast(my_min.y, 0), simd_broadcast(my_min.z, 0));
    my_max = float3(simd_broadcast(my_max.x, 0), simd_broadcast(my_max.y, 0), simd_broadcast(my_max.z, 0));
    my_p_idx = simd_broadcast(my_p_idx, 0);

    float3 local_repulsive_force(0.0f);

    device MultiBvhNode* bvh = PTR(MultiBvhNode, pc.bvh);

    while (true) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint stack_ptr = shared_stack_ptrs[simdgroup_index_in_threadgroup];
        if (stack_ptr == 0) break;

        stack_ptr--;
        uint node_idx = shared_stacks[simdgroup_index_in_threadgroup][stack_ptr];
        if (thread_index_in_simdgroup == 0) shared_stack_ptrs[simdgroup_index_in_threadgroup] = stack_ptr;

        uint meta = bvh[node_idx].met[thread_index_in_simdgroup];
        bool valid = is_vd(bvh[node_idx].vmk, thread_index_in_simdgroup);

        float3 c_min = float3(
            bvh[node_idx].mx[thread_index_in_simdgroup],
            bvh[node_idx].my[thread_index_in_simdgroup],
            bvh[node_idx].mz[thread_index_in_simdgroup]
        );
        float3 c_max = float3(
            bvh[node_idx].mxx[thread_index_in_simdgroup],
            bvh[node_idx].mxy[thread_index_in_simdgroup],
            bvh[node_idx].mxz[thread_index_in_simdgroup]
        );
        uint child_payload = bvh[node_idx].chd[thread_index_in_simdgroup];

        bool hit_aabb = valid && iAABB(my_min, my_max, c_min, c_max);
        bool is_leaf_node = is_lf(meta);

        bool hit_node = hit_aabb && !is_leaf_node;
        bool hit_leaf = hit_aabb && is_leaf_node && (my_p_idx != child_payload);

        ulong leaf_ballot = get_ballot(hit_leaf);

        while (leaf_ballot != 0) {
            uint bit = ctz(leaf_ballot);
            leaf_ballot &= ~(1ul << bit);

            uint other_idx = simd_shuffle(child_payload, bit);
            uint block_idx = other_idx / 32;
            uint local_idx = other_idx % 32;
            uint base_idx = block_idx * 320 + local_idx;

            float3 other_pos = float3(
                as_type<float>(atomic_load_explicit(&particles_buf[base_idx + 0], memory_order_relaxed)),
                as_type<float>(atomic_load_explicit(&particles_buf[base_idx + 32], memory_order_relaxed)),
                as_type<float>(atomic_load_explicit(&particles_buf[base_idx + 64], memory_order_relaxed))
            );

            float3 diff = my_pos - other_pos;
            float dist_sq = dot(diff, diff);
            float min_dist = pc.particle_radius * 2.0f;

            if (dist_sq > 1e-12f && dist_sq < min_dist * min_dist) {
                float dist = sqrt(dist_sq);
                float penetration = min_dist - dist;
                float3 normal = diff / dist;

                float force_mag = pc.stiffness * penetration;
                local_repulsive_force += normal * force_mag;
            }
        }

        ulong node_ballot = get_ballot(hit_node);
        uint node_count = popcount(node_ballot);
        uint push_offset = popcount(node_ballot & ((1ul << thread_index_in_simdgroup) - 1ul));

        if (hit_node) {
            shared_stacks[simdgroup_index_in_threadgroup][stack_ptr + push_offset] = child_payload;
        }
        if (thread_index_in_simdgroup == 0) shared_stack_ptrs[simdgroup_index_in_threadgroup] = stack_ptr + node_count;
    }

    local_repulsive_force = float3(simd_sum(local_repulsive_force.x), simd_sum(local_repulsive_force.y), simd_sum(local_repulsive_force.z));

    if (thread_index_in_simdgroup == 0 && dot(local_repulsive_force, local_repulsive_force) > 0.0f) {
        device Wrench* wr = PTR(Wrench, pc.wrench_buffer);
        atomic_add_f(&wr[my_p_idx].fx, local_repulsive_force.x);
        atomic_add_f(&wr[my_p_idx].fy, local_repulsive_force.y);
        atomic_add_f(&wr[my_p_idx].fz, local_repulsive_force.z);
    }
}


// --- msl_ccd.txt ---
#include <metal_stdlib>
using namespace metal;

struct PushConstants_ccd {
    device struct MultiBvhBuffer* particle_bvh;
    device struct SparseCollisions* output_list;
    device struct ParticleData* particles;
    uint root_index;
    uint total_particles;
    float particle_radius;
    float dt;
};

[[kernel]]
void ccd(constant PushConstants_ccd& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint idx = thread_position_in_grid.x; 
    if (idx >= pc.total_particles) return;

    uint my_prim_id = idx;
    uint baseA = (my_prim_id / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (my_prim_id % SUBGROUP_SIZE);
    
    float3 my_center = float3(pc.particles->data[baseA+0], pc.particles->data[baseA+1*SUBGROUP_SIZE], pc.particles->data[baseA+2*SUBGROUP_SIZE]);
    float3 my_vel = float3(pc.particles->data[baseA+3*SUBGROUP_SIZE], pc.particles->data[baseA+4*SUBGROUP_SIZE], pc.particles->data[baseA+5*SUBGROUP_SIZE]);
    float3 p1 = my_center + my_vel * pc.dt;

    AABB my_aabb;
    my_aabb.minBounds = min(my_center - float3(pc.particle_radius), p1 - float3(pc.particle_radius));
    my_aabb.maxBounds = max(my_center + float3(pc.particle_radius), p1 + float3(pc.particle_radius));

    uint stack[64]; 
    int stackPtr = 0; 
    if (pc.root_index != 0xFFFFFFFFu) stack[stackPtr++] = pc.root_index;
    
    uint collisions_found = 0;

    while (stackPtr > 0) {
        uint node_idx = stack[--stackPtr];
        
        for (uint i = 0; i < SUBGROUP_SIZE; ++i) {
            if (!bvh_node_is_valid(pc.particle_bvh->nodes[node_idx].valid_mask, i)) continue;

            AABB bound;
            bound.minBounds = float3(pc.particle_bvh->nodes[node_idx].min_x[i], pc.particle_bvh->nodes[node_idx].min_y[i], pc.particle_bvh->nodes[node_idx].min_z[i]);
            bound.maxBounds = float3(pc.particle_bvh->nodes[node_idx].max_x[i], pc.particle_bvh->nodes[node_idx].max_y[i], pc.particle_bvh->nodes[node_idx].max_z[i]);

            if (intersectAABB(my_aabb, bound)) {
                uint meta = pc.particle_bvh->nodes[node_idx].metadata[i];
                uint offset = bvh_get_index(meta);

                if (bvh_is_leaf(meta)) {
                    if (my_prim_id < offset) {
                        float toi = 0.0, depth = 0.0; 
                        float3 normal = float3(0.0);
                        float3 point = float3(0.0);
                        
                        uint baseB = (offset / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (offset % SUBGROUP_SIZE);
                        float3 other_vel = float3(pc.particles->data[baseB+3*SUBGROUP_SIZE], pc.particles->data[baseB+4*SUBGROUP_SIZE], pc.particles->data[baseB+5*SUBGROUP_SIZE]) * pc.dt;
                        
                        float4x4 transA = float4x4(1.0); 
                        transA[3].xyz = my_center;
                        
                        float4x4 transB = float4x4(1.0); 
                        transB[3].xyz = float3(pc.particle_bvh->nodes[node_idx].com_x[i], pc.particle_bvh->nodes[node_idx].com_y[i], pc.particle_bvh->nodes[node_idx].com_z[i]);

                        if (compute_toi_generic(0, float3(pc.particle_radius,0,0), transA, my_vel * pc.dt, 0, float3(pc.particle_radius,0,0), transB, other_vel, 1e-3, 10, toi, normal, point, depth)) {
                            if (collisions_found < 16) {
                                uint outIdx = idx * 16 + collisions_found++;
                                pc.output_list->pairs[outIdx].valid = 1; 
                                pc.output_list->pairs[outIdx].prim_a = my_prim_id; 
                                pc.output_list->pairs[outIdx].prim_b = offset;
                                pc.output_list->pairs[outIdx].toi = toi; 
                                pc.output_list->pairs[outIdx].contact_normal = float4(normal, 0.0);
                                pc.output_list->pairs[outIdx].contact_point = float4(point, 1.0); 
                                pc.output_list->pairs[outIdx].penetration_depth = depth;
                            }
                        }
                    }
                } else if (offset != 0xFFFFFFFFu) {
                    stack[stackPtr++] = offset;
                }
            }
        }
    }
}

// --- msl_narrow_ccd.txt ---
#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.glsl"
#include "imex_math.glsl"
#include "physics_core.glsl"

struct PushConstants {
    uint64_t scene_entities;
    uint64_t output_list;
    uint64_t cross_output_list;
    uint64_t particles;
    uint64_t pair_buffer;
    uint64_t cross_pair_buffer;
    uint64_t lca_entities;
    float dt;
    float particle_radius;
    uint space_type;
};

[[kernel]]
void narrow_ccd(
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    constant PushConstants& pc [[buffer(0)]]
) {
    uint pair_idx = thread_position_in_grid.x;
    
    uint idA, idB, lca_id;
    bool is_partA = false, is_partB = false;

    if (pc.space_type == 1) { // Cross
        device atomic_uint* cross_pairs_count_ptr = (device atomic_uint*)pc.cross_pair_buffer;
        uint cross_pairs_count = atomic_load_explicit(cross_pairs_count_ptr, memory_order_relaxed);
        if (pair_idx >= cross_pairs_count) return;
        device CrossPair* pairs = (device CrossPair*)(pc.cross_pair_buffer + 16);
        CrossPair pair = pairs[pair_idx];
        idA = pair.macro_id;
        idB = pair.micro_id;
        lca_id = pair.lca_id;
    } else { // Standard
        device atomic_uint* pair_buffer_count_ptr = (device atomic_uint*)pc.pair_buffer;
        uint pair_buffer_count = atomic_load_explicit(pair_buffer_count_ptr, memory_order_relaxed);
        if (pair_idx >= pair_buffer_count) return;
        device uint2* pairs = (device uint2*)(pc.pair_buffer + 8);
        uint2 pair = pairs[pair_idx];
        idA = pair.x;
        idB = pair.y;
    }

    float3 pos_A, vel_A, extents_A;
    uint shape_A;
    float4 orient_A = float4(0, 0, 0, 1);

    if (idA == 0xFFFFFFFFu) { 
        is_partA = true;
    }
    
    device RigidBody* bodies = (device RigidBody*)pc.scene_entities;
    RigidBody ent_A = bodies[idA];
    RigidBody ent_B = bodies[idB];
    
    shape_A = ent_A.shape_type;
    extents_A = ent_A.shape_extents;
    orient_A = ent_A.orientation;
    pos_A = ent_A.position_mass.xyz;
    vel_A = ent_A.linear_vel_drag.xyz;
    
    uint shape_B = ent_B.shape_type;
    float3 extents_B = ent_B.shape_extents;
    float4 orient_B = ent_B.orientation;
    float3 pos_B = ent_B.position_mass.xyz;
    float3 vel_B = ent_B.linear_vel_drag.xyz;

    float4x4 trans_A = float4x4(1.0);
    float4x4 trans_B = float4x4(1.0);
    
    if (pc.space_type == 1) {
        device LcaEntity* lca_entities = (device LcaEntity*)pc.lca_entities;
        LcaEntity lca = lca_entities[lca_id];
        float3 macro_rel_vel_au = vel_A - lca.linear_velocity;
        pos_A = (lca.inv_transform * float4(pos_A, 1.0)).xyz * AU_TO_KM;
        float3x3 lca_inv_trans_3x3 = float3x3(lca.inv_transform.columns[0].xyz, lca.inv_transform.columns[1].xyz, lca.inv_transform.columns[2].xyz);
        vel_A = (lca_inv_trans_3x3 * macro_rel_vel_au) * AU_TO_KM;
        extents_A *= AU_TO_KM;
        trans_A = lca.inv_transform; 
    }
    
    float3x3 rotA = quat_to_mat3(orient_A);
    trans_A = float4x4(
        float4(rotA.columns[0], 0),
        float4(rotA.columns[1], 0),
        float4(rotA.columns[2], 0),
        float4(pos_A, 1.0)
    );
    
    float3x3 rotB = quat_to_mat3(orient_B);
    trans_B = float4x4(
        float4(rotB.columns[0], 0),
        float4(rotB.columns[1], 0),
        float4(rotB.columns[2], 0),
        float4(pos_B, 1.0)
    );

    float toi, depth;
    float3 normal, contact;
    
    if (compute_toi_generic(shape_A, extents_A, trans_A, vel_A, shape_B, extents_B, trans_B, vel_B, 1e-3, 10, toi, normal, contact, depth)) {
        if (pc.space_type == 1) {
            device atomic_uint* count_ptr = (device atomic_uint*)pc.cross_output_list;
            uint count = atomic_fetch_add_explicit(count_ptr, 1, memory_order_relaxed);
            if (count < 4000u) {
                device CrossPair* pairs = (device CrossPair*)(pc.cross_output_list + 16);
                pairs[count].valid = 1u;
                pairs[count].macro_id = idA;
                pairs[count].micro_id = idB;
                pairs[count].lca_id = lca_id;
                pairs[count].toi = toi;
                pairs[count].contact_normal = float4(normal, 0.0);
                pairs[count].contact_point = float4(contact, 1.0);
                pairs[count].penetration_depth = depth;
            }
        } else {
            device atomic_uint* count_ptr = (device atomic_uint*)pc.output_list;
            uint count = atomic_fetch_add_explicit(count_ptr, 1, memory_order_relaxed);
            if (count < 4000u) {
                device SparseCollisionPair* pairs = (device SparseCollisionPair*)(pc.output_list + 16);
                pairs[count].entity_a = idA;
                pairs[count].prim_a = idA;
                pairs[count].entity_b = idB;
                pairs[count].prim_b = idB;
                pairs[count].toi = toi;
                pairs[count].contact_normal = float4(normal, 0.0);
                pairs[count].contact_point = float4(contact, 1.0);
                pairs[count].penetration_depth = depth;
                pairs[count].bda_a = pc.scene_entities + idA * 128u;
                pairs[count].bda_b = pc.scene_entities + idB * 128u;
                pairs[count].frame_bda = 0; 
                pairs[count].valid = 1u;
            }
        }
    }
}


// --- msl_lcp_solver.txt ---
#include <metal_stdlib>

#include "imex_math.metal"

using namespace metal;

#define MAX_BODIES_PER_ISLAND 32
#define SUBGROUP_SIZE 32

struct PushConstants {
    device ParticleData* particles;
    device PackedCollisions* collisions;
    device ImpulseOutput* outputs;
    uint total_clusters;
    device RigidBodyArray* rigid_bodies;
    float dt;
    float restitution;
};

void generate_tangents(float3 normal, thread float3& t1, thread float3& t2) {
    if (abs(normal.x) >= 0.57735) {
        t1 = normalize(float3(normal.y, -normal.x, 0.0));
    } else {
        t1 = normalize(float3(0.0, normal.z, -normal.y));
    }
    t2 = cross(normal, t1);
}

float compute_effective_mass(float3 dir, float3 rA, float3 rB, float invMA, float invMB, float3 invIA, float3 invIB, float4 qA, float4 qB) {
    float3 I_crossA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, dir)));
    float3 I_crossB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, dir)));
    return 1.0 / max(invMA + invMB + dot(I_crossA, cross(rA, dir)) + dot(I_crossB, cross(rB, dir)), 1e-6);
}

void AtomicAddFloatShared(threadgroup atomic_uint& dest, float val) {
    uint old_val = atomic_load_explicit(&dest, memory_order_relaxed);
    uint assumed_val;
    uint new_val;
    do {
        assumed_val = old_val;
        new_val = as_type<uint>(as_type<float>(assumed_val) + val);
    } while (!atomic_compare_exchange_weak_explicit(&dest, &old_val, new_val, memory_order_relaxed, memory_order_relaxed));
}

[[kernel]]
void lcp_solver(
    constant PushConstants& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]])
{
    uint local_id = thread_index_in_threadgroup;
    uint contact_idx = thread_position_in_grid.x;
    
    device PackedCollisions* collisions = pc.collisions;
    device RigidBodyArray* rigid_bodies = pc.rigid_bodies;
    device ParticleData* particles = pc.particles;
    device ImpulseOutput* outputs = pc.outputs;
    
    uint collisions_count = collisions->count;
    bool valid = (contact_idx < collisions_count);

    threadgroup atomic_uint shared_v_x[MAX_BODIES_PER_ISLAND];
    threadgroup atomic_uint shared_v_y[MAX_BODIES_PER_ISLAND];
    threadgroup atomic_uint shared_v_z[MAX_BODIES_PER_ISLAND];
    threadgroup atomic_uint shared_w_x[MAX_BODIES_PER_ISLAND];
    threadgroup atomic_uint shared_w_y[MAX_BODIES_PER_ISLAND];
    threadgroup atomic_uint shared_w_z[MAX_BODIES_PER_ISLAND];
    threadgroup float accumulated_normal[128];
    threadgroup float accumulated_t1[128];
    threadgroup float accumulated_t2[128];

    accumulated_normal[local_id] = 0.0;
    accumulated_t1[local_id] = 0.0;
    accumulated_t2[local_id] = 0.0;

    if (local_id < MAX_BODIES_PER_ISLAND) {
        RigidBody rb = rigid_bodies->bodies[local_id];
        atomic_store_explicit(&shared_v_x[local_id], as_type<uint>(rb.linear_vel_drag.x), memory_order_relaxed);
        atomic_store_explicit(&shared_v_y[local_id], as_type<uint>(rb.linear_vel_drag.y), memory_order_relaxed);
        atomic_store_explicit(&shared_v_z[local_id], as_type<uint>(rb.linear_vel_drag.z), memory_order_relaxed);
        atomic_store_explicit(&shared_w_x[local_id], as_type<uint>(rb.angular_vel_drag.x), memory_order_relaxed);
        atomic_store_explicit(&shared_w_y[local_id], as_type<uint>(rb.angular_vel_drag.y), memory_order_relaxed);
        atomic_store_explicit(&shared_w_z[local_id], as_type<uint>(rb.angular_vel_drag.z), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!valid) return;

    PackedPair pair = collisions->pairs[contact_idx];
    bool is_partA = (pair.a.entity_id == 0xFFFFFFFFu);
    bool is_partB = (pair.b.entity_id == 0xFFFFFFFFu);
    uint idA = pair.a.primitive_index;
    uint idB = pair.b.primitive_index;

    float invMA = 0.0;
    float invMB = 0.0;
    float3 invIA = float3(0.0);
    float3 invIB = float3(0.0);
    float4 qA = float4(0, 0, 0, 1);
    float4 qB = float4(0, 0, 0, 1);
    float3 posA = float3(0.0);
    float3 posB = float3(0.0);
    float3 vA_init = float3(0.0);
    float3 wA_init = float3(0.0);
    float3 vB_init = float3(0.0);
    float3 wB_init = float3(0.0);

    if (is_partA) {
        uint baseA = (idA / SUBGROUP_SIZE) * 10u * SUBGROUP_SIZE + (idA % SUBGROUP_SIZE);
        posA = float3(
            as_type<float>(particles->data[baseA]),
            as_type<float>(particles->data[baseA + SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseA + 2 * SUBGROUP_SIZE])
        );
        vA_init = float3(
            as_type<float>(particles->data[baseA + 3 * SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseA + 4 * SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseA + 5 * SUBGROUP_SIZE])
        );
        float mass = as_type<float>(particles->data[baseA + 6u * SUBGROUP_SIZE]);
        invMA = (mass > 0.0) ? 1.0 / mass : 0.0;
    } else {
        RigidBody rbA = rigid_bodies->bodies[idA];
        invMA = rbA.position_mass.w > 0.0 ? 1.0 / rbA.position_mass.w : 0.0;
        invIA = rbA.inertia_tensor_inv.xyz;
        qA = rbA.orientation;
        posA = rbA.position_mass.xyz;
        vA_init = rbA.linear_vel_drag.xyz;
        wA_init = rbA.angular_vel_drag.xyz;
    }

    if (is_partB) {
        uint baseB = (idB / SUBGROUP_SIZE) * 10u * SUBGROUP_SIZE + (idB % SUBGROUP_SIZE);
        posB = float3(
            as_type<float>(particles->data[baseB]),
            as_type<float>(particles->data[baseB + SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseB + 2 * SUBGROUP_SIZE])
        );
        vB_init = float3(
            as_type<float>(particles->data[baseB + 3 * SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseB + 4 * SUBGROUP_SIZE]),
            as_type<float>(particles->data[baseB + 5 * SUBGROUP_SIZE])
        );
        float mass = as_type<float>(particles->data[baseB + 6u * SUBGROUP_SIZE]);
        invMB = (mass > 0.0) ? 1.0 / mass : 0.0;
    } else {
        RigidBody rbB = rigid_bodies->bodies[idB];
        invMB = rbB.position_mass.w > 0.0 ? 1.0 / rbB.position_mass.w : 0.0;
        invIB = rbB.inertia_tensor_inv.xyz;
        qB = rbB.orientation;
        posB = rbB.position_mass.xyz;
        vB_init = rbB.linear_vel_drag.xyz;
        wB_init = rbB.angular_vel_drag.xyz;
    }

    float3 n = pair.contact_normal.xyz;
    float3 t1, t2;
    generate_tangents(n, t1, t2);
    float3 rA = pair.contact_point.xyz - posA;
    float3 rB = pair.contact_point.xyz - posB;
    
    float eff_m_n = compute_effective_mass(n, rA, rB, invMA, invMB, invIA, invIB, qA, qB);
    float eff_m_t1 = compute_effective_mass(t1, rA, rB, invMA, invMB, invIA, invIB, qA, qB);
    float eff_m_t2 = compute_effective_mass(t2, rA, rB, invMA, invMB, invIA, invIB, qA, qB);

    float3 v_rel_init = (vB_init + cross(wB_init, rB)) - (vA_init + cross(wA_init, rA));
    float bounce = dot(v_rel_init, n) < -0.1 ? -pc.restitution * dot(v_rel_init, n) : 0.0;
    float target_v_n = bounce + ((0.2 / max(pc.dt, 1e-6)) * max(pair.penetration_depth - 0.01, 0.0));

    for (int iter = 0; iter < 20; ++iter) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float3 vA = vA_init;
        float3 wA = wA_init;
        if (!is_partA && idA < MAX_BODIES_PER_ISLAND) {
            vA = float3(as_type<float>(atomic_load_explicit(&shared_v_x[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_y[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_z[idA], memory_order_relaxed)));
            wA = float3(as_type<float>(atomic_load_explicit(&shared_w_x[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_y[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_z[idA], memory_order_relaxed)));
        }

        float3 vB = vB_init;
        float3 wB = wB_init;
        if (!is_partB && idB < MAX_BODIES_PER_ISLAND) {
            vB = float3(as_type<float>(atomic_load_explicit(&shared_v_x[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_y[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_z[idB], memory_order_relaxed)));
            wB = float3(as_type<float>(atomic_load_explicit(&shared_w_x[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_y[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_z[idB], memory_order_relaxed)));
        }

        float3 v_rel = (vB + cross(wB, rB)) - (vA + cross(wA, rA));
        float jn_delta = eff_m_n * (-dot(v_rel, n) + target_v_n);
        float old_jn = accumulated_normal[local_id];
        float new_jn = max(old_jn + jn_delta, 0.0);
        jn_delta = new_jn - old_jn;
        accumulated_normal[local_id] = new_jn;
        float3 P_n = jn_delta * n;

        if (!is_partA && invMA > 0.0 && idA < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idA], -P_n.x * invMA);
            AtomicAddFloatShared(shared_v_y[idA], -P_n.y * invMA);
            AtomicAddFloatShared(shared_v_z[idA], -P_n.z * invMA);
            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -P_n)));
            AtomicAddFloatShared(shared_w_x[idA], dwA.x);
            AtomicAddFloatShared(shared_w_y[idA], dwA.y);
            AtomicAddFloatShared(shared_w_z[idA], dwA.z);
        }
        if (!is_partB && invMB > 0.0 && idB < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idB], P_n.x * invMB);
            AtomicAddFloatShared(shared_v_y[idB], P_n.y * invMB);
            AtomicAddFloatShared(shared_v_z[idB], P_n.z * invMB);
            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, P_n)));
            AtomicAddFloatShared(shared_w_x[idB], dwB.x);
            AtomicAddFloatShared(shared_w_y[idB], dwB.y);
            AtomicAddFloatShared(shared_w_z[idB], dwB.z);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!is_partA && idA < MAX_BODIES_PER_ISLAND) {
            vA = float3(as_type<float>(atomic_load_explicit(&shared_v_x[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_y[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_z[idA], memory_order_relaxed)));
            wA = float3(as_type<float>(atomic_load_explicit(&shared_w_x[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_y[idA], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_z[idA], memory_order_relaxed)));
        }
        if (!is_partB && idB < MAX_BODIES_PER_ISLAND) {
            vB = float3(as_type<float>(atomic_load_explicit(&shared_v_x[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_y[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_v_z[idB], memory_order_relaxed)));
            wB = float3(as_type<float>(atomic_load_explicit(&shared_w_x[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_y[idB], memory_order_relaxed)),
                        as_type<float>(atomic_load_explicit(&shared_w_z[idB], memory_order_relaxed)));
        }
        v_rel = (vB + cross(wB, rB)) - (vA + cross(wA, rA));

        float max_fric = 0.5 * accumulated_normal[local_id];
        float jt1_delta = eff_m_t1 * (-dot(v_rel, t1));
        float old_jt1 = accumulated_t1[local_id];
        float new_jt1 = clamp(old_jt1 + jt1_delta, -max_fric, max_fric);
        jt1_delta = new_jt1 - old_jt1;
        accumulated_t1[local_id] = new_jt1;

        float jt2_delta = eff_m_t2 * (-dot(v_rel, t2));
        float old_jt2 = accumulated_t2[local_id];
        float new_jt2 = clamp(old_jt2 + jt2_delta, -max_fric, max_fric);
        jt2_delta = new_jt2 - old_jt2;
        accumulated_t2[local_id] = new_jt2;

        float3 P_t = jt1_delta * t1 + jt2_delta * t2;

        if (!is_partA && invMA > 0.0 && idA < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idA], -P_t.x * invMA);
            AtomicAddFloatShared(shared_v_y[idA], -P_t.y * invMA);
            AtomicAddFloatShared(shared_v_z[idA], -P_t.z * invMA);
            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -P_t)));
            AtomicAddFloatShared(shared_w_x[idA], dwA.x);
            AtomicAddFloatShared(shared_w_y[idA], dwA.y);
            AtomicAddFloatShared(shared_w_z[idA], dwA.z);
        }
        if (!is_partB && invMB > 0.0 && idB < MAX_BODIES_PER_ISLAND) {
            AtomicAddFloatShared(shared_v_x[idB], P_t.x * invMB);
            AtomicAddFloatShared(shared_v_y[idB], P_t.y * invMB);
            AtomicAddFloatShared(shared_v_z[idB], P_t.z * invMB);
            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, P_t)));
            AtomicAddFloatShared(shared_w_x[idB], dwB.x);
            AtomicAddFloatShared(shared_w_y[idB], dwB.y);
            AtomicAddFloatShared(shared_w_z[idB], dwB.z);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    outputs->impulses[contact_idx] = accumulated_normal[local_id] * n + accumulated_t1[local_id] * t1 + accumulated_t2[local_id] * t2;
}

// --- msl_apply_impulses.txt ---
#include <metal_stdlib>
using namespace metal;

inline void atomic_add_float(device atomic_uint* ptr, float val) {
    uint old_val = atomic_load_explicit(ptr, memory_order_relaxed);
    uint assumed_val;
    do {
        assumed_val = old_val;
        uint new_val = as_type<uint>(as_type<float>(assumed_val) + val);
    } while (!atomic_compare_exchange_weak_explicit(ptr, &old_val, new_val, memory_order_relaxed, memory_order_relaxed));
}

float4 quat_conj(float4 q) {
    return float4(-q.xyz, q.w);
}

float3 quat_rotate(float4 q, float3 v) {
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float3 quat_rotate_inv(float4 q, float3 v) {
    return quat_rotate(quat_conj(q), v);
}

struct ColliderId {
    uint entity_id;
    uint primitive_index;
};

struct PackedPair {
    ColliderId a;
    ColliderId b;
    float toi;
    float4 contact_normal;
    float4 contact_point;
    float penetration_depth;
};

struct PackedCollisions {
    uint dispatch_x;
    uint dispatch_y;
    uint dispatch_z;
    uint count;
    PackedPair pairs[1];
};

struct PushConstants {
    device atomic_uint* particles;
    device PackedCollisions* collisions;
    device float3* impulses; // float3 in MSL has 16-byte size/alignment matching std430 vec3
    device atomic_uint* rigid_bodies;
};

constant uint SUBGROUP_SIZE = 32;

kernel void apply_impulses(
    constant PushConstants& pc [[buffer(0)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= pc.collisions->count) return;

    PackedPair pair = pc.collisions->pairs[global_id];
    float3 impulse = pc.impulses[global_id];
    if (length(impulse) < 1e-6) return;

    uint pA_id = pair.a.primitive_index;
    uint pB_id = pair.b.primitive_index;

    bool is_rb_a = (pair.a.entity_id != 0xFFFFFFFFu);
    bool is_rb_b = (pair.b.entity_id != 0xFFFFFFFFu);

    if (is_rb_a) {
        uint base = pA_id * 28;
        float mass = as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 3], memory_order_relaxed));
        float invMA = mass > 0.0 ? 1.0 / mass : 0.0;

        float3 invIA = float3(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 16], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 17], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 18], memory_order_relaxed))
        );
        float4 qA = float4(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 4], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 5], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 6], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 7], memory_order_relaxed))
        );
        float3 posA = float3(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 0], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 1], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 2], memory_order_relaxed))
        );

        float3 rA = pair.contact_point.xyz - posA;

        if (invMA > 0.0) {
            float3 dvA = -impulse * invMA;
            atomic_add_float(&pc.rigid_bodies[base + 8], dvA.x);
            atomic_add_float(&pc.rigid_bodies[base + 9], dvA.y);
            atomic_add_float(&pc.rigid_bodies[base + 10], dvA.z);

            float3 dwA = quat_rotate(qA, invIA * quat_rotate_inv(qA, cross(rA, -impulse)));
            atomic_add_float(&pc.rigid_bodies[base + 12], dwA.x);
            atomic_add_float(&pc.rigid_bodies[base + 13], dwA.y);
            atomic_add_float(&pc.rigid_bodies[base + 14], dwA.z);
        }
    } else {
        uint base = (pA_id / SUBGROUP_SIZE) * (10u * SUBGROUP_SIZE) + (pA_id % SUBGROUP_SIZE);
        float mass = as_type<float>(atomic_load_explicit(&pc.particles[base + 6u * SUBGROUP_SIZE], memory_order_relaxed));
        float invMA = mass > 0.0 ? 1.0 / mass : 0.0;
        if (invMA > 0.0) {
            float3 dvA = -impulse * invMA;
            atomic_add_float(&pc.particles[base + 3u * SUBGROUP_SIZE], dvA.x);
            atomic_add_float(&pc.particles[base + 4u * SUBGROUP_SIZE], dvA.y);
            atomic_add_float(&pc.particles[base + 5u * SUBGROUP_SIZE], dvA.z);
        }
    }

    if (is_rb_b) {
        uint base = pB_id * 28;
        float mass = as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 3], memory_order_relaxed));
        float invMB = mass > 0.0 ? 1.0 / mass : 0.0;

        float3 invIB = float3(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 16], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 17], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 18], memory_order_relaxed))
        );
        float4 qB = float4(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 4], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 5], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 6], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 7], memory_order_relaxed))
        );
        float3 posB = float3(
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 0], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 1], memory_order_relaxed)),
            as_type<float>(atomic_load_explicit(&pc.rigid_bodies[base + 2], memory_order_relaxed))
        );

        float3 rB = pair.contact_point.xyz - posB;

        if (invMB > 0.0) {
            float3 dvB = impulse * invMB;
            atomic_add_float(&pc.rigid_bodies[base + 8], dvB.x);
            atomic_add_float(&pc.rigid_bodies[base + 9], dvB.y);
            atomic_add_float(&pc.rigid_bodies[base + 10], dvB.z);

            float3 dwB = quat_rotate(qB, invIB * quat_rotate_inv(qB, cross(rB, impulse)));
            atomic_add_float(&pc.rigid_bodies[base + 12], dwB.x);
            atomic_add_float(&pc.rigid_bodies[base + 13], dwB.y);
            atomic_add_float(&pc.rigid_bodies[base + 14], dwB.z);
        }
    } else {
        uint base = (pB_id / SUBGROUP_SIZE) * (10u * SUBGROUP_SIZE) + (pB_id % SUBGROUP_SIZE);
        float mass = as_type<float>(atomic_load_explicit(&pc.particles[base + 6u * SUBGROUP_SIZE], memory_order_relaxed));
        float invMB = mass > 0.0 ? 1.0 / mass : 0.0;
        if (invMB > 0.0) {
            float3 dvB = impulse * invMB;
            atomic_add_float(&pc.particles[base + 3u * SUBGROUP_SIZE], dvB.x);
            atomic_add_float(&pc.particles[base + 4u * SUBGROUP_SIZE], dvB.y);
            atomic_add_float(&pc.particles[base + 5u * SUBGROUP_SIZE], dvB.z);
        }
    }
}


// --- msl_stream_compact.txt ---
#if defined(KERNEL_stream_compact)

struct PushConstants_stream_compact {
    device void* sparse_in;
    device void* packed_out;
    uint total_elements;
};

[[kernel]]
void stream_compact(
    constant PushConstants_stream_compact& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
#ifdef DEBUG_SHADERS
    if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0 && thread_position_in_grid.z == 0) {
        // MSL doesn't typically support debugPrintfEXT natively, but we can log or ignore
    }
#endif

    uint id = thread_position_in_grid.x;
    
    device uint* sparse_in_count = (device uint*)pc.sparse_in;
    uint in_count = *sparse_in_count;

    if (id == 0) {
        device uint* packed_out_dispatch = (device uint*)pc.packed_out;
        packed_out_dispatch[3] = in_count; // count at offset 12
        uint blocks = (in_count + 127) / 128;
        packed_out_dispatch[0] = blocks;   // dispatch_x
        packed_out_dispatch[1] = 1;        // dispatch_y
        packed_out_dispatch[2] = 1;        // dispatch_z
    }

    if (id < in_count) {
        device SparseCollisionData* sparse_pairs = (device SparseCollisionData*)((device char*)pc.sparse_in + 16);
        device PackedPair* packed_pairs = (device PackedPair*)((device char*)pc.packed_out + 16);
        
        SparseCollisionData in_data = sparse_pairs[id];
        
        packed_pairs[id].a.entity_id = in_data.entity_a;
        packed_pairs[id].a.primitive_index = in_data.prim_a;
        packed_pairs[id].b.entity_id = in_data.entity_b;
        packed_pairs[id].b.primitive_index = in_data.prim_b;
        packed_pairs[id].toi = in_data.toi;
        packed_pairs[id].contact_normal = in_data.contact_normal;
        packed_pairs[id].contact_point = in_data.contact_point;
        packed_pairs[id].penetration_depth = in_data.penetration_depth;
    }
}

#endif // KERNEL_stream_compact

// --- msl_lbvh_build.txt ---
struct PushConstants_lbvh_build {
    MultiBvhBuffer bvh;
    MortonArray sorted_morton;
    AtomicCounters counters;
    ParticleData particles;
    uint num_primitives;
    float particle_radius;
    float dt;
};

int common_prefix(constant PushConstants_lbvh_build& pc, uint n, int i, int j) {
    if (j < 0 || j >= (int)n) return -1;
    uint key1 = pc.sorted_morton.entries[i].x; uint key2 = pc.sorted_morton.entries[j].x;
    if (key1 == key2) {
        uint idx1 = pc.sorted_morton.entries[i].y; uint idx2 = pc.sorted_morton.entries[j].y;
        return 32 + (31 - clz(idx1 ^ idx2));
    }
    return 31 - clz(key1 ^ key2);
}

float2 determine_range(constant PushConstants_lbvh_build& pc, uint n, int i) {
    int d = sign((float)(common_prefix(pc, n, i, i + 1) - common_prefix(pc, n, i, i - 1)));
    int min_p = common_prefix(pc, n, i, i - d), l_max = 2;
    while (common_prefix(pc, n, i, i + l_max * d) > min_p) l_max *= 2;
    int l = 0, t = l_max / 2;
    while (t >= 1) { if (common_prefix(pc, n, i, i + (l + t) * d) > min_p) l += t; t /= 2; }
    return float2(min(i, i + l * d), max(i, i + l * d));
}

int find_split(constant PushConstants_lbvh_build& pc, uint n, int first, int last) {
    int common_node = common_prefix(pc, n, first, last), split = first, step = last - first;
    do {
        step = (step + 1) >> 1; int new_split = split + step;
        if (new_split < last && common_prefix(pc, n, first, new_split) > common_node) split = new_split;
    } while (step > 1);
    return split;
}

[[kernel]]
void lbvh_build(constant PushConstants_lbvh_build& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint idx = thread_position_in_grid.x, n = pc.num_primitives;
    if (idx >= n) return;
    uint num_internal_nodes = n - 1;

    if (idx < num_internal_nodes) {
        float2 range = determine_range(pc, n, int(idx));
        int split = find_split(pc, n, int(range.x), int(range.y));
        uint left_child = (split == int(range.x)) ? (num_internal_nodes + split) : uint(split);
        uint right_child = (split + 1 == int(range.y)) ? (num_internal_nodes + split + 1) : uint(split + 1);

        pc.bvh.nodes[idx].child_indices[0] = left_child;
        pc.bvh.nodes[idx].child_indices[1] = right_child;
        pc.bvh.nodes[idx].valid_mask = uint2(3u, 0u);
        pc.bvh.nodes[left_child].parent_idx = idx;
        pc.bvh.nodes[right_child].parent_idx = idx;
    }

    uint leaf_idx = num_internal_nodes + idx, p_id = pc.sorted_morton.entries[idx].y;
    uint base = (p_id / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (p_id % SUBGROUP_SIZE);

    float3 pos = float3(P_READ(pc.particles, base+0), P_READ(pc.particles, base+1*SUBGROUP_SIZE), P_READ(pc.particles, base+2*SUBGROUP_SIZE));
    float3 vel = float3(P_READ(pc.particles, base+3*SUBGROUP_SIZE), P_READ(pc.particles, base+4*SUBGROUP_SIZE), P_READ(pc.particles, base+5*SUBGROUP_SIZE));
    float mass = P_READ(pc.particles, base+6*SUBGROUP_SIZE), r = pc.particle_radius;

    float3 p1 = pos + vel * pc.dt;
    float3 l_min = min(pos - float3(r), p1 - float3(r)), l_max = max(pos + float3(r), p1 + float3(r));

    uint current = pc.bvh.nodes[leaf_idx].parent_idx;
    uint is_right = (pc.bvh.nodes[current].child_indices[1] == leaf_idx) ? 1 : 0;

    pc.bvh.nodes[current].min_x[is_right] = l_min.x; pc.bvh.nodes[current].max_x[is_right] = l_max.x;
    pc.bvh.nodes[current].min_y[is_right] = l_min.y; pc.bvh.nodes[current].max_y[is_right] = l_max.y;
    pc.bvh.nodes[current].min_z[is_right] = l_min.z; pc.bvh.nodes[current].max_z[is_right] = l_max.z;
    pc.bvh.nodes[current].masses[is_right] = mass;
    pc.bvh.nodes[current].com_x[is_right] = pos.x; pc.bvh.nodes[current].com_y[is_right] = pos.y; pc.bvh.nodes[current].com_z[is_right] = pos.z;
    pc.bvh.nodes[current].metadata[is_right] = bvh_pack_metadata(true, BVH_FRAME_MICRO, BVH_SHAPE_AABB, p_id);

    threadgroup_barrier(mem_flags::mem_device);

    while (current != 0xFFFFFFFFu) {
        if (atomic_fetch_add_explicit((device atomic_uint*)&pc.counters.counts[current], 1, memory_order_relaxed) == 0) break;

        float3 c_l_min = float3(pc.bvh.nodes[current].min_x[0], pc.bvh.nodes[current].min_y[0], pc.bvh.nodes[current].min_z[0]);
        float3 c_l_max = float3(pc.bvh.nodes[current].max_x[0], pc.bvh.nodes[current].max_y[0], pc.bvh.nodes[current].max_z[0]);
        float l_m = pc.bvh.nodes[current].masses[0];
        float3 l_com = float3(pc.bvh.nodes[current].com_x[0], pc.bvh.nodes[current].com_y[0], pc.bvh.nodes[current].com_z[0]);

        float3 c_r_min = float3(pc.bvh.nodes[current].min_x[1], pc.bvh.nodes[current].min_y[1], pc.bvh.nodes[current].min_z[1]);
        float3 c_r_max = float3(pc.bvh.nodes[current].max_x[1], pc.bvh.nodes[current].max_y[1], pc.bvh.nodes[current].max_z[1]);
        float r_m = pc.bvh.nodes[current].masses[1];
        float3 r_com = float3(pc.bvh.nodes[current].com_x[1], pc.bvh.nodes[current].com_y[1], pc.bvh.nodes[current].com_z[1]);

        float3 c_min = min(c_l_min, c_r_min), c_max = max(c_l_max, c_r_max);
        float c_mass = l_m + r_m;
        float3 c_com = c_mass > 0.0 ? (l_com * l_m + r_com * r_m) / c_mass : (l_com + r_com) * 0.5;

        uint parent = pc.bvh.nodes[current].parent_idx;
        if (parent != 0xFFFFFFFFu) {
            uint is_r = (pc.bvh.nodes[parent].child_indices[1] == current) ? 1 : 0;
            pc.bvh.nodes[parent].min_x[is_r] = c_min.x; pc.bvh.nodes[parent].max_x[is_r] = c_max.x;
            pc.bvh.nodes[parent].min_y[is_r] = c_min.y; pc.bvh.nodes[parent].max_y[is_r] = c_max.y;
            pc.bvh.nodes[parent].min_z[is_r] = c_min.z; pc.bvh.nodes[parent].max_z[is_r] = c_max.z;
            pc.bvh.nodes[parent].masses[is_r] = c_mass;
            pc.bvh.nodes[parent].com_x[is_r] = c_com.x; pc.bvh.nodes[parent].com_y[is_r] = c_com.y; pc.bvh.nodes[parent].com_z[is_r] = c_com.z;
            pc.bvh.nodes[parent].metadata[is_r] = bvh_pack_metadata(false, BVH_FRAME_MICRO, BVH_SHAPE_AABB, current);
        }
        threadgroup_barrier(mem_flags::mem_device);
        current = parent;
    }
}


// --- msl_lbvh_collapse.txt ---
#include <metal_stdlib>
using namespace metal;

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#define BVH_FRAME_MACRO  0u
#define BVH_FRAME_MICRO  1u
#define BVH_SHAPE_AABB   0u
#define BVH_SHAPE_OBB    1u
#define BVH_SHAPE_SPHERE 2u

template<typename T>
inline T spvFindUMSB(T x) {
    return select(clz(T(0)) - (clz(x) + T(1)), T(-1), x == T(0));
}

struct MultiBvhNode {
    float min_x[SUBGROUP_SIZE]; float max_x[SUBGROUP_SIZE];
    float min_y[SUBGROUP_SIZE]; float max_y[SUBGROUP_SIZE];
    float min_z[SUBGROUP_SIZE]; float max_z[SUBGROUP_SIZE];
    uint child_indices[SUBGROUP_SIZE]; uint metadata[SUBGROUP_SIZE];
    float masses[SUBGROUP_SIZE];
    float com_x[SUBGROUP_SIZE]; float com_y[SUBGROUP_SIZE]; float com_z[SUBGROUP_SIZE];
    uint particle_start[SUBGROUP_SIZE]; uint particle_count[SUBGROUP_SIZE];
    uint2 valid_mask;
    uint parent_idx;
    uint pad;
    uint permutations[8][SUBGROUP_SIZE];
};

struct MultiBvhBuffer {
    MultiBvhNode nodes[1];
};

struct CollapseMapBuffer {
    uint binary_roots[1];
};

struct PushConstants {
    device MultiBvhBuffer* binary_bvh;
    device MultiBvhBuffer* multi_bvh;
    device CollapseMapBuffer* collapse_map;
    uint num_multi_nodes;
};

inline bool bvh_is_leaf(uint meta) { return (meta & 0x80000000u) != 0u; }
inline uint bvh_get_index(uint meta) { return meta & 0x07FFFFFFu; }
inline uint bvh_pack_metadata(bool is_leaf, uint frame, uint shape, uint index) {
    uint meta = index & 0x07FFFFFFu;
    meta |= (shape & 3u) << 27;
    meta |= (frame & 3u) << 29;
    if (is_leaf) meta |= 0x80000000u;
    return meta;
}

[[kernel]]
void lbvh_collapse(
    constant PushConstants& pc [[buffer(0)]],
    uint3 gl_WorkGroupID [[threadgroup_position_in_grid]],
    uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]]
) {
    uint multi_node_idx = gl_WorkGroupID.x;
    if (multi_node_idx >= pc.num_multi_nodes) return;

    uint lane = gl_SubgroupInvocationID;
    uint binary_idx = pc.collapse_map->binary_roots[multi_node_idx];
    
    bool is_leaf = false;
    uint payload = 0;
    uint f_parent = 0;
    uint f_dir = 0;

    int depth = int(spvFindUMSB(SUBGROUP_SIZE)) - 1;
    for (int d = depth; d >= 0; d--) {
        uint dir = (lane >> uint(d)) & 1u;
        uint meta = pc.binary_bvh->nodes[binary_idx].metadata[dir];
        
        is_leaf = bvh_is_leaf(meta);
        uint next_idx = bvh_get_index(meta);

        f_parent = binary_idx;
        f_dir = dir;
        if (is_leaf) { payload = next_idx; break; }
        binary_idx = next_idx;
    }

    if (!is_leaf) {
        payload = binary_idx;
        f_parent = pc.binary_bvh->nodes[binary_idx].parent_idx;
        f_dir = (pc.binary_bvh->nodes[f_parent].child_indices[1] == binary_idx) ? 1u : 0u;
    }

    pc.multi_bvh->nodes[multi_node_idx].min_x[lane] = pc.binary_bvh->nodes[f_parent].min_x[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].max_x[lane] = pc.binary_bvh->nodes[f_parent].max_x[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].min_y[lane] = pc.binary_bvh->nodes[f_parent].min_y[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].max_y[lane] = pc.binary_bvh->nodes[f_parent].max_y[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].min_z[lane] = pc.binary_bvh->nodes[f_parent].min_z[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].max_z[lane] = pc.binary_bvh->nodes[f_parent].max_z[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].child_indices[lane] = payload;
    
    pc.multi_bvh->nodes[multi_node_idx].metadata[lane] = bvh_pack_metadata(is_leaf, BVH_FRAME_MICRO, BVH_SHAPE_AABB, payload);
    
    pc.multi_bvh->nodes[multi_node_idx].masses[lane] = pc.binary_bvh->nodes[f_parent].masses[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].com_x[lane] = pc.binary_bvh->nodes[f_parent].com_x[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].com_y[lane] = pc.binary_bvh->nodes[f_parent].com_y[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].com_z[lane] = pc.binary_bvh->nodes[f_parent].com_z[f_dir];

    if (lane == 0u) {
        uint mask_x = (SUBGROUP_SIZE >= 32u) ? 0xFFFFFFFFu : ((1u << SUBGROUP_SIZE) - 1u);
        uint mask_y = 0u;
        if (SUBGROUP_SIZE > 32u) mask_y = (SUBGROUP_SIZE >= 64u) ? 0xFFFFFFFFu : ((1u << (SUBGROUP_SIZE - 32u)) - 1u);
        
        pc.multi_bvh->nodes[multi_node_idx].valid_mask = uint2(mask_x, mask_y);
        for (uint i = 0u; i < 8u; ++i) {
            for (uint j = 0u; j < SUBGROUP_SIZE; ++j) {
                pc.multi_bvh->nodes[multi_node_idx].permutations[i][j] = j;
            }
        }
    }
}


// --- msl_lbvh_prepass.txt ---
#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.msl"

#ifdef KERNEL_LBVH_PREPASS

struct PushConstants_lbvh_prepass {
    device MultiBvhNode* bvh;
    device uint* counters;
    uint num_internal_nodes;
};

kernel void lbvh_prepass(
    constant PushConstants_lbvh_prepass& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.num_internal_nodes) return;
    
    pc.counters[idx] = 0u;
    
    if (idx == 0u) {
        pc.bvh[0].parent_idx = 0xFFFFFFFFu;
    }
}

#endif // KERNEL_LBVH_PREPASS


// --- msl_morton_encode.txt ---
// @assets/sim/morton_encode.comp
//
// Calculates a 30-bit Morton Code for each particle to be used for radix sorting
//
// Target: MSL Metal 3.0

struct PushConstants_morton_encode {
    MortonArray morton_out;
    ParticleData particles;
    uint num_particles;
    float3 scene_min;
    float3 scene_max;
};

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
inline uint morton_encode_expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline uint morton_encode_morton3D(float3 norm_pos) {
    norm_pos = clamp(norm_pos, 0.0f, 1.0f);
    uint x = uint(norm_pos.x * 1023.0f);
    uint y = uint(norm_pos.y * 1023.0f);
    uint z = uint(norm_pos.z * 1023.0f);
    return (morton_encode_expandBits(x) << 2) | (morton_encode_expandBits(y) << 1) | morton_encode_expandBits(z);
}

[[kernel]]
void morton_encode(constant PushConstants_morton_encode& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.num_particles) return;

    // AOSOA unpacking matching your particle structure
    uint block_idx = idx / SUBGROUP_SIZE;
    uint local_idx = idx % SUBGROUP_SIZE;
    uint base = block_idx * (10 * SUBGROUP_SIZE) + local_idx;

    float3 pos = float3(
        P_READ(pc.particles, base + 0 * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 1 * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 2 * SUBGROUP_SIZE)
    );

    // Normalize relative to scene bounds
    float3 extents = pc.scene_max - pc.scene_min;
    float3 norm_pos = (pos - pc.scene_min) / max(extents, float3(1e-5f));

    uint m_code = morton_encode_morton3D(norm_pos);

    pc.morton_out.entries[idx] = uint2(m_code, idx);
}

// --- msl_radix_sort.txt ---
// @assets/sim/radix_sort.comp

struct PushConstants_radix_sort {
    device uint2* input_keys;
    device uint2* output_keys;
    device uint* histograms;
    uint num_particles;
    uint shift;
    uint stage;
    uint num_blocks;
};

#define STAGE_COUNT   0
#define STAGE_SCAN    1
#define STAGE_SCATTER 2

#define RADIX 16
#define ELEMENTS_PER_BLOCK 4096

[[kernel]]
void radix_sort(
    constant PushConstants_radix_sort& pc [[buffer(0)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroups_per_threadgroup [[simdgroups_per_threadgroup]]
) {
    uint lid = thread_position_in_threadgroup.x;
    uint wid = threadgroup_position_in_grid.x;
    uint sg_id = thread_index_in_simdgroup;
    uint sg_group_id = simdgroup_index_in_threadgroup;
    
    threadgroup atomic_uint s_counts[RADIX];
    threadgroup uint s_offsets[RADIX];
    threadgroup uint s_sg_counts[64];
    threadgroup uint s_bin_sums[RADIX];

    if (pc.stage == STAGE_COUNT) {
        if (lid < RADIX) atomic_store_explicit(&s_counts[lid], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint block_start = wid * ELEMENTS_PER_BLOCK;
        uint block_end = min(block_start + ELEMENTS_PER_BLOCK, pc.num_particles);

        for (uint i = block_start + lid; i < block_end; i += 256) {
            uint key = (pc.input_keys[i].x >> pc.shift) & 0xFu;
            atomic_fetch_add_explicit(&s_counts[key], 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < RADIX) {
            pc.histograms[lid * pc.num_blocks + wid] = atomic_load_explicit(&s_counts[lid], memory_order_relaxed);
        }
    }
    else if (pc.stage == STAGE_SCAN) {
        if (lid < RADIX) {
            uint bin_sum = 0;
            for (uint w = 0; w < pc.num_blocks; ++w) {
                bin_sum += pc.histograms[lid * pc.num_blocks + w];
            }
            s_bin_sums[lid] = bin_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            uint global_offset = 0;
            for (uint i = 0; i < RADIX; ++i) {
                uint val = s_bin_sums[i];
                s_bin_sums[i] = global_offset;
                global_offset += val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < RADIX) {
            uint running_offset = s_bin_sums[lid];
            for (uint w = 0; w < pc.num_blocks; ++w) {
                uint val = pc.histograms[lid * pc.num_blocks + w];
                pc.histograms[lid * pc.num_blocks + w] = running_offset;
                running_offset += val;
            }
        }
    }
    else if (pc.stage == STAGE_SCATTER) {
        if (lid < RADIX) {
            s_offsets[lid] = pc.histograms[lid * pc.num_blocks + wid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint block_start = wid * ELEMENTS_PER_BLOCK;
        uint block_end = min(block_start + ELEMENTS_PER_BLOCK, pc.num_particles);

        for (uint chunk_start = block_start; chunk_start < block_end; chunk_start += 256) {
            uint i = chunk_start + lid;
            bool valid = (i < block_end);
            
            uint2 my_entry = uint2(0, 0);
            if (valid) {
                my_entry = pc.input_keys[i];
            }
            uint my_key = valid ? ((my_entry.x >> pc.shift) & 0xFu) : 0xFFFFFFFFu;

            uint local_offset = 0;
            uint my_global_base = 0;

            for (uint b = 0; b < RADIX; ++b) {
                bool match = (my_key == b);

                uint sg_match_count = simd_sum(match ? 1 : 0);
                uint my_sg_offset = simd_prefix_exclusive_sum(match ? 1 : 0);

                if (sg_id == 0) {
                    s_sg_counts[sg_group_id] = sg_match_count;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lid == 0) {
                    uint sum = 0;
                    for (uint sg = 0; sg < simdgroups_per_threadgroup; ++sg) {
                        uint c = s_sg_counts[sg];
                        s_sg_counts[sg] = sum;
                        sum += c;
                    }
                    atomic_store_explicit(&s_counts[b], sum, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (match) {
                    local_offset = s_sg_counts[sg_group_id] + my_sg_offset;
                    my_global_base = s_offsets[b];
                }

                if (lid == 0) {
                    s_offsets[b] += atomic_load_explicit(&s_counts[b], memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (valid) {
                uint dest = my_global_base + local_offset;
                pc.output_keys[dest] = my_entry;
            }
        }
    }
}

// --- msl_motion_bounds.txt ---
#include <metal_stdlib>
using namespace metal;



#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#ifndef PRIMITIVE_TYPE
#define PRIMITIVE_TYPE 0
#endif

struct PushConstants_motion_bounds {
    device MultiBvhNode* bvh;
    device uint* primitive_data;
    uint num_primitives;
    float dt;
    float particle_radius;
};

[[kernel]]
void motion_bounds(
    constant PushConstants_motion_bounds& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.num_primitives) return;

    if (PRIMITIVE_TYPE == 0) {
        uint base = (idx / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (idx % SUBGROUP_SIZE);
        
        float pos_x = as_type<float>(pc.primitive_data[base + 0]);
        float pos_y = as_type<float>(pc.primitive_data[base + 1 * SUBGROUP_SIZE]);
        float pos_z = as_type<float>(pc.primitive_data[base + 2 * SUBGROUP_SIZE]);
        float3 pos = float3(pos_x, pos_y, pos_z);
        
        float vel_x = as_type<float>(pc.primitive_data[base + 3 * SUBGROUP_SIZE]);
        float vel_y = as_type<float>(pc.primitive_data[base + 4 * SUBGROUP_SIZE]);
        float vel_z = as_type<float>(pc.primitive_data[base + 5 * SUBGROUP_SIZE]);
        float3 vel = float3(vel_x, vel_y, vel_z);

        float3 p1 = pos + vel * pc.dt;
        float3 min_p = min(pos, p1) - pc.particle_radius;
        float3 max_p = max(pos, p1) + pc.particle_radius;

        uint leaf_idx = (pc.num_primitives - 1) + idx;
        uint parent = pc.bvh[leaf_idx].parent_idx;
        uint is_right = (pc.bvh[parent].child_indices[1] == leaf_idx) ? 1 : 0;

        pc.bvh[parent].min_x[is_right] = min_p.x; 
        pc.bvh[parent].max_x[is_right] = max_p.x;
        pc.bvh[parent].min_y[is_right] = min_p.y; 
        pc.bvh[parent].max_y[is_right] = max_p.y;
        pc.bvh[parent].min_z[is_right] = min_p.z; 
        pc.bvh[parent].max_z[is_right] = max_p.z;
    }
}


// --- msl_motion_refit.txt ---
#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.h"

struct PushConstants {
    device MultiBvhNode* bvh;
    device DepthIndices* depth_indices;
    uint total_nodes_at_depth;
};

[[kernel]]
void motion_refit(
    constant PushConstants& pc [[buffer(0)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= pc.total_nodes_at_depth) return;

    uint node_idx = pc.depth_indices->indices[global_id + 4];
    for (uint i = 0; i < 2; ++i) {
        uint child = pc.bvh[node_idx].child_indices[i];
        if (bvh_is_leaf(pc.bvh[node_idx].metadata[i])) {
            pc.bvh[node_idx].min_x[i] = pc.bvh[child].min_x[0];
            pc.bvh[node_idx].max_x[i] = pc.bvh[child].max_x[0];
            pc.bvh[node_idx].min_y[i] = pc.bvh[child].min_y[0];
            pc.bvh[node_idx].max_y[i] = pc.bvh[child].max_y[0];
            pc.bvh[node_idx].min_z[i] = pc.bvh[child].min_z[0];
            pc.bvh[node_idx].max_z[i] = pc.bvh[child].max_z[0];
        } else {
            pc.bvh[node_idx].min_x[i] = min(pc.bvh[child].min_x[0], pc.bvh[child].min_x[1]);
            pc.bvh[node_idx].max_x[i] = max(pc.bvh[child].max_x[0], pc.bvh[child].max_x[1]);
            pc.bvh[node_idx].min_y[i] = min(pc.bvh[child].min_y[0], pc.bvh[child].min_y[1]);
            pc.bvh[node_idx].max_y[i] = max(pc.bvh[child].max_y[0], pc.bvh[child].max_y[1]);
            pc.bvh[node_idx].min_z[i] = min(pc.bvh[child].min_z[0], pc.bvh[child].min_z[1]);
            pc.bvh[node_idx].max_z[i] = max(pc.bvh[child].max_z[0], pc.bvh[child].max_z[1]);
        }
    }
}


// --- msl_reduce_toi.txt ---
#include <metal_stdlib>
using namespace metal;

struct ColliderId {
    uint entity_id;
    uint primitive_index;
};

struct PackedPair {
    ColliderId a;
    ColliderId b;
    float toi;
    float4 contact_normal;
    float4 contact_point;
    float penetration_depth;
};

struct PackedCollisions {
    uint dispatch_x;
    uint dispatch_y;
    uint dispatch_z;
    uint count;
    PackedPair pairs[1];
};

struct OutputTOI {
    atomic_uint min_tc_uint;
};

struct PushConstants {
    device void* particles;
    device PackedCollisions* collisions;
    device OutputTOI* out_toi;
    float particle_radius;
    float dt;
};

constant uint MAX_SUBGROUPS = 128 / 32;

[[kernel]]
void reduce_toi(
    constant PushConstants& pc [[buffer(0)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroups_per_threadgroup [[simdgroups_per_threadgroup]]
) {
    threadgroup uint shared_min_toi[MAX_SUBGROUPS];

    float tc = pc.dt; // Default to max time

    if (global_id < pc.collisions->count) {
        tc = pc.collisions->pairs[global_id].toi;
    }

    // Subgroup reduction
    float subgroup_min_tc = simd_min(tc);

    if (thread_index_in_simdgroup == 0) {
        shared_min_toi[simdgroup_index] = as_type<uint>(subgroup_min_tc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Workgroup reduction
    if (local_id == 0) {
        uint wg_min_uint = shared_min_toi[0];
        for (uint i = 1; i < simdgroups_per_threadgroup; i++) {
            wg_min_uint = min(wg_min_uint, shared_min_toi[i]);
        }

        // Global reduction
        atomic_fetch_min_explicit(&pc.out_toi->min_tc_uint, wg_min_uint, memory_order_relaxed);
    }
}


// --- msl_graph_coloring.txt ---
#include <metal_stdlib>
using namespace metal;

struct ColliderId {
    uint entity_id;
    uint primitive_index;
};

struct PackedPair {
    ColliderId a;
    ColliderId b;
    float toi;
    float4 contact_normal;
    float4 contact_point;
    float penetration_depth;
};

struct PackedCollisions {
    uint dispatch_x;
    uint dispatch_y;
    uint dispatch_z;
    uint count;
    PackedPair pairs[1];
};

struct PushConstants_graph_coloring {
    device PackedCollisions* collisions;
    device uint* colors;
    device uint* weights;
    uint total_pairs;
};

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

[[kernel]]
void graph_coloring(
    constant PushConstants_graph_coloring& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.total_pairs) return;

    // NVIDIA Parallel ILU factorization graph coloring adapted for Vulkan 1.1 SPV1.4 Memory Model
    // We color the contact pairs (edges) so that independent contacts can be solved in parallel.

    // 1. Initialize weights
    pc.weights[idx] = hash(idx + 1);
    pc.colors[idx] = 0; // 0 means uncolored

    // Memory barrier to ensure all weights are visible
    threadgroup_barrier(mem_flags::mem_device);

    // 2. Luby's algorithm for independent sets
    bool colored = false;
    uint my_color = 1;
    uint my_weight = pc.weights[idx];
    
    PackedPair my_pair = pc.collisions->pairs[idx];
    uint my_a = my_pair.a.primitive_index;
    uint my_b = my_pair.b.primitive_index;

    for (int iter = 0; iter < 10; ++iter) {
        if (!colored) {
            bool is_max = true;
            
            // Check adjacent contacts (contacts sharing body A or body B)
            for (uint j = 0; j < pc.total_pairs; ++j) {
                if (idx == j) continue;
                PackedPair other_pair = pc.collisions->pairs[j];
                uint other_a = other_pair.a.primitive_index;
                uint other_b = other_pair.b.primitive_index;
                
                if (my_a == other_a || my_a == other_b || my_b == other_a || my_b == other_b) {
                    uint other_color = pc.colors[j];
                    if (other_color == 0 || other_color == my_color) {
                        uint other_weight = pc.weights[j];
                        if (other_weight > my_weight || (other_weight == my_weight && j > idx)) {
                            is_max = false;
                            break;
                        }
                    }
                }
            }
            
            if (is_max) {
                pc.colors[idx] = my_color;
                colored = true;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        if (!colored) {
            my_color++;
        }
    }
}


// --- msl_convert_particles.txt ---
#include <metal_stdlib>
using namespace metal;

#include "../debug_utils.metal"

struct ParticleData {
    uint id_low;
    uint id_high;
    uint age_low;
    uint age_high;
    packed_float3 position;
    float mass;
    packed_float3 velocity;
    uint is_active;
};

struct DrawIndirectCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct PushConstants {
    device float* aosoa_particles;
    device ParticleData* mega_particles;
    device DrawIndirectCommand* mega_indirect;
    device atomic_uint* atomic_counters;
    uint mega_indirect_index;
    uint mega_particle_offset;
};

constant uint SUBGROUP_SIZE = 32;

[[kernel]]
void convert_particles(
    uint thread_position_in_grid [[thread_position_in_grid]],
    constant PushConstants& pc [[buffer(0)]]
) {
    uint total_particles = atomic_load_explicit(&pc.atomic_counters[0], memory_order_relaxed);

    // Only thread 0 writes the indirect command
    if (thread_position_in_grid == 0) {
        pc.mega_indirect[pc.mega_indirect_index].vertexCount = 4;
        pc.mega_indirect[pc.mega_indirect_index].instanceCount = total_particles;
        pc.mega_indirect[pc.mega_indirect_index].firstVertex = 0;
        pc.mega_indirect[pc.mega_indirect_index].firstInstance = pc.mega_particle_offset;
    }

    uint idx = thread_position_in_grid;
    if (idx >= total_particles) {
        return;
    }

    uint in_block = idx / SUBGROUP_SIZE;
    uint in_lane  = idx % SUBGROUP_SIZE;
    uint in_base  = in_block * 10 * SUBGROUP_SIZE + in_lane;

    float3 pos;
    pos.x = pc.aosoa_particles[in_base + 0 * SUBGROUP_SIZE];
    pos.y = pc.aosoa_particles[in_base + 1 * SUBGROUP_SIZE];
    pos.z = pc.aosoa_particles[in_base + 2 * SUBGROUP_SIZE];

    float3 vel;
    vel.x = pc.aosoa_particles[in_base + 3 * SUBGROUP_SIZE];
    vel.y = pc.aosoa_particles[in_base + 4 * SUBGROUP_SIZE];
    vel.z = pc.aosoa_particles[in_base + 5 * SUBGROUP_SIZE];

    float mass = pc.aosoa_particles[in_base + 6 * SUBGROUP_SIZE];

    uint out_idx = pc.mega_particle_offset + idx;

    // We do not have IDs or Age from the physics simulation right now.
    // They could be added in emit_particles.comp in the future.
    pc.mega_particles[out_idx].id_low = 0;
    pc.mega_particles[out_idx].id_high = 0;
    pc.mega_particles[out_idx].age_low = 0;
    pc.mega_particles[out_idx].age_high = 0;
    pc.mega_particles[out_idx].position = pos;
    pc.mega_particles[out_idx].mass = mass;
    pc.mega_particles[out_idx].velocity = vel;
    pc.mega_particles[out_idx].is_active = 1;
}

// --- msl_barnes_hut.txt ---
#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.h"
#include "imex_math.h"

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#define SUBGROUPS_PER_WG (256 / SUBGROUP_SIZE)

struct MultiBvhNode {
    float min_x[SUBGROUP_SIZE]; float max_x[SUBGROUP_SIZE];
    float min_y[SUBGROUP_SIZE]; float max_y[SUBGROUP_SIZE];
    float min_z[SUBGROUP_SIZE]; float max_z[SUBGROUP_SIZE];
    uint  child_indices[SUBGROUP_SIZE]; uint metadata[SUBGROUP_SIZE];
    float masses[SUBGROUP_SIZE];
    float com_x[SUBGROUP_SIZE]; float com_y[SUBGROUP_SIZE]; float com_z[SUBGROUP_SIZE];
    uint  particle_start[SUBGROUP_SIZE]; uint particle_count[SUBGROUP_SIZE];
    uint2 valid_mask;
    uint  parent_idx;
    uint  pad;
    uint  permutations[8][SUBGROUP_SIZE];
};

struct Wrench { uint force_x; uint force_y; uint force_z; uint torque_x; uint torque_y; uint torque_z; };

struct PushConstants {
    device uint* particles;
    device MultiBvhNode* bvh;
    device uint* cluster_list;
    device Wrench* wrenches;
    uint num_clusters;
    float dt;
    float theta;
    float G;
    float softening_sq;
    uint root_node_idx;
    uint cluster_threshold;
};

inline void AtomicAddFloat(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(addr, &expected, as_type<uint>(as_type<float>(expected) + val), memory_order_relaxed, memory_order_relaxed)) {
    }
}

inline bool bvh_node_is_valid(uint2 valid_mask, uint lane_id) {
    if (lane_id < 32) return (valid_mask.x & (1u << lane_id)) != 0u;
    else return (valid_mask.y & (1u << (lane_id - 32))) != 0u;
}

inline bool bvh_is_leaf(uint meta) { return (meta & 0x80000000u) != 0u; }

[[kernel]]
void barnes_hut(
    constant PushConstants& pc [[buffer(0)]],
    uint3 gl_WorkGroupID [[threadgroup_position_in_grid]],
    uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint gl_SubgroupID [[simdgroup_index_in_threadgroup]]
) {
    uint cluster_job_idx = gl_WorkGroupID.x * SUBGROUPS_PER_WG + gl_SubgroupID;
    if (cluster_job_idx >= pc.num_clusters) return;

    threadgroup uint shared_stacks[SUBGROUPS_PER_WG][64];
    threadgroup uint shared_stack_ptrs[SUBGROUPS_PER_WG];

    uint target_node_idx = pc.cluster_list[cluster_job_idx];
    device MultiBvhNode& t_node = pc.bvh[target_node_idx];
    bool i_am_valid = bvh_node_is_valid(t_node.valid_mask, lane_id);
    uint my_p_idx = t_node.child_indices[lane_id];

    float3 my_pos = float3(0.0);
    float my_mass = 0.0;
    if (i_am_valid) {
        uint base = (my_p_idx / SUBGROUP_SIZE) * 10 * SUBGROUP_SIZE + (my_p_idx % SUBGROUP_SIZE);
        my_pos = float3(
            as_type<float>(pc.particles[base]),
            as_type<float>(pc.particles[base + 1 * SUBGROUP_SIZE]),
            as_type<float>(pc.particles[base + 2 * SUBGROUP_SIZE])
        );
        my_mass = as_type<float>(pc.particles[base + 6 * SUBGROUP_SIZE]);
    }

    float3 safe_pos = i_am_valid ? my_pos : float3(0.0);
    float3 min_pos = simd_min(i_am_valid ? my_pos : float3(1e20));
    float3 max_pos = simd_max(i_am_valid ? my_pos : float3(-1e20));
    float3 cluster_extents = max_pos - min_pos;
    float target_size = max(cluster_extents.x, max(cluster_extents.y, cluster_extents.z));
    float sum_mass = simd_sum(i_am_valid ? my_mass : 0.0);
    float3 target_com = simd_sum(safe_pos * my_mass) / max(sum_mass, 1e-6f);

    float3 my_acc = float3(0.0);
    if (lane_id == 0) { 
        shared_stacks[gl_SubgroupID][0] = pc.root_node_idx; 
        shared_stack_ptrs[gl_SubgroupID] = 1; 
    }

    while (true) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint stack_ptr = shared_stack_ptrs[gl_SubgroupID]; 
        if (stack_ptr == 0) break;
        
        stack_ptr--;
        uint source_node_idx = shared_stacks[gl_SubgroupID][stack_ptr]; 
        if (lane_id == 0) shared_stack_ptrs[gl_SubgroupID] = stack_ptr;

        device MultiBvhNode& s_node = pc.bvh[source_node_idx];
        bool s_valid = bvh_node_is_valid(s_node.valid_mask, lane_id);
        bool s_is_leaf = bvh_is_leaf(s_node.metadata[lane_id]);

        float3 s_com = float3(s_node.com_x[lane_id], s_node.com_y[lane_id], s_node.com_z[lane_id]);
        float s_mass = s_node.masses[lane_id];
        uint s_idx = s_node.child_indices[lane_id];
        uint s_start = s_node.particle_start[lane_id];
        uint s_count = s_node.particle_count[lane_id];

        float3 s_extents = float3(
            s_node.max_x[lane_id] - s_node.min_x[lane_id],
            s_node.max_y[lane_id] - s_node.min_y[lane_id],
            s_node.max_z[lane_id] - s_node.min_z[lane_id]
        );
        float s_size = max(s_extents.x, max(s_extents.y, s_extents.z));

        bool pass_mac = ((s_size + target_size) / max(length(s_com - target_com), 1e-6f)) < pc.theta;
        bool pass_lod_thresh = (s_count <= pc.cluster_threshold) && !((my_p_idx >= s_start) && (my_p_idx < s_start + s_count));
        bool action_accumulate = s_valid && (pass_mac || pass_lod_thresh || s_is_leaf);
        bool action_traverse = s_valid && !action_accumulate;

        ulong acc_ballot = simd_ballot(action_accumulate);
        while (acc_ballot != 0) {
            uint src_lane = __builtin_ctzll(acc_ballot);
            acc_ballot &= ~(1ul << src_lane); 
            
            if (i_am_valid) {
                float3 k_com = float3(simd_broadcast(s_com.x, src_lane), simd_broadcast(s_com.y, src_lane), simd_broadcast(s_com.z, src_lane));
                float k_mass = simd_broadcast(s_mass, src_lane); 
                uint k_idx = simd_broadcast(s_idx, src_lane); 
                bool k_leaf = simd_broadcast(s_is_leaf, src_lane);

                if (!(k_leaf && my_p_idx == k_idx)) {
                    float3 p_dir = k_com - my_pos; 
                    float p_dist_sq = dot(p_dir, p_dir);
                    my_acc += (p_dir / max(sqrt(p_dist_sq), 1e-6f)) * ((pc.G * k_mass) / (p_dist_sq + pc.softening_sq));
                }
            }
        }

        uint prefix_count = simd_prefix_exclusive_sum(action_traverse ? 1 : 0);
        if (action_traverse) {
            shared_stacks[gl_SubgroupID][stack_ptr + prefix_count] = s_idx;
        }
        
        uint total_trav = simd_sum(action_traverse ? 1 : 0);
        if (lane_id == 0) {
            shared_stack_ptrs[gl_SubgroupID] = stack_ptr + total_trav;
        }
    }

    if (i_am_valid) {
        float3 g_f = my_acc * my_mass;
        device Wrench& w = pc.wrenches[my_p_idx];
        AtomicAddFloat((device atomic_uint*)&w.force_x, g_f.x);
        AtomicAddFloat((device atomic_uint*)&w.force_y, g_f.y);
        AtomicAddFloat((device atomic_uint*)&w.force_z, g_f.z);
    }
}

// --- msl_emit_particles.txt ---
#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.msl"

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#ifdef KERNEL_emit_particles
struct PushConstants {
    device uint* particles;
    device uint* candidates;
    device MultiBvhNode* bvh;
    device atomic_uint* counter;
    uint root_index;
    uint num_candidates;
    uint2 pad;
    float3 sun_pos;
};
#endif

#ifndef INTERSECT_RAY_AABB_DEFINED
#define INTERSECT_RAY_AABB_DEFINED
bool intersectRayAABB(float3 rO, float3 rD, float3 invD, float3 mi, float3 mx, float max_t) {
    float3 t0 = (mi - rO) * invD;
    float3 t1 = (mx - rO) * invD;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float tnear = max(max(tmin.x, tmin.y), tmin.z);
    float tfar = min(min(tmax.x, tmax.y), tmax.z);
    return tnear <= tfar && tfar > 0.0f && tnear < max_t;
}
#endif

[[kernel]]
void emit_particles(
    uint3 gl_GlobalInvocationID [[thread_position_in_grid]],
    constant PushConstants& pc [[buffer(0)]]
) {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.num_candidates) return;
    
    uint stride = 10 * SUBGROUP_SIZE;
    uint base = (gid / SUBGROUP_SIZE) * stride + (gid % SUBGROUP_SIZE);

    float pos_x = as_type<float>(pc.candidates[base]);
    float pos_y = as_type<float>(pc.candidates[base + SUBGROUP_SIZE]);
    float pos_z = as_type<float>(pc.candidates[base + 2 * SUBGROUP_SIZE]);
    float3 pos = float3(pos_x, pos_y, pos_z);
    
    float3 dir = pc.sun_pos - pos;
    float dist = length(dir);
    if (dist < 1e-5f) return;
    dir /= dist;
    float3 invDir = 1.0f / dir;

    bool occluded = false;
    uint stack[64];
    int stackPtr = 0;
    if (pc.root_index != 0xFFFFFFFFu) stack[stackPtr++] = pc.root_index;

    while(stackPtr > 0 && !occluded) {
        uint node = stack[--stackPtr];
        for (uint i = 0; i < SUBGROUP_SIZE; ++i) {
            if (!bvh_node_is_valid(pc.bvh[node].valid_mask, i)) continue;
            
            float3 mn = float3(pc.bvh[node].min_x[i], pc.bvh[node].min_y[i], pc.bvh[node].min_z[i]);
            float3 mx = float3(pc.bvh[node].max_x[i], pc.bvh[node].max_y[i], pc.bvh[node].max_z[i]);

            if (intersectRayAABB(pos + dir * 0.1f, dir, invDir, mn, mx, dist)) {
                if (bvh_is_leaf(pc.bvh[node].metadata[i])) { 
                    occluded = true; 
                    break; 
                }
                else if (bvh_get_index(pc.bvh[node].metadata[i]) != 0xFFFFFFFFu) {
                    stack[stackPtr++] = bvh_get_index(pc.bvh[node].metadata[i]);
                }
            }
        }
    }

    if (!occluded) {
        uint out_idx = atomic_fetch_add_explicit(&pc.counter[0], 1, memory_order_relaxed);
        uint out_base = (out_idx / SUBGROUP_SIZE) * stride + (out_idx % SUBGROUP_SIZE);
        for (int i = 0; i < 10; ++i) {
            pc.particles[out_base + i * SUBGROUP_SIZE] = pc.candidates[base + i * SUBGROUP_SIZE];
        }
    }
}


