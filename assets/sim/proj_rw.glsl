// ── dust_project_packet on the rows : the candidate's words read at each use (QF,
// its AoS record), the screen packet's fields written to its slot as they are computed (PSET) and
// read back from it (PF); the slot's later words hold the phase intermediates until their final
// value (17-22 jx, jy until kx, ky; 10 deepK until depthAu; 9 detG until det). The same
// expressions in the same order as dust_project_packet (dust_common.glsl); `precise` on the
// locals whose values were in its NoContraction cone (through P.cov / P.kx / P.ky / P.eye)
layout(buffer_reference, std430, buffer_reference_align = 4) buffer DustRowR { uint v[]; };
#define DUST_RW_BUF DustRowR
// the project stage as one pass (no define) or as three (-DDUST_PROJ_PASS=1, 2, 3; the slot the carrier)
#ifndef DUST_PROJ_PASS
#define DUST_PROJ_PASS 0
#endif
layout(buffer_reference, std430, buffer_reference_align = 4) buffer DustRowW { uint w[]; };
uvec2 dust_rw_qa, dust_rw_pa, dust_rw_pr;
uint dust_rw_rec = 0u;
uvec3 dust_rw_c;
uvec2 dust_word_addr(uint word) { uvec2 a = uvec2(pc.pyramid); uint o = word * 4u; uint lo = a.x + o; return uvec2(lo, a.y + (lo < o ? 1u : 0u)); }
// loads through functions: glslang's precise propagation never enters a call, while a buffer
// reference load inside a precise cone would mark every buffer reference store of the shader
uint dust_rw_paw, dust_rw_qaw;
uint dust_rw_qw(uint k) { return DUST_RW_BUF(dust_rw_qa).v[k]; }
uint dust_rw_pw(uint k) { return DUST_RW_BUF(dust_rw_pr).v[k]; }
float dust_rw_recf(uint k) { return uintBitsToFloat(pc.pyramid.v[dust_rw_rec + k * DUST_SQUEUE_CAPACITY]); }
#if DUST_PROJ_PASS == 3
shared uint dust_rw_lq[64], dust_rw_lp[64], dust_rw_lr[64];
#define DUST_O_JRL() { memoryBarrierShared(); dust_rw_qaw = dust_rw_lq[gl_LocalInvocationID.x]; dust_rw_paw = dust_rw_lp[gl_LocalInvocationID.x]; dust_rw_rec = dust_rw_lr[gl_LocalInvocationID.x]; dust_rw_qa = dust_word_addr(dust_rw_qaw); dust_rw_pr = dust_word_addr(dust_rw_paw); dust_rw_pa = dust_rw_pr; }
#else
#define DUST_O_JRL()
#endif
#define QW(k) dust_rw_qw(k)
#define QF(k) uintBitsToFloat(QW(k))
#define PW(k) dust_rw_pw(k)
#define PF(k) uintBitsToFloat(PW(k))
#define PSET(k, f) pc.pyramid.v[dust_rw_paw + (k)] = floatBitsToUint(f);
#define RECF(k) dust_rw_recf(k)
#define r_mean vec3(QF(0u), QF(1u), QF(2u))
#define r_d vec3(QF(9u), QF(10u), QF(11u))
#define r_flux QF(12u)
#define r_rho vec2(QF(13u), QF(14u))
#define r_Sm mat3(QF(3u), QF(4u), QF(5u), QF(4u), QF(6u), QF(7u), QF(5u), QF(7u), QF(8u))
#define r_jx vec3(PF(17u), PF(18u), PF(19u))
#define r_jy vec3(PF(20u), PF(21u), PF(22u))
float dust_rw_pyrf(uint w) { return float(pc.pyramid.v[w]); }
#define r_W dust_rw_pyrf(DUST_PYR_WIDTH)
#define r_H dust_rw_pyrf(DUST_PYR_HEIGHT)
// phase A: the Jacobian rows jx, jy (slot words 17-22) and deepK (word 10); mu, rho final
bool dust_project_rw_a() {
    DUST_O_JRL()
    precise vec4 clip = pc.mvp * vec4(r_mean, 1.0);
    if (!(clip.w > 0.0)) return false;
    PSET(12u, r_rho.x) PSET(13u, r_rho.y)
    vec3 r0 = vec3(pc.mvp[0][0], pc.mvp[1][0], pc.mvp[2][0]);
    vec3 r1 = vec3(pc.mvp[0][1], pc.mvp[1][1], pc.mvp[2][1]);
    vec3 r3 = vec3(pc.mvp[0][3], pc.mvp[1][3], pc.mvp[2][3]);
    precise float wEnd = dot(r3, r_mean + r_d) + pc.mvp[3][3];
    bool shallow = abs(wEnd - clip.w) <= DUST_SPLIT_EPS * clip.w;
    precise float jAt = (shallow && ((dust_pyr_split_word >> 23u) & 1u) == 0u) ? 0.5 : 0.0;
    float deepK = 1.0;
    if (!shallow && dust_deep_chord && clip.w > 0.0 && wEnd > 0.0) {
        vec2 f = dust_deep_chord_factors(clip.w, wEnd, r_rho);
        jAt = f.x;
        deepK = f.y;
    }
    PSET(10u, deepK)
    precise vec4 clipJ = pc.mvp * vec4(r_mean + jAt * r_d, 1.0);
    if (!(clipJ.w > 0.0)) clipJ = clip;
    precise vec2 ndcJ = clipJ.xy / clipJ.w;
    precise vec3 jx = (r0 - ndcJ.x * r3) * (0.5 * r_W / clipJ.w);
    precise vec3 jy = (r1 - ndcJ.y * r3) * (0.5 * r_H / clipJ.w);
    PSET(17u, jx.x) PSET(18u, jx.y) PSET(19u, jx.z)
    PSET(20u, jy.x) PSET(21u, jy.y) PSET(22u, jy.z)
    return true;
}
// phases B-E: cov, seg, kx/ky, the amplitude and the gates
bool dust_project_rw_b() {
    DUST_O_JRL()
    {
        precise vec3 sx = r_Sm * r_jx, sy = r_Sm * r_jy;
        precise vec3 cov_ = vec3(dot(r_jx, sx) + DUST_PIXEL_FILTER_VAR, dot(r_jx, sy), dot(r_jy, sy) + DUST_PIXEL_FILTER_VAR);
        PSET(2u, cov_.x) PSET(3u, cov_.y) PSET(4u, cov_.z)
        dust_rw_c = floatBitsToUint(cov_);
    }
    {
        vec2 seg_ = vec2(dot(r_jx, r_d), dot(r_jy, r_d));
        PSET(5u, seg_.x) PSET(6u, seg_.y)
    }
    DUST_O_JRL()
    {
        float g00 = dot(r_jx, r_jx), g01 = dot(r_jx, r_jy), g11 = dot(r_jy, r_jy);
        precise float detG = max(g00 * g11 - g01 * g01, 0.0);
        precise vec3 kx_, ky_;
        if (detG > 1e-30) {
            kx_ = (r_jx * g11 - r_jy * g01) / detG;
            ky_ = (r_jy * g00 - r_jx * g01) / detG;
        } else {
            kx_ = vec3(0.0);
            ky_ = vec3(0.0);
        }
        PSET(9u, detG)
        PSET(17u, kx_.x) PSET(18u, kx_.y) PSET(19u, kx_.z)
        PSET(20u, ky_.x) PSET(21u, ky_.y) PSET(22u, ky_.z)
    }
#if DUST_PROJ_PASS
    return true;
}
bool dust_project_rw_b2() {
    dust_rw_c = uvec3(PW(2u), PW(3u), PW(4u));
#endif
    DUST_O_JRL()
    precise float half_ = 0.5 * (uintBitsToFloat(dust_rw_c.x) + uintBitsToFloat(dust_rw_c.z));
    precise float dd = sqrt((0.5 * (uintBitsToFloat(dust_rw_c.x) - uintBitsToFloat(dust_rw_c.z))) * (0.5 * (uintBitsToFloat(dust_rw_c.x) - uintBitsToFloat(dust_rw_c.z))) + uintBitsToFloat(dust_rw_c.y) * uintBitsToFloat(dust_rw_c.y));
    float lamMax = max(half_ + dd, DUST_PIXEL_FILTER_VAR);
    precise float detRaw = uintBitsToFloat(dust_rw_c.x) * uintBitsToFloat(dust_rw_c.z) - uintBitsToFloat(dust_rw_c.y) * uintBitsToFloat(dust_rw_c.y);
    float detC = max(max(detRaw, DUST_DET_ANISO_FLOOR * lamMax * lamMax), 1e-30);
    DUST_O_JRL()
    float amp_ = RECF(13u) * r_flux * PF(10u) * sqrt(PF(9u)) / (2.0 * DUST_PI_F32 * sqrt(detC));
    PSET(9u, detC)
    {
        bool perspective;
        vec3 fwd = dust_view_forward(pc.mvp, pc.eyeLocal.xyz, perspective);
        if (dot(fwd, fwd) > 0.0) {
            float depth = dot(r_mean + 0.5 * r_d - pc.eyeLocal.xyz, fwd);
            float sigmaV = sqrt(max(dust_var_along(float[6](QF(3u), QF(4u), QF(5u), QF(6u), QF(7u), QF(8u)), fwd), 0.0));
            if (sigmaV > 0.0) amp_ *= 0.5 * (1.0 + dust_erf(depth / (sigmaV * 1.4142135623730951)));
            else if (depth < 0.0) amp_ = 0.0;
        }
    }
    PSET(7u, amp_)
    if (!(PF(7u) > 0.0) || isinf(PF(7u))) return false;
    DUST_O_JRL()
    float sigmaMin_ = sqrt(max(detC / lamMax, DUST_PIXEL_FILTER_VAR * 0.5));
    PSET(8u, sigmaMin_)
    // mu here, next to its one arithmetic use (mu + seg): the product stays in the add's
    // contraction context as in dust_project_packet; clip again with its precise decoration
    precise vec4 clipB = pc.mvp * vec4(r_mean, 1.0);
    vec2 ndc = clipB.xy / clipB.w;
    vec2 mu_ = (ndc * 0.5 + 0.5) * vec2(r_W, r_H);
    PSET(0u, mu_.x) PSET(1u, mu_.y)
    float ex = DUST_SPLAT_SIGMAS * sqrt(PF(2u)), ey = DUST_SPLAT_SIGMAS * sqrt(PF(4u));
    float x0 = min(mu_.x, mu_.x + PF(5u)) - ex, x1 = max(mu_.x, mu_.x + PF(5u)) + ex;
    float y0 = min(mu_.y, mu_.y + PF(6u)) - ey, y1 = max(mu_.y, mu_.y + PF(6u)) + ey;
    if (x1 < 0.0 || y1 < 0.0 || x0 > r_W || y0 > r_H || !(x0 <= x1) || !(y0 <= y1)) return false;
    float depthAu_ = length(r_mean - pc.eyeLocal.xyz) / DUST_AU_M;
    PSET(10u, depthAu_)
    if (any(isnan(mu_)) || any(isinf(mu_)) || any(isnan(vec3(PF(2u), PF(3u), PF(4u)))) || any(isinf(vec3(PF(2u), PF(3u), PF(4u))))
        || any(isnan(vec2(PF(5u), PF(6u)))) || any(isinf(vec2(PF(5u), PF(6u)))) || isnan(PF(8u)) || isnan(PF(10u))) return false;
    return true;
}
// phase F (an on-screen packet only: an off-screen slot is never read): the photometry words
void dust_project_rw_f() {
    DUST_O_JRL()
    bool perspective;
    vec3 fwd = dust_view_forward(pc.mvp, pc.eyeLocal.xyz, perspective);
    precise vec4 eye_ = (perspective || dot(fwd, fwd) == 0.0) ? vec4(pc.eyeLocal.xyz, 1.0) : vec4(-fwd, 0.0);
    PSET(27u, eye_.x) PSET(28u, eye_.y) PSET(29u, eye_.z) PSET(30u, eye_.w)
    PSET(14u, QF(0u)) PSET(15u, QF(1u)) PSET(16u, QF(2u))
    PSET(23u, RECF(28u)) PSET(24u, RECF(29u)) PSET(25u, RECF(30u))
    PSET(26u, RECF(31u))
    PSET(31u, pc.albedo)
    float wMean_ = 1.0;
    if (pc.albedo > 0.0) {
        precise vec3 toEye = eye_.xyz - eye_.w * r_mean;
        wMean_ = dust_radiance_weight_o(uvec2(pc.pyramid), pc.albedo, vec3(PF(23u), PF(24u), PF(25u)), toEye, PF(26u));
    }
    PSET(32u, wMean_)
}
