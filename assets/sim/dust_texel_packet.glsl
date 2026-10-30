// @assets/sim/dust_texel_packet.glsl
//
// Included by a texel-lane consumer after it defines DUST_SP_F(k) / DUST_SP_U(k), which may
// read the buffer at `pyr` (the functions' first parameter: the address, uvec2(pc.pyramid); no
// buffer reference is a parameter of a helper: the Adreno 619 compiler puts one in private memory).
// ---- the screen packet's texel functions with the packet's words read at their use -----------
// dust_capsule_tau and dust_texel_weight take the packet as a by-value struct, so every field is
// loaded before the body starts (34 words in flight: the texel lane's peak). These copies have the
// same statements, the same expressions in the same order; only each `P.field` is the load of that
// word at its use, through the includer's DUST_SP_F(k) (the float bits of word k of the packet as
// dust_queue_put_ewa lays it out: mu 0-1, cov 2-4, seg 5-6, amp 7, sigmaMin 8, det 9, depthAu 10,
// open 11 (bits), rho 12-13, mean 14-16, kx 17-19, ky 20-22, sun 23-25, rhAu 26, eye 27-30,
// albedo 31, wMean 32) and DUST_SP_U(k) (the bits). Keep them in step with dust_common.glsl.
#define DUST_SP_V2(k) vec2(DUST_SP_F(k), DUST_SP_F((k) + 1u))
#define DUST_SP_V3(k) vec3(DUST_SP_F(k), DUST_SP_F((k) + 1u), DUST_SP_F((k) + 2u))
#define DUST_SP_OPEN_START ((DUST_SP_U(11u) & 1u) != 0u)
#define DUST_SP_OPEN_END ((DUST_SP_U(11u) & 2u) != 0u)
// dust_capsule_tau(P, x)
float dust_capsule_tau_at(uvec2 pyr, vec2 x) {
    float det = DUST_SP_F(9u);
    if (!(det > 0.0)) return 0.0;
    float m00 = DUST_SP_F(4u) / det, m01 = -DUST_SP_F(3u) / det, m11 = DUST_SP_F(2u) / det;
    vec2 e = x - DUST_SP_V2(0u);
    float c = m00 * e.x * e.x + 2.0 * m01 * e.x * e.y + m11 * e.y * e.y;
    vec2 d = DUST_SP_V2(5u);
    float a = m00 * d.x * d.x + 2.0 * m01 * d.x * d.y + m11 * d.y * d.y;
    if (!(a > 1e-6)) return DUST_SP_F(7u) * exp(-0.5 * c);
    float b = m00 * d.x * e.x + m01 * (d.x * e.y + d.y * e.x) + m11 * d.y * e.y;
    float sa = sqrt(a);
    float u0 = b / a;
    // an open end continues into the neighbouring bin: no Gaussian cap there
    float phiEnd = DUST_SP_OPEN_END ? 1.0 : 0.5 * (1.0 + dust_erf((1.0 - u0) * sa * 0.70710678));
    float phiStart = DUST_SP_OPEN_START ? 0.0 : 0.5 * (1.0 + dust_erf(-u0 * sa * 0.70710678));
    float span = phiEnd - phiStart;
    precise vec2 r = e - u0 * d;
    precise float q = max(m00 * r.x * r.x + 2.0 * m01 * r.x * r.y + m11 * r.y * r.y, 0.0);
    float i0 = sqrt(2.0 * DUST_PI_F32 / a) * span;
    float weight;
    if (DUST_SP_F(13u) == DUST_SP_F(12u)) {
        weight = DUST_SP_F(12u) * i0;
    } else {
        float i1;
        if (!DUST_SP_OPEN_START && !DUST_SP_OPEN_END) {
            float z1 = -0.5 * a * u0 * u0;
            float z2 = -0.5 * a * (1.0 - u0) * (1.0 - u0);
            float dz = z2 - z1;
            float diff = abs(dz) < 1.0 ? -exp(z1) * dust_expm1(dz) : exp(z1) - exp(z2);
            i1 = u0 * i0 + diff / a;
        } else {
            float eStart = DUST_SP_OPEN_START ? 0.0 : exp(-0.5 * a * u0 * u0);
            float eEnd = DUST_SP_OPEN_END ? 0.0 : exp(-0.5 * a * (1.0 - u0) * (1.0 - u0));
            i1 = u0 * i0 + (eStart - eEnd) / a;
        }
        weight = max(DUST_SP_F(12u) * i0 + (DUST_SP_F(13u) - DUST_SP_F(12u)) * i1, 0.0);
    }
    return DUST_SP_F(7u) * exp(-0.5 * q) * weight;
}
// dust_texel_weight(P, x)
float dust_texel_weight_at(uvec2 pyr, vec2 x) {
    vec2 dxy = x - DUST_SP_V2(0u);
    vec3 d = DUST_SP_V3(17u) * dxy.x + DUST_SP_V3(20u) * dxy.y;
    vec3 pos = DUST_SP_V3(14u) + d;
    vec3 rVec = DUST_SP_V3(23u) * (-DUST_SP_F(26u) * DUST_AU_M) + d;
    float rn = max(length(rVec), 1.0);
    return dust_t_radiance_weight(pyr, DUST_SP_F(31u), rVec * (-1.0 / rn),
                                  DUST_SP_V3(27u) - DUST_SP_F(30u) * pos, rn / DUST_AU_M);
}
#undef DUST_SP_V2
#undef DUST_SP_V3
#undef DUST_SP_OPEN_START
#undef DUST_SP_OPEN_END
