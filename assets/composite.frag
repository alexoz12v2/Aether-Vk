#version 450 core
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

// Compositing fragment shader: merges macro and micro layers by linearizing
// their depths and picking the nearer fragment per pixel, then lays the dust over the result.
//
// Uses subpass input attachments for zero-copy reads on tile-based GPUs.
// MRT output[1] (finalGlobalDepth) carries (layer_index_f32, local_distance)
// of the winning fragment for CPU-side UI occlusion queries.
//
// Dust (v4): the view's splat pyramid (dust_splat.comp, dust::PyramidLayout), read through its
// device address: level l holds texels of 2^l px with (tau, tau*r, tau*g, tau*b) in fixed point
// (dustUnit per count, exposure-scaled optical depth x px^2) and the nearest dust depth (AU, f32
// bits). The optical depth at a pixel is the sum over the levels of the bilinear texel value
// divided by the texel area; the display stretch (asinh, dust::display_stretch) maps it relative
// to the view's white point; dust behind an opaque surface of the winning layer is dropped.

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput macroColor;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput macroDepth;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput macroGlobalDepth;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput microColor;
layout(input_attachment_index = 4, set = 0, binding = 4) uniform subpassInput microDepth;
layout(input_attachment_index = 5, set = 0, binding = 5) uniform subpassInput microGlobalDepth;

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer DustPyramid {
    uint v[];
};

const uint  DUST_PYR_TAU_MAX     = 3u;
const uint  DUST_PYR_LEVEL_MASK  = 7u;
const uint  DUST_PYR_TABLE       = 16u;
const uint  DUST_PYR_TEXEL_WORDS = 5u;
const float DUST_COLOR_SCALE     = 16.0; // the colour words over the tau counts (dust::DUST_COLOR_SCALE)

layout(push_constant, std430) uniform CompositePush {
    float macroNear;        // 0
    float macroFar;         // 4
    float microNear;        // 8
    float microFar;         // 12
    float macroScale;       // 16
    float microScale;       // 20
    uint  isOrthographic;   // 24: 1 = orthographic (linear depth), 0 = perspective
    float dustSoftening;    // 28: asinh stretch softening s (dust::display_stretch), 0 = linear
    float dustBlackPoint;   // 32: dust::DUST_BLACK_POINT (0 with AETHERVK_DUST_EXPOSURE=fixed)
    uint  layerRefLo;       // 36: LayerRefs address, low word (0 = no references)
    DustPyramid dustPyramid;// 40: the view's pyramid (0 = no dust)
    float dustWhite;        // 48: white point (exposure-scaled tau per px^2)
    uint  dustLevels;       // 52
    float dustUnit;         // 56: tau x px^2 per count
    uint  layerRefHi;       // 60: LayerRefs address, high word
    vec4  layerUnitAu[2];   // 64: AU per global-depth unit of layers 0..7
    vec2  dustNucleusPx;    // 96: the nucleus' framebuffer pixel (y down) of the 1/rho normalisation
    float dustRhoMin;       // 104: dust::DUST_RHO_MIN_PX, the normalisation saturates inside it
    float dustRhoRef;       // 108: dust::DUST_RHO_REF_PX, factor 1 there
    float dustWhiteScale;   // 112: exposure correction 2^-EV on the white (exactly 1 at EV 0)
    uint  dustRhoNorm;      // 116: 1 = coma 1/rho normalisation (dust::DUST_VIEW_RHO_NORM)
    uvec2 _pad;             // 120 (total 128)
};

// Distance reference of each depth layer (layer units, 0 = absolute): micro meshes write their
// global depth as |c + u| - refDist[layer] (rteDistanceOffset), restored below before the dust.
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer LayerRefs {
    float refDist[8];
};

// Mirror: dust::display_stretch. Opacity of a dust optical depth `tau` relative to the view's
// white point: 0 up to the black point b, linear below `s`, logarithmic above, 1 at the white
// point, so dust 1e4 fainter than the brightest still shows.
// highlight roll-off of the dust colour (mirror of dust::DUST_HIGHLIGHT_START / _WHITE)
const float DUST_HIGHLIGHT_START = 0.8;
const float DUST_HIGHLIGHT_WHITE = 0.7;
float displayStretch(float tau, float b, float s) {
    tau -= max(b, 0.0);
    if (!(tau > 0.0)) return 0.0;
    if (!(s > 0.0)) return min(tau, 1.0);
    return min(asinh(tau / s) / asinh(1.0 / s), 1.0);
}

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 finalGlobalDepth; // (layer_index_f32, local_distance) of winner

// Linearize a reverse-Z depth value to view-space distance.
// Reverse-Z:  depth = 1.0 -> at near plane,  depth = 0.0 -> at far plane.
// Returns the physical distance from the camera.
float linearizeReverseZ(float d, float near, float far) {
    if (isOrthographic != 0u) {
        // Orthographic reverse-Z: d = (far - dist) / (far - near)  (linear; near may be < 0)
        return far - d * (far - near);
    }
    // Perspective reverse-Z: d = near * (far - dist) / (dist * (far - near))
    return (near * far) / mix(near, far, d);
}

void main() {
    vec4 cMacro = subpassLoad(macroColor);
    float dMacro = subpassLoad(macroDepth).r;
    vec2 gdMacro = subpassLoad(macroGlobalDepth).rg;
    vec4 cMicro = subpassLoad(microColor);
    float dMicro = subpassLoad(microDepth).r;
    vec2 gdMicro = subpassLoad(microGlobalDepth).rg;

    // Linearize both depths to physical distance (in layer-local scale * macroScale/microScale = AU)
    float distMacro = linearizeReverseZ(dMacro, macroNear, macroFar) * macroScale;
    float distMicro = linearizeReverseZ(dMicro, microNear, microFar) * microScale;

    // Pick the fragment that is nearer to the camera.
    // When a layer has no content at a pixel, its depth is 0.0 (reverse-Z clear value).
    // However, some pipelines (e.g. sphere gizmo wireframes) write color but NOT depth
    // (NO_DEPTH_WRITE). For those pixels, depth stays at clear value but color is valid.
    if (dMicro == 0.0 && cMicro.a == 0.0) {
        // Micro layer is truly empty (no color, no depth), use Macro
        outColor = cMacro;
        finalGlobalDepth = gdMacro;
    } else if (dMicro == 0.0 && cMicro.a > 0.0) {
        // Micro layer has color but no depth (e.g. wireframe gizmo, or depth was cleared
        // between layers) — blend over macro. Use gdMicro since micro content exists here.
        // The micro target is cleared to transparent black and blended "over" (or additively for
        // dust), so its rgb is already premultiplied by alpha: premultiplied over, not mix()
        // (which would multiply by alpha twice and wipe out faint content).
        outColor = vec4(cMicro.rgb + cMacro.rgb * (1.0 - cMicro.a), max(cMacro.a, cMicro.a));
        finalGlobalDepth = gdMicro; // micro content present — prefer micro global depth
    } else if (dMacro == 0.0) {
        // Macro layer is empty, use Micro
        outColor = cMicro;
        finalGlobalDepth = gdMicro;
    } else if (gdMacro.x >= 15.5 && gdMicro.x >= 0.5
               && int(gdMicro.x + 0.5) == int(gdMacro.x + 0.5) - 16) {
        // An anchored macro track piece (tag 16 + m) over a mesh of the micro layer m it is
        // anchored to: both global depths are offsets from the same host f64 reference
        // (|o_m| = D_m * layerUnitAu[m]), formed from small f32 values only, ~1 m at any distance
        // (composite precision plan §2.4). The linearised depths above are ~60 km * D/AU coarse.
        int m = clamp(int(gdMicro.x + 0.5), 0, 7);
        if (gdMacro.y <= gdMicro.y * layerUnitAu[m >> 2][m & 3]) {
            outColor = cMacro;
            finalGlobalDepth = gdMacro;
        } else {
            outColor = cMicro;
            finalGlobalDepth = gdMicro;
        }
    } else if (distMacro <= distMicro) {
        outColor = cMacro;
        finalGlobalDepth = gdMacro;
    } else {
        outColor = cMicro;
        finalGlobalDepth = gdMicro;
    }

    // Global depth back to its meaning (layer, absolute local distance) or the sentinel: anchored
    // tracks are not occluders / pick targets (as before), micro offsets get their reference back
    // (f32: the ~ulp(D) granularity the readback always had).
    if (finalGlobalDepth.x >= 15.5) {
        finalGlobalDepth = vec2(-1.0, -1.0);
    } else if (finalGlobalDepth.x >= 0.5 && (layerRefLo | layerRefHi) != 0u) {
        LayerRefs refs = LayerRefs(uvec2(layerRefLo, layerRefHi));
        int li = clamp(int(finalGlobalDepth.x + 0.5), 0, 7);
        finalGlobalDepth.y += refs.refDist[li];
    }

    // Dust last, over everything. The pyramid's levels summed at this pixel (mirror of
    // dust::composite_sample); dust whose nearest packet lies behind the winning opaque surface is
    // hidden (per-texel occlusion by the nearest dust depth).
    if (uvec2(dustPyramid) != uvec2(0u) && dustLevels > 0u) {
        uint mask = dustPyramid.v[DUST_PYR_LEVEL_MASK];
        float tau = 0.0;
        vec3 rgb = vec3(0.0);
        float depthMin = 3.4e38;
        for (uint l = 0u; l < dustLevels; ++l) {
            if ((mask & (1u << l)) == 0u) continue;
            uint off = dustPyramid.v[DUST_PYR_TABLE + 3u * l];
            uint w = dustPyramid.v[DUST_PYR_TABLE + 3u * l + 1u];
            uint h = dustPyramid.v[DUST_PYR_TABLE + 3u * l + 2u];
            float s = float(1u << l);
            vec2 f = gl_FragCoord.xy / s - 0.5;
            vec2 f0 = floor(f);
            vec2 a = f - f0;
            float scale = dustUnit / (s * s);
            for (uint k = 0u; k < 4u; ++k) {
                vec2 d = vec2(float(k & 1u), float(k >> 1u));
                float wgt = (d.x > 0.5 ? a.x : 1.0 - a.x) * (d.y > 0.5 ? a.y : 1.0 - a.y);
                if (!(wgt > 0.0)) continue;
                uint tx = uint(clamp(f0.x + d.x, 0.0, float(w - 1u)));
                uint ty = uint(clamp(f0.y + d.y, 0.0, float(h - 1u)));
                uint i = off + (ty * w + tx) * DUST_PYR_TEXEL_WORDS;
                uint c = dustPyramid.v[i];
                if (c == 0u) continue;
                tau += wgt * float(c) * scale;
                // the colour words are DUST_COLOR_SCALE times the tau counts (dust::DUST_COLOR_SCALE)
                rgb += wgt * vec3(float(dustPyramid.v[i + 1u]), float(dustPyramid.v[i + 2u]), float(dustPyramid.v[i + 3u])) * (scale / DUST_COLOR_SCALE);
                float dz = uintBitsToFloat(dustPyramid.v[i + 4u]);
                depthMin = min(depthMin, dz);
            }
        }
        if (tau > 0.0) {
            // occlusion by the winning opaque surface (global depth: layer index, distance in the
            // layer's units); the sentinel (-1, -1) means no opaque surface
            vec2 gd = finalGlobalDepth;
            bool occluded = false;
            if (gd.x >= 0.0 && gd.y >= 0.0) {
                int li = clamp(int(gd.x + 0.5), 0, 7);
                float unitAu = layerUnitAu[li >> 2][li & 3];
                occluded = depthMin > gd.y * unitAu;
            }
            if (!occluded) {
                float white = dustWhite > 0.0 ? dustWhite : 1.0;
                // the view options (mirror of dust::display_stretch_view): the coma's 1/rho
                // normalisation (the optical depth times the projected distance from the
                // nucleus, saturated inside dustRhoMin) and the exposure correction on the
                // white; off and EV 0 leave tau and white untouched
                // (the colour words scale with tau: the hue rgb/tau stays the stream colour)
                if (dustRhoNorm != 0u) {
                    float k = max(length(gl_FragCoord.xy - dustNucleusPx), dustRhoMin) / dustRhoRef;
                    tau *= k;
                    rgb *= k;
                }
                white *= dustWhiteScale;
                float a = displayStretch(tau / white, dustBlackPoint, dustSoftening);
                if (a > 0.0) {
                    vec3 hue = rgb / max(tau, 1e-30);
                    // overexposure rolls the colour off towards white (dust::display_color): a
                    // dense coma reads as bright, not as a flat disc of the stream colour
                    hue = mix(hue, vec3(1.0), DUST_HIGHLIGHT_WHITE * smoothstep(DUST_HIGHLIGHT_START, 1.0, a));
                    outColor = vec4(hue * a + outColor.rgb * (1.0 - a), max(outColor.a, a));
                }
            }
        }
    }
}
