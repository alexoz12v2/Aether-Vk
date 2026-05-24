#version 450 core
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_nonuniform_qualifier : require

struct UiElement {
    vec4 bounds; vec4 clipRect;
    vec4 colorStart; vec4 colorEnd;
    vec4 colorBorder; vec4 colorShadow;
    vec4 borderRadius; vec4 shadowParams;
    vec2 gradientDir; float borderWidth; uint textureId;
    uint flags; float opacity; float rotation; uint _pad;
};

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ElementArray {
    UiElement elements[];
};

layout(push_constant, std430) uniform PushConstants {
    ElementArray elementsPtr;
    mat4 viewProj;
    vec2 viewportSize;
} pc;

layout(location = 0) in vec2 vLocalPos;
layout(location = 1) in vec2 vHalfSize;
layout(location = 2) in flat uint vElementId;
layout(location = 3) in vec2 vUV;
layout(location = 4) in vec2 vPixelPos;

layout(set = 0, binding = 0) uniform sampler2D bindlessTextures[];

layout(location = 0) out vec4 outColor;

#define FLAG_CLIP 1u

// Mathematically evaluates true CSS border-radius distance
float sdRoundRect(vec2 p, vec2 extents, vec4 r) {
    // r = vec4(TopLeft, TopRight, BottomRight, BottomLeft)
    vec2 rad = (p.x > 0.0) ? r.yz : r.xw; // right: TR(y)/BR(z), left: TL(x)/BL(w)
    float radius = (p.y > 0.0) ? rad.y : rad.x; // bottom: BR(z)/BL(w), top: TR(y)/TL(x)
    
    radius = min(radius, min(extents.x, extents.y)); // Safety clamp
    
    vec2 q = abs(p) - extents + radius;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
}

void main() {
    UiElement el = pc.elementsPtr.elements[vElementId];
    
    // 1. HARDWARE CLIPPING (CSS overflow: hidden)
    // Using discard saves monumental fill rate for scrolled interfaces
    if ((el.flags & FLAG_CLIP) != 0u) {
        if (vPixelPos.x < el.clipRect.x || vPixelPos.y < el.clipRect.y ||
            vPixelPos.x > el.clipRect.z || vPixelPos.y > el.clipRect.w) {
            discard;
        }
    }
    
    // Hardware anti-aliasing delta (Pixel Width)
    float fw = max(length(vec2(dFdx(vLocalPos.x), dFdy(vLocalPos.y))), 0.001); 

    // 2. SOFT DROP SHADOWS
    vec4 finalColor = vec4(0.0);
    if (el.colorShadow.a > 0.0) {
        vec2 shadowPos = vLocalPos - el.shadowParams.xy;
        float shadowBlur = max(el.shadowParams.z, 0.0);
        float shadowSpread = el.shadowParams.w;
        
        vec4 shadowRadii = max(el.borderRadius + shadowSpread, vec4(0.0));
        vec2 shadowExtents = max(vHalfSize + shadowSpread, vec2(0.0));
        
        float dShadow = sdRoundRect(shadowPos, shadowExtents, shadowRadii);
        
        float shadowAlpha = 1.0 - smoothstep(-max(shadowBlur, fw), max(shadowBlur, fw), dShadow);
        finalColor = el.colorShadow;
        finalColor.a *= shadowAlpha;
    }
    
    // 3. MAIN BOX BACKGROUND & GRADIENT
    float dBox = sdRoundRect(vLocalPos, vHalfSize, el.borderRadius);
    float boxAlpha = 1.0 - smoothstep(-fw, fw, dBox);
    
    vec4 bgColor = el.colorStart;
    vec2 size = vHalfSize * 2.0;
    vec2 actualUV = (size.x > 0.0 && size.y > 0.0) ? (vLocalPos / size) + 0.5 : vec2(0.5);
    
    if (dot(el.gradientDir, el.gradientDir) > 0.001) {
        float t = clamp(dot(actualUV - 0.5, el.gradientDir) + 0.5, 0.0, 1.0);
        bgColor = mix(el.colorStart, el.colorEnd, t);
    }
    
    vec4 texColor = vec4(1.0);
    if (el.textureId != 0xFFFFFFFFu) {
        texColor = texture(bindlessTextures[nonuniformEXT(el.textureId)], clamp(actualUV, 0.0, 1.0));
    }
    
    vec4 contentColor = bgColor * texColor;
    
    // 4. INSET BORDERS
    vec4 boxColor = contentColor;
    if (el.borderWidth > 0.0) {
        float dBorder = dBox + el.borderWidth;
        float innerAlpha = 1.0 - smoothstep(-fw, fw, dBorder);
        boxColor = mix(el.colorBorder, contentColor, innerAlpha);
    }
    
    boxColor.a *= boxAlpha; // Bind strictly to bounds limits
    
    // 5. SRC_OVER COMPOSITING (UI Over Shadow)
    if (boxColor.a > 0.0 || finalColor.a > 0.0) {
        float outA = boxColor.a + finalColor.a * (1.0 - boxColor.a);
        vec3 outRGB = vec3(0.0);
        if (outA > 0.0) {
            outRGB = (boxColor.rgb * boxColor.a + finalColor.rgb * finalColor.a * (1.0 - boxColor.a)) / outA;
        }
        finalColor = vec4(outRGB, outA);
    }
    
    // Optional master fade out
    finalColor.a *= el.opacity;
    if (finalColor.a < 0.001) discard;
    
    outColor = finalColor;
}