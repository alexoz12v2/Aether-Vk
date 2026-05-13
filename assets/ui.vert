#version 450 core
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct UiElement {
    vec4 bounds; vec4 clipRect;
    vec4 colorStart; vec4 colorEnd;
    vec4 colorBorder; vec4 colorShadow;
    vec4 borderRadius; vec4 shadowParams;
    vec2 gradientDir; float borderWidth; uint textureId;
    uint flags; float opacity; float rotation; uint _pad;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer ElementArray {
    UiElement elements[];
};

layout(push_constant, scalar) uniform PushConstants {
    ElementArray elementsPtr;
    mat4 viewProj;
    vec2 viewportSize;
} pc;

layout(location = 0) out vec2 vLocalPos;
layout(location = 1) out vec2 vHalfSize;
layout(location = 2) out flat uint vElementId;
layout(location = 3) out vec2 vUV;
layout(location = 4) out vec2 vPixelPos;

void main() {
    uint idx = gl_InstanceIndex; // 1 Instance = 1 UI Element
    UiElement el = pc.elementsPtr.elements[idx];
    
    vec2 size = max(el.bounds.zw, vec2(0.0));
    vec2 halfSize = size * 0.5;
    
    // Dynamically expand bounding quad to prevent shadow/AA clipping
    float shadowSpread = el.shadowParams.w;
    float shadowBlur = max(el.shadowParams.z, 0.0);
    float maxShadowOffset = max(abs(el.shadowParams.x), abs(el.shadowParams.y));
    
    // Safe expansion margin (shadow + 2px safety for AA)
    float expansion = max(0.0, shadowSpread) + shadowBlur * 2.0 + maxShadowOffset + 2.0;
    
    vec2 uvs[6] = vec2[](
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    vec2 uv = uvs[gl_VertexIndex];
    vUV = uv;
    
    vec2 localPos = (uv - 0.5) * size;
    
    vec2 expandDir = sign(localPos);
    if (expandDir.x == 0.0) expandDir.x = (uv.x > 0.5) ? 1.0 : -1.0;
    if (expandDir.y == 0.0) expandDir.y = (uv.y > 0.5) ? 1.0 : -1.0;
    
    localPos += expandDir * expansion;
    vLocalPos = localPos;
    vHalfSize = halfSize;
    vElementId = idx;
    
    // Evaluate Optional Rotation
    vec2 rotatedPos = localPos;
    if (el.rotation != 0.0) {
        float c = cos(el.rotation);
        float s = sin(el.rotation);
        rotatedPos = vec2(
            rotatedPos.x * c - rotatedPos.y * s,
            rotatedPos.x * s + rotatedPos.y * c
        );
    }
    
    vec2 globalPos = el.bounds.xy + halfSize + rotatedPos;
    vPixelPos = globalPos; // Retain global unprojected pos for clip mapping
    
    gl_Position = pc.viewProj * vec4(globalPos, 0.0, 1.0);
}