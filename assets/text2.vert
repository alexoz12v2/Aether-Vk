#version 450 core
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct TextGlyph {
    vec2 pos;
    vec2 size;
    vec4 uv_bounds;   // xy = min, zw = max
    vec4 color;       // per-character color
    uint texture_id;  // bindless atlas index
    uint style;       // Bit 0 = Italic, Bit 1 = Bold
    uvec2 _pad;       // 64-byte alignment padding
};

// Bindless BDA block for glyphs
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer GlyphArray {
    TextGlyph glyphs[];
};

// 72 bytes Push Constant
layout(push_constant, scalar) uniform Push {
    GlyphArray glyphsPtr;
    mat4 viewProj;
} push;

layout(location = 0) out vec2 outUV;
layout(location = 1) flat out uint outTextureId;
layout(location = 2) out vec4 outColor;
layout(location = 3) flat out uint outStyle;

// 4 vertices for TRIANGLE_STRIP quad
const vec2 quad[4] = vec2[](
    vec2(0.0, 0.0), // Top-left
    vec2(0.0, 1.0), // Bottom-left
    vec2(1.0, 0.0), // Top-right
    vec2(1.0, 1.0)  // Bottom-right
);

void main() {
    TextGlyph glyph = push.glyphsPtr.glyphs[gl_InstanceIndex];
    vec2 inPosition = quad[gl_VertexIndex];
    
    outUV = mix(glyph.uv_bounds.xy, glyph.uv_bounds.zw, inPosition);
    outTextureId = glyph.texture_id;
    outColor = glyph.color;
    outStyle = glyph.style;
    
    vec2 localPos = inPosition * glyph.size;
    
    // Feature: Software pseudo-italics styling
    // OpenGL/Vulkan Y points down. Y=0 is the top of the glyph.
    if ((glyph.style & 1u) != 0u) {
        float slant_strength = 0.25 * glyph.size.y;
        localPos.x += (1.0 - inPosition.y) * slant_strength;
    }

    vec2 screenPos = glyph.pos + localPos;
    gl_Position = push.viewProj * vec4(screenPos, 0.0, 1.0);
}
