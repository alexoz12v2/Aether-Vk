#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 inUV;
layout(location = 1) flat in uint inTextureId;
layout(location = 2) in vec4 inColor;
layout(location = 3) flat in uint inStyle;

// Bindless array containing up to max_fonts 
layout(set = 0, binding = 0) uniform sampler2D textureAtlases[];

layout(location = 0) out vec4 outColor;

void main() {
    float alpha = 1.0;
    
    // Feature: If textureId == MaxUint, render a solid quad (useful for underlines/backgrounds)
    if (inTextureId != 0xFFFFFFFFu) {
        uint texIdx = nonuniformEXT(inTextureId);
        alpha = texture(textureAtlases[texIdx], inUV).r;
        
        // Feature: Software Bold styling
        // Sample neighboring pixels in the atlas to artificially thicken the glyphs
        if ((inStyle & 2u) != 0u) {
            vec2 offset = 1.0 / vec2(textureSize(textureAtlases[texIdx], 0));
            float aRight = texture(textureAtlases[texIdx], inUV + vec2(offset.x, 0.0)).r;
            float aLeft  = texture(textureAtlases[texIdx], inUV - vec2(offset.x, 0.0)).r;
            float aDown  = texture(textureAtlases[texIdx], inUV + vec2(0.0, offset.y)).r;
            float aUp    = texture(textureAtlases[texIdx], inUV - vec2(0.0, offset.y)).r;
            
            alpha = max(alpha, max(max(aRight, aLeft), max(aDown, aUp)));
        }
    }
    
    if (alpha < 0.01) discard; // Save fill rate and reduce quad overdraw cost
    outColor = vec4(inColor.rgb, inColor.a * alpha);
}
