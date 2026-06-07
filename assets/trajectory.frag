#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vUV;
layout(location = 2) in flat float vLineWidth;
layout(location = 3) in flat uint vTexId;

layout(set = 0, binding = 0) uniform sampler2D trajectoryTextures[];

layout(location = 0) out vec4 outColor;

void main() {
    float distFromCenter = abs(vUV.y);

    float feather = 2.0 / max(vLineWidth, 1.0);

    // MATHEMATICALLY SAFE smoothstep:
    // Edge0 is now clamped to strictly <= Edge1, preventing Mali/Adreno Undefined Behavior.
    float alphaEdge = 1.0 - smoothstep(max(0.0, 1.0 - feather), 1.0, distFromCenter);

    if (alphaEdge <= 0.001) discard;

    // Note: Assuming a solid white texture at index 0 or similar for untextured lines.
    // vec4 tex = texture(trajectoryTextures[nonuniformEXT(vTexId)], vec2(vUV.x, vUV.y * 0.5 + 0.5));
    // For now we just output the color directly if textures aren't set up yet, to test rendering
    vec4 tex = vec4(1.0);

    outColor = vColor * tex;
    outColor.a *= alphaEdge;
}