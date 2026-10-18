#version 450 core
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "rte_distance.glsl"

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vUV;
layout(location = 2) in flat float vLineWidth;
layout(location = 3) in flat uint vTexId;
layout(location = 4) in vec3 vAnchorU;
layout(location = 5) in flat uint vAnchorLayer;
layout(location = 6) in flat vec4 vAnchorDist;
layout(location = 7) in flat float vInvAnchorLen;

layout(set = 0, binding = 0) uniform sampler2D trajectoryTextures[];

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outGlobalDepth; // MRT: (layer_index_f32, local_distance)

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

    // Anchored pieces: (16 + m, |o_m + u| - |o_m|) in AU, the precise offset the composite compares
    // with micro layer m's mesh offsets; the composite turns it into the sentinel afterwards.
    // Otherwise not an occlusion/pick target: same sentinel as the MRT clear value.
    outGlobalDepth = (vAnchorLayer != 0xFFFFFFFFu)
        ? vec2(16.0 + float(vAnchorLayer), rteDistanceOffset(vAnchorDist, vInvAnchorLen, vAnchorU))
        : vec2(-1.0, -1.0);
}