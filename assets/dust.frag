// @assets/dust.frag

#version 450 core

#extension GL_EXT_buffer_reference2      : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_GOOGLE_include_directive   : require

#include "debug_utils.glsl"
#include "bvh_utils.glsl"

// 128 bytes
layout(push_constant, std430) uniform PushConstants {
    ParticleChunkBuffer globalParticleBuffer;
    ParticlePageTable   particlePageTable;

    mat4  viewProj;
    vec4  streamColor;
    uint  chunkOffset;
    uint  currentTime;  // In 1/300 seconds
    float maxTtl;       // limit for age fading (in 1/300 s)
    float macroScale;   // base point sprite pixel size multiplier
    float microRadius;  // physical size of the dust specks (eg 0.15)
    uint  numSpots;     // how many spots per blob (eg 12 or 16)
    float dispersionRate;
} pc;

layout(location = 0) in vec4 v_Color;
layout(location = 1) in flat uint v_Seed;
layout(location = 2) in flat float v_PointSize;

layout(location = 0) out vec4 fragColor;

uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float randomFloat(uint h) {
    return float(h) * (1.0 / 4294967295.0);
}

void main() {
    // 1. Re-map UV to [-0.5 -> 0.5] to restrict bounds to a circular blob
    vec2 uv = gl_PointCoord - vec2(0.5);
    float macroDistSq = dot(uv, uv);
    if (macroDistSq > 0.25) discard;

    // 1 pixel length in UV space is (1.0 / pointSize). We want a minimum radius of 0.5 pixels
    float minRadiusUV = 0.5 / max(1.0, v_PointSize);

    // clamp the render radius to be at least half a pixel
    float renderRadius = max(pc.microRadius, minRadiusUV);

    // Energy Conservation:
    // Area = PI * r^2. If we artificially enlarge the radius, we must reduce the brightness
    // Scale = (True Radius)^2 / (Render Radius) ^2
    // Added 1e-8 to prevent division by zero in case of completely dead particles
    float energyScale = (pc.microRadius * pc.microRadius) / (renderRadius * renderRadius);

    // use renderRadius to clip artificially enlarged grains off the 0.5 macro particle edge
    float maxPlacementR = max(0.0, 0.5 - renderRadius);

    // 2. Procedural cluster generation
    float minDist = 1.0;
    for (uint i = 0; i < pc.numSpots; i++) {
        // Unique hash seed specific to this spot within this specific macro-particle
        uint seed = v_Seed ^ pcg(i * 1973u + 9277u);
        uint h1 = pcg(seed);
        uint h2 = pcg(h1);

        // fast random vec2 in range [-1.0, 1.0] without trig functions
        vec2 randomDir = vec2(
            float(h1) * (2.0 / 4294967295.0) - 1.0,
            float(h2) * (2.0 / 4294967295.0) - 1.0
        );

        // Normalize the vector manually to form a circle, multiplied by random radius
        // We use absolute value of randomDir.x to replace sqrt() for area distribution bias
        // so that, even without sin/cos + sqrt, we should obtain a uniform area distribution
        float r = abs(randomDir.x) * maxPlacementR;

        // add small epsilon to avoid division by zero
        vec2 spotPos = (randomDir / (length(randomDir) + 0.0001)) * r;

        float d = distance(uv, spotPos);
        minDist = min(minDist, d);
    }

    // empty space outside of dust specks
    if (minDist > renderRadius) discard;

    // 3. Volumetric Shading: Borders of each particle should be darker
    // 0.0 at the center of the spot, 1.0 at the absolute edge of the spot
    float spotGradient = minDist / renderRadius;

    // Power curve transitions bright core gracefully into the dark rim
    float intensity = 1.0 - spotGradient * spotGradient;

    // edge mixes down to 20% brightness while core sits at 100%
    // apply energyScale to dim the particle if we had to enlarge it
    float darkness = mix(0.2, 1.0, intensity) * energyScale;

    // smooth the macro-particle's edges for soft anti-aliased overlap merging
    float macroFade = smoothstep(0.25, 0.2, macroDistSq);

    fragColor = vec4(v_Color.rgb * darkness, v_Color.a * macroFade);
}