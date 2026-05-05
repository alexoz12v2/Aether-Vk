#version 450 core

// --- Required Extensions ---
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require

// TODO: With GL_GOOGLE_include_directive enabled, move the bulky noise math into a shared file in your engine to keep things clean! #include "../noise_functions.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

// Thanks to GL_EXT_scalar_block_layout, we can use 'scalar' here.
// You no longer need manual "pad0" and "pad1" floats to fight alignment padding!
layout(push_constant, scalar) uniform PushConstants {
    mat4 viewProj;
    vec3 cameraUp;
    float time;
    vec3 cameraRight;
    float seed;
    vec4 color;
    float radius;
} pc;

// --- Dithering Math ---
// Interleaved Gradient Noise (IGN). Excellent screen-space noise for dithering.
float getDither(vec2 pos) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pos, magic.xy)));
}

// --- Procedural Noise Math ---
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

// Fractal Brownian Motion for the "cloud" texture
float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.866, -0.5, 0.5, 0.866);
    for (int i = 0; i < 3; i++) {
        v += a * noise(p);
        p = rot * p * 2.0;
        a *= 0.5;
    }
    return v;
}

void main() {
    float dist = length(inUV);
    if (dist > 1.0) {
        discard;
    }

    // 1. Generate Swirling Dust Noise
    vec2 noiseUV = inUV * 2.0 + vec2(pc.seed * 13.1);
    noiseUV += vec2(pc.time * 0.2);
    float n = fbm(noiseUV);

    // 2. Warp the Shape (No perfect circles)
    float warpedDist = dist + (n - 0.5) * 0.5;

    // 3. Gaussian Falloff & Density
    float density = exp(-warpedDist * warpedDist * 6.0);
    density *= smoothstep(0.1, 0.9, n); // Carve out fluffy gaps

    // Desired transparency
    float targetAlpha = clamp(density * pc.color.a, 0.0, 1.0);

    // 4. STOCHASTIC DITHERING (The Depth Sorting Fix)
    // Compare the calculated alpha against screen-space noise.
    if (targetAlpha < getDither(gl_FragCoord.xy)) {
        discard;
    }

    // 5. Fake Volumetric Shading
    vec3 coreColor = pc.color.rgb * 0.4; // Dense shadowy core
    vec3 edgeColor = mix(pc.color.rgb, vec3(1.0), 0.5); // Thin light-scattering edges
    vec3 finalColor = mix(edgeColor, coreColor, density);

    // 6. Final Output
    // Alpha is explicitly 1.0. The fragment is mathematically perfectly solid.
    outColor = vec4(finalColor, 1.0);
}