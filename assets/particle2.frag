#version 450 core

// --- Required Extensions ---
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require

// TODO: With GL_GOOGLE_include_directive enabled, move the bulky noise math into a shared file in your engine to keep things clean! #include "../noise_functions.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

// Bindings and Push Constants remain exactly as provided
layout(push_constant, std430) uniform PushConstants {
    mat4 viewProj;
    vec3 cameraUp;
    float time;
    vec3 cameraRight;
    float seed;
    vec4 color;
    float radius;
    float cameraPos_x;
    float cameraPos_y;
    float cameraPos_z;
} pc;

// --- Dithering Math (Optional Fallback) ---
float getDither(vec2 pos, float seed) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    // FIX: Adding the particle's seed prevents overlapping particles from 
    // sharing the exact same noise pattern, which caused the "flat" look.
    return fract(magic.z * fract(dot(pos, magic.xy) + seed));
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

// Fractal Brownian Motion
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

    // 1. Fake 3D Volume (Crucial for Distance Perception)
    // Treats the flat 2D sprite as a 3D hemisphere. This allows us to apply
    // directional lighting, giving your brain visual cues for depth and volume.
    float z = sqrt(max(0.0, 1.0 - dist * dist));
    vec3 normal = vec3(inUV.x, inUV.y, z);

    // 2. Generate Streaking Comet Dust Noise
    // Comet tails stretch due to solar winds. Scaling X and Y differently creates streaks.
    vec2 noiseUV = inUV * vec2(1.5, 3.0) + vec2(pc.seed * 13.1);
    noiseUV.y -= pc.time * 1.2; // Flow backward along the tail
    float n = fbm(noiseUV);

    // 3. Warp the Shape & Calculate Density
    float warpedDist = dist + (n - 0.5) * 0.4;
    
    // Core vs Halo: Comets have a hot, dense core and a diffuse, streaky halo.
    float density = exp(-warpedDist * warpedDist * 6.0);
    density *= mix(0.4, 1.0, smoothstep(0.1, 0.9, n)); // Soften the noise gaps

    float targetAlpha = clamp(density * pc.color.a, 0.0, 1.0);

    // 4. Volumetric Shading (Directional Light + Mie Scattering)
    vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0)); // Fake sunlight direction
    float diffuse = max(dot(normal, lightDir), 0.0);
    
    // Comets scatter light strongly forward (rim lighting creates a glowing edge)
    float rim = pow(1.0 - max(dot(normal, vec3(0.0, 0.0, 1.0)), 0.0), 2.5);

    vec3 coreColor = pc.color.rgb * 1.5; // Bright inner dust
    vec3 shadowColor = pc.color.rgb * 0.3; // Darker shadowed dust
    vec3 scatterColor = mix(pc.color.rgb, vec3(1.0), 0.8); // Glowing scattered edge

    // Combine lighting
    vec3 finalColor = mix(shadowColor, coreColor, diffuse);
    finalColor += scatterColor * rim * 0.8; // Add scattering glow

    // ==========================================
    // 5. THE OUTPUT (Fixing the "indiscriminable points")
    // ==========================================
    
    // [RECOMMENDED: TRUE ALPHA BLENDING]
    // Outputting smooth alpha removes the static noise and naturally blends the 
    // comet tail.
    outColor = vec4(finalColor, targetAlpha);

    // [FALLBACK: OPAQUE PIPELINE DITHERING]
    // If your pipeline strictly prohibits alpha blending, uncomment this block 
    // and comment out the outColor line above. Offsetting the dither with `pc.seed` 
    // ensures overlapping particles don't identically mask each other out.
    /*
    if (targetAlpha < getDither(gl_FragCoord.xy, pc.seed * 13.37)) {
        discard;
    }
    outColor = vec4(finalColor, 1.0);
    */
}