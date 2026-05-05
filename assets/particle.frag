#version 450 core

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec3 cameraUp;
    float pad0;
    vec3 cameraRight;
    float pad1;
    vec4 color;
    float radius;
} pc;

// Interleaved Gradient Noise function
float getDither(vec2 pos) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pos, magic.xy)));
}

void main() {
    // Simple circular particle
    float dist = length(inUV);
    if (dist > 1.0) {
        discard;
    }
    
    // Soft edge
    float alpha = smoothstep(1.0, 0.8, dist) * pc.color.a;

    // Dithering: Discard fragments based on screen-space noise
    // Note: This is an approximation. To achieve true semitransparent particles, you should
    // - disable depth write
    // - use a GPU shader to sort particles back to front before rendering them
    if (alpha < getDither(gl_FragCoord.xy)) {
        discard;
    }

    // Determine edge color based on brightness
    float luminance = dot(pc.color.rgb, vec3(0.299, 0.587, 0.114));
    vec3 edgeColor = luminance > 0.5 ? vec3(0.0) : vec3(1.0);
    
    // Interpolate towards edge color
    float edgeFactor = smoothstep(0.5, 1.0, dist);
    vec3 finalColor = mix(pc.color.rgb, edgeColor, edgeFactor);

    outColor = vec4(finalColor, alpha);
}
