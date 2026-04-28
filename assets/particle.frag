#version 460

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
    uint bufferIndex;
    vec2 pad2;
} pc;

void main() {
    // Simple circular particle
    float dist = length(inUV);
    if (dist > 1.0) {
        discard;
    }
    
    // Soft edge
    float alpha = smoothstep(1.0, 0.8, dist) * pc.color.a;
    outColor = vec4(pc.color.rgb, alpha);
}
