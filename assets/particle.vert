#version 460

struct Particle {
    uint id_low;
    uint id_high;
    uint age_low;
    uint age_high;
    vec3 position;
    float mass;
    vec3 velocity;
    uint active_flag;
};

// Bind this EXACTLY ONCE globally. No nonuniformEXT array needed!
layout(std430, set = 0, binding = 0) readonly buffer ParticleDataBuffer {
    Particle particles[];
};

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec3 cameraUp;
    float pad0;
    vec3 cameraRight;
    float pad1;
    vec4 color;
    float radius;
    vec3 pad2;
} pc;

layout(location = 0) out vec2 outUV;

const vec2 quadVertices[4] = vec2[4](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0)
);

void main() {
    Particle p = particles[gl_InstanceIndex];
    
    // If not active, draw degenerated
    if (p.active_flag == 0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    
    vec2 localPos = quadVertices[gl_VertexIndex];
    outUV = localPos;
    
    vec3 worldPos = p.position + (pc.cameraRight * localPos.x + pc.cameraUp * localPos.y) * pc.radius;
    gl_Position = pc.viewProj * vec4(worldPos, 1.0);
}
