#version 450 core

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

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
    float time;
    vec3 cameraRight;
    float seed;
    vec4 color;
    float radius;
    float cameraPos_x;
    float cameraPos_y;
    float cameraPos_z;
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
    
    vec3 currentPos = p.position; // DO NOT extrapolate using uptime!
    vec3 cameraPos = vec3(pc.cameraPos_x, pc.cameraPos_y, pc.cameraPos_z);
    vec3 relativePos = currentPos - cameraPos;
    vec3 worldPos = relativePos + (pc.cameraRight * localPos.x + pc.cameraUp * localPos.y) * pc.radius;
    gl_Position = pc.viewProj * vec4(worldPos, 1.0);
}
