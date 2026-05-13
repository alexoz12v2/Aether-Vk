#version 450 core

layout(push_constant) uniform Push {
    vec2 pos;
    vec2 scale;
    vec4 color;
    vec4 uv_bounds; // xy = min, zw = max
    uint texture_id;
    uint _pad[3];
    mat4 viewProj;
} push;

layout(location = 0) out vec2 outUV;
layout(location = 1) flat out uint outTextureId;

const vec2 quad[4] = vec2[] (
  vec2(0.0, 0.0),
  vec2(0.0, 1.0),
  vec2(1.0, 0.0),
  vec2(1.0, 1.0)
);

void main() {
    vec2 inPosition = quad[gl_VertexIndex];
    
    // Map 0..1 to uv_bounds
    outUV = mix(push.uv_bounds.xy, push.uv_bounds.zw, inPosition);
    
    vec2 screenPos = push.pos + (inPosition * push.scale);
    
    gl_Position = push.viewProj * vec4(screenPos, 0.0, 1.0);
    outTextureId = push.texture_id;
}
