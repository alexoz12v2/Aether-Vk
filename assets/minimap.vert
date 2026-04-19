#version 450 core

struct Planet {
    vec2 pos;
    float size;
    float pad;
    vec4 color;
};

layout(push_constant) uniform Push {
    vec2 offset;
    vec2 size;
    vec2 playerPos;
    float maxDistance;
    uint numPlanets;
    Planet planets[16];
} push;

layout(location = 0) out vec2 outUV;

const vec2 quad[4] = vec2[] (
  vec2(0.0, 0.0),
  vec2(0.0, 1.0),
  vec2(1.0, 0.0),
  vec2(1.0, 1.0)
);

void main() {
    vec2 inPosition = quad[gl_VertexIndex];
    outUV = inPosition;
    
    vec2 screenPos = push.offset + (inPosition * push.size);
    
    gl_Position = vec4(screenPos, 0.0, 1.0);
}