#version 450 core

layout(push_constant) uniform Push {
  mat4 viewProj;
  vec3 centerPos;
  float size;
  vec3 color;
  float _pad0;
  vec3 cameraUp;
  float _pad1;
  vec3 cameraRight;
  float _pad2;
} push;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec3 outColor;

const vec2 quad[4] = vec2[] (
  vec2(-1.0, -1.0),
  vec2( 1.0, -1.0),
  vec2(-1.0,  1.0),
  vec2( 1.0,  1.0)
);

void main() {
  vec2 uv = quad[gl_VertexIndex];
  outUV = uv;
  outColor = push.color;

  vec3 worldPos = push.centerPos + push.cameraRight * uv.x * push.size + push.cameraUp * uv.y * push.size;
  gl_Position = push.viewProj * vec4(worldPos, 1.0);
}