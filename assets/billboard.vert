#version 450 core

layout(push_constant) uniform Push {
  mat4 viewProj;
  vec3 centerPos;
  float _pad0;
  vec3 cameraUp;
  float _pad1;
  vec3 cameraRight;
  float _pad2;
  vec2 size;
  uint isScreenSpace;
  uint textureId;
} push;

layout(location = 0) out vec2 outUV;
layout(location = 1) flat out uint outTextureId;

const vec2 quad[4] = vec2[] (
  vec2(-1.0, -1.0),
  vec2( 1.0, -1.0),
  vec2(-1.0,  1.0),
  vec2( 1.0,  1.0)
);

void main() {
  vec2 uv = quad[gl_VertexIndex];
  outUV = uv * 0.5 + 0.5; // [0, 1] UVs
  outTextureId = push.textureId;

  if (push.isScreenSpace == 1) {
    // Treat centerPos as screen coordinates [-1, 1], and size as fraction of screen.
    // We ignore cameraRight and cameraUp.
    vec2 screenPos = push.centerPos.xy + uv * push.size;
    gl_Position = vec4(screenPos, 0.0, 1.0);
  } else {
    // World space billboard
    vec3 worldPos = push.centerPos + push.cameraRight * uv.x * push.size.x * 0.5 + push.cameraUp * uv.y * push.size.y * 0.5;
    gl_Position = push.viewProj * vec4(worldPos, 1.0);
  }
}
