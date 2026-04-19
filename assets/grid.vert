#version 450 core

layout(push_constant) uniform Push {
  mat4 viewProj;
  vec3 cameraPos;
  float nearPlane;
  float farPlane;
  float density;
  vec3 gridColor;
} push;

layout(location = 0) out vec4 outUnprojectedNear;
layout(location = 1) out vec4 outUnprojectedFar;

const vec2 quad[4] = vec2[] (
  vec2(-1.0, -1.0),
  vec2( 1.0, -1.0),
  vec2(-1.0,  1.0),
  vec2( 1.0,  1.0)
);

void main() {
  vec2 uv = quad[gl_VertexIndex];
  
  mat4 invViewProj = inverse(push.viewProj);
  outUnprojectedNear = invViewProj * vec4(uv.x, uv.y, 0.0, 1.0);
  outUnprojectedFar  = invViewProj * vec4(uv.x, uv.y, 1.0, 1.0);
  
  gl_Position = vec4(uv.x, uv.y, 0.0, 1.0);
}
