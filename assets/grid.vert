#version 450 core

layout(push_constant) uniform Push {
  mat4 viewProj;
  vec3 cameraPos;
  float nearPlane;
  float farPlane;
  float density;
  vec3 gridColor;
} push;

layout(location = 0) out vec3 outNearPos;
layout(location = 1) out vec3 outFarPos;

const vec2 quad[4] = vec2[] (
  vec2(-1.0, -1.0),
  vec2( 1.0, -1.0),
  vec2(-1.0,  1.0),
  vec2( 1.0,  1.0)
);

void main() {
  vec2 uv = quad[gl_VertexIndex];
  
  mat4 invViewProj = inverse(push.viewProj);
  vec4 unprojectedNear = invViewProj * vec4(uv.x, uv.y, 0.0, 1.0);
  vec4 unprojectedFar  = invViewProj * vec4(uv.x, uv.y, 1.0, 1.0);
  
  outNearPos = unprojectedNear.xyz / unprojectedNear.w;
  outFarPos  = unprojectedFar.xyz / unprojectedFar.w;
  
  gl_Position = vec4(uv.x, uv.y, 0.0, 1.0);
}
