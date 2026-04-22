#version 450 core

layout(push_constant) uniform Push {
  mat4 viewProj;
  vec3 p1;
  float _pad0;
  vec3 p2;
  float _pad1;
  vec3 cameraUp;
  float _pad2;
  vec3 cameraRight;
  float _pad3;
  vec3 color;
} push;

layout(location = 0) out vec3 outColor;

void main() {
  outColor = push.color;

  // Scale ticks based on a fixed ratio of the main line length, or a reasonable constant
  float dist = length(push.p2 - push.p1);
  float tickSize = dist * 0.05 + 0.1; // proportional to distance but with a minimum
  
  vec3 pos = vec3(0.0);
  if (gl_VertexIndex == 0) pos = push.p1 - push.cameraUp * tickSize;
  if (gl_VertexIndex == 1) pos = push.p1 + push.cameraUp * tickSize;
  if (gl_VertexIndex == 2) pos = push.p1;
  if (gl_VertexIndex == 3) pos = push.p2;
  if (gl_VertexIndex == 4) pos = push.p2 - push.cameraUp * tickSize;
  if (gl_VertexIndex == 5) pos = push.p2 + push.cameraUp * tickSize;

  gl_Position = push.viewProj * vec4(pos, 1.0);
}
