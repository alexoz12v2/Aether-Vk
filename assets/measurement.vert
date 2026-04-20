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

  float tickSize = 15.0; // world units or scaled units? Wait, we want screen space size.
  // Better approach: calculate tick size in world space based on distance to camera, or just use a fixed world unit size if it's simpler. Let's use a scale factor passed from the host, or just calculate from viewProj.w 
  
  // Actually, we can use the up and right vectors to offset in world space.
  float scale1 = length(push.p1 - (inverse(push.viewProj) * vec4(0.0, 0.0, 0.0, 1.0)).xyz) * 0.01; // rough scale
  float scale2 = length(push.p2 - (inverse(push.viewProj) * vec4(0.0, 0.0, 0.0, 1.0)).xyz) * 0.01;

  vec3 pos = vec3(0.0);
  if (gl_VertexIndex == 0) pos = push.p1 - push.cameraUp * scale1;
  if (gl_VertexIndex == 1) pos = push.p1 + push.cameraUp * scale1;
  if (gl_VertexIndex == 2) pos = push.p1;
  if (gl_VertexIndex == 3) pos = push.p2;
  if (gl_VertexIndex == 4) pos = push.p2 - push.cameraUp * scale2;
  if (gl_VertexIndex == 5) pos = push.p2 + push.cameraUp * scale2;

  gl_Position = push.viewProj * vec4(pos, 1.0);
}
