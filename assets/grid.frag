#version 450 core

layout(location = 0) in vec4 inUnprojectedNear;
layout(location = 1) in vec4 inUnprojectedFar;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
  mat4 viewProj;
  vec3 cameraPos;
  float nearPlane;
  float farPlane;
  float density;
  vec3 gridColor;
} push;

void main() {
  vec3 nearPos = inUnprojectedNear.xyz / inUnprojectedNear.w;
  vec3 farPos  = inUnprojectedFar.xyz / inUnprojectedFar.w;

  vec3 viewDir = farPos - nearPos;
  if (abs(viewDir.z) < 1e-6) {
    discard;
  }
  
  float t = -nearPos.z / viewDir.z;
  if (t <= 0.0) {
    discard;
  }
  
  vec3 worldPos = nearPos + t * viewDir;
  float linearDepth = length(worldPos - push.cameraPos);
  
  if (linearDepth > push.farPlane) {
    discard;
  }
  
  // Adjust the scale of the grid based on push.density.
  // The density should be multiplied to map world space to grid space.
  float gridDensity = push.density * 5.0; // making it a bit less coarse
  
  vec2 p = worldPos.xy * gridDensity;
  vec2 dp = fwidth(p);
  
  float lineWidth = 0.01; // thinner lines
  
  vec2 grid = smoothstep(lineWidth + dp, max(lineWidth - dp, 0.0), abs(fract(p + 0.5) - 0.5));
  float alpha = max(grid.x, grid.y);
  
  vec2 p10 = p * 0.1;
  vec2 dp10 = dp * 0.1;
  float majorLineWidth = 0.005; // thinner major lines
  vec2 grid10 = smoothstep(majorLineWidth + dp10, max(majorLineWidth - dp10, 0.0), abs(fract(p10 + 0.5) - 0.5));
  float alpha10 = max(grid10.x, grid10.y);
  
  alpha = max(alpha * 0.3, alpha10 * 0.7);
  
  float distFade = 1.0 - smoothstep(push.nearPlane + (push.farPlane - push.nearPlane) * 0.01, push.farPlane * 0.2, linearDepth);
  alpha *= distFade;
  
  vec4 clipPos = push.viewProj * vec4(worldPos, 1.0);
  
  // Add a slight depth bias to the grid to prevent Z-fighting with objects exactly at z=0
  clipPos.z -= 0.00001 * clipPos.w;
  
  if (alpha < 0.01) {
    discard;
  }
  
  outColor = vec4(push.gridColor, alpha);
  
  gl_FragDepth = (clipPos.z / clipPos.w);
}
