#version 450 core

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
  layout(offset = 0) mat4 viewProj;
  layout(offset = 64) mat4 invViewProj;
  layout(offset = 128) vec3 cameraPos;
  layout(offset = 140) float nearPlane;
  layout(offset = 144) float farPlane;
  layout(offset = 148) float density;
  layout(offset = 152) vec2 _pad1;
  layout(offset = 160) vec3 gridColor;
  layout(offset = 172) float _pad2;
} push;

void main() {
  vec4 unprojectedNear = push.invViewProj * vec4(inUV.x, inUV.y, 0.0, 1.0);
  vec4 unprojectedFar  = push.invViewProj * vec4(inUV.x, inUV.y, 1.0, 1.0);
  
  vec3 nearPos = unprojectedNear.xyz / unprojectedNear.w;
  vec3 farPos  = unprojectedFar.xyz / unprojectedFar.w;
  
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
  
  float lineWidth = 0.03; 
  
  vec2 grid = smoothstep(lineWidth + dp, lineWidth - dp, abs(fract(p + 0.5) - 0.5));
  float alpha = max(grid.x, grid.y);
  
  vec2 p10 = p * 0.1;
  vec2 dp10 = dp * 0.1;
  float majorLineWidth = 0.015;
  vec2 grid10 = smoothstep(majorLineWidth + dp10, majorLineWidth - dp10, abs(fract(p10 + 0.5) - 0.5));
  float alpha10 = max(grid10.x, grid10.y);
  
  alpha = max(alpha * 0.3, alpha10 * 0.7);
  
  float distFade = 1.0 - smoothstep(push.nearPlane + (push.farPlane - push.nearPlane) * 0.01, push.farPlane * 0.2, linearDepth);
  alpha *= distFade;
  
  if (alpha < 0.01) {
    discard;
  }
  
  outColor = vec4(push.gridColor, alpha);
  
  vec4 clipPos = push.viewProj * vec4(worldPos, 1.0);
  gl_FragDepth = (clipPos.z / clipPos.w);
}
