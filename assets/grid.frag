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
  vec3 inNearPos = inUnprojectedNear.xyz / inUnprojectedNear.w;
  vec3 inFarPos  = inUnprojectedFar.xyz / inUnprojectedFar.w;

  float t = -inNearPos.z / (inFarPos.z - inNearPos.z);
  
  if (t <= 0.0) {
    discard;
  }
  
  vec3 worldPos = inNearPos + t * (inFarPos - inNearPos);
  
  float linearDepth = length(worldPos - push.cameraPos);
  if (linearDepth > push.farPlane) {
    discard;
  }
  
  vec2 p = worldPos.xy * push.density;
  vec2 dp = fwidth(p);
  
  float lineWidth = 0.003; 
  
  vec2 grid = smoothstep(lineWidth + dp, lineWidth - dp, abs(fract(p + 0.5) - 0.5));
  vec2 blend = smoothstep(0.1, 0.8, dp);
  vec2 coverage = mix(grid, vec2(lineWidth), blend);
  
  float alpha = max(coverage.x, coverage.y);
  
  vec2 p10 = p * 0.1;
  vec2 dp10 = dp * 0.1;
  float majorLineWidth = 0.001;
  vec2 grid10 = smoothstep(majorLineWidth + dp10, majorLineWidth - dp10, abs(fract(p10 + 0.5) - 0.5));
  vec2 coverage10 = mix(grid10, vec2(majorLineWidth), smoothstep(0.1, 0.8, dp10));
  
  alpha = max(alpha * 0.5, max(coverage10.x, coverage10.y) * 0.8);
  
  float distFromCenter = max(abs(p10.x), abs(p10.y));
  float gridFade = 1.0 - smoothstep(2.8, 3.0, distFromCenter);
  alpha *= gridFade;
  
  float fade = 1.0 - smoothstep(push.nearPlane + (push.farPlane - push.nearPlane) * 0.5, push.farPlane, linearDepth);
  alpha *= fade;
  
  if (alpha < 0.01) {
    discard;
  }
  
  outColor = vec4(push.gridColor, alpha);
  
  vec4 clipPos = push.viewProj * vec4(worldPos, 1.0);
  gl_FragDepth = (clipPos.z / clipPos.w);
}
