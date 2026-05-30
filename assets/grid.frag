#version 450 core

layout(location = 0) in vec2 inNDC;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
  mat4 viewProj;
  vec3 cameraPos;
  float nearPlane;
  float farPlane;
  float density;
  vec3 gridColor;
} push;

// Anti-aliased grid line: returns [0, 1] alpha for a repeating line pattern.
// 'p'         : scaled coordinate (one grid cell = 1.0 unit)
// 'lineWidth' : half-width of the line in grid-space units
float gridLine(float p, float dp, float lineWidth) {
  float dist = abs(fract(p + 0.5) - 0.5);
  float edge0 = max(lineWidth - dp, 0.0);
  float edge1 = lineWidth + dp + 1e-5;
  return 1.0 - smoothstep(edge0, edge1, dist); // Just return the anti-aliased line
}

void main() {
  // unproject per pixel onto view space near and far to avoid interpolation singularities
  mat4 invViewProj = inverse(push.viewProj);
  vec4 unprojNear = invViewProj * vec4(inNDC, 1.0, 1.0);
  vec4 unprojFar = invViewProj * vec4(inNDC, 0.0, 1.0);

  vec3 nearPos = unprojNear.xyz / unprojNear.w;
  vec3 farPos  = unprojFar.xyz  / unprojFar.w;

  vec3 viewDir = farPos - nearPos;
  if (abs(viewDir.z) < 1e-6) {
    discard;
  }

  // In RTE space, camera is at z=0.  The absolute world Z=0 plane sits at
  // RTE z = -push.cameraPos.z, so we solve for t along the view ray.
  float target_z = -push.cameraPos.z;
  float t = (target_z - nearPos.z) / viewDir.z;

  if (t <= 0.0) {
    discard;
  }

  vec3 worldPos   = nearPos + t * viewDir;       // RTE position on the grid plane
  float linearDepth = length(worldPos);           // distance from camera (RTE origin)

  if (linearDepth > push.farPlane) {
    discard;
  }

  // Recover absolute XY by adding the camera's absolute position back.
  vec2 absolutePosXY = worldPos.xy + push.cameraPos.xy;

  // Scale coordinate so one grid cell = 1 world unit.
  float gridDensity = push.density * 5.0;

  // Compute fwidth on RTE worldPos to avoid catastrophic precision loss from adding a large cameraPos
  vec2 p = absolutePosXY * gridDensity;
  vec2 dp = fwidth(worldPos.xy * gridDensity);

  // 1. Calculate axis-independent lines
  float ax = gridLine(p.x, dp.x, 0.01);
  float ay = gridLine(p.y, dp.y, 0.01);
  float alpha = max(ax, ay) * 0.3;

  // 2. Apply a UNIFIED radial Moiré fade for minor lines
  float radialFade = 1.0 - smoothstep(0.2, 0.8, length(dp));
  alpha *= radialFade;

  // 3. Repeat for major lines
  vec2 p10 = p * 0.1;
  vec2 dp10 = dp * 0.1;
  float bx = gridLine(p10.x, dp10.x, 0.005);
  float by = gridLine(p10.y, dp10.y, 0.005);
  float alpha10 = max(bx, by) * 0.7;

  // 4. Unified radial Moiré fade for major lines
  float radialFade10 = 1.0 - smoothstep(0.2, 0.8, length(dp10));
  alpha10 *= radialFade10;

  // 5. Combine everything
  alpha = max(alpha, alpha10);

  // Distance fade
  float fadeStart = push.nearPlane + (push.farPlane - push.nearPlane) * 0.01;
  float fadeEnd   = push.farPlane * 0.2;
  float distFade  = 1.0 - smoothstep(fadeStart, fadeEnd, linearDepth);
  alpha *= distFade;

  if (alpha < 0.01) {
    discard;
  }

  outColor = vec4(push.gridColor, alpha);

  // Write NDC depth with a small bias to avoid Z-fighting with objects on z=0.
  vec4 clipPos = push.viewProj * vec4(worldPos, 1.0);
  gl_FragDepth = clamp(clipPos.z / clipPos.w + 0.00001, 0.0, 1.0);
}