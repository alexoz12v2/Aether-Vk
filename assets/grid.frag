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

  // Base scale coordinate
  float baseDensity = push.density * 5.0;
  
  // Continuous LOD based on distance (linearDepth)
  // Determine the power of 10 for the current depth
  float lod = log(max(linearDepth * baseDensity, 1e-6)) / log(10.0);
  float lodLevel = floor(lod);
  float lodFraction = fract(lod);

  // Compute 3 scales of grids: minor, major, and macro (to handle the fade smoothly)
  float scale0 = pow(10.0, -lodLevel + 1.0); // minor
  float scale1 = scale0 * 0.1;               // major
  float scale2 = scale1 * 0.1;               // macro

  // 1. Minor lines (fade out as they get dense)
  vec2 p0 = absolutePosXY * baseDensity * scale0;
  vec2 dp0 = fwidth(worldPos.xy * baseDensity * scale0);
  float alpha0 = max(gridLine(p0.x, dp0.x, 0.005), gridLine(p0.y, dp0.y, 0.005)) * 0.3;
  alpha0 *= 1.0 - smoothstep(0.1, 0.8, length(dp0));
  // Fade out minor lines as we zoom out / look further away
  alpha0 *= (1.0 - lodFraction);

  // 2. Major lines (fully visible, blend into macro)
  vec2 p1 = absolutePosXY * baseDensity * scale1;
  vec2 dp1 = fwidth(worldPos.xy * baseDensity * scale1);
  float alpha1 = max(gridLine(p1.x, dp1.x, 0.008), gridLine(p1.y, dp1.y, 0.008)) * 0.5;
  alpha1 *= 1.0 - smoothstep(0.1, 0.8, length(dp1));

  // 3. Macro lines (fade in from the distance)
  vec2 p2 = absolutePosXY * baseDensity * scale2;
  vec2 dp2 = fwidth(worldPos.xy * baseDensity * scale2);
  float alpha2 = max(gridLine(p2.x, dp2.x, 0.012), gridLine(p2.y, dp2.y, 0.012)) * 0.8;
  alpha2 *= 1.0 - smoothstep(0.1, 0.8, length(dp2));
  // Fade in macro lines
  alpha2 *= lodFraction;

  // Combine
  float alpha = max(max(alpha0, alpha1), alpha2);

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