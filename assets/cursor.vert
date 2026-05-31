#version 450 core

layout(push_constant) uniform Push {
  mat4 view;         
  mat4 viewProj;     
  mat4 model;
  float cursorSize;  
  vec2 window_extent;
} push;

layout(location = 0) out vec3 outRo;
layout(location = 1) out vec3 outWorldPos;
layout(location = 2) out float outScale;

const vec2 quad[4] = vec2[] (
  vec2(-1.0, -1.0),
  vec2( 1.0, -1.0),
  vec2(-1.0,  1.0),
  vec2( 1.0,  1.0)
);

void main() {
  vec2 uv = quad[gl_VertexIndex];

  mat4 invView = inverse(push.view);
  vec3 camPos = invView[3].xyz;
  vec3 cursorPos = push.model[3].xyz;

  vec3 right = normalize(invView[0].xyz);
  vec3 up    = normalize(invView[2].xyz);

  float dist = max(length(camPos - cursorPos), 1e-10);

  // Log-based scaling: cursor size adapts smoothly across AU to km range.
  // At dist=10 AU → pct≈0.07 (7%), at dist=0.001 AU → pct≈0.03 (3%),
  // at dist=1e-8 AU → pct≈0.01 (1%). Never disappears completely.
  float logDist = log(dist * 1e6) / log(10.0); // shift so typical AU values → positive
  float t = clamp(logDist / 12.0, 0.0, 1.0);   // 0..1 over ~12 decades
  float pct = mix(0.07, 0.12, t);               // 7% to 12% of screen
  
  // Choose minimum size between width and height
  float desiredSizePixels = min(push.window_extent.x, push.window_extent.y) * pct;

  vec4 centerClip = push.viewProj * vec4(cursorPos, 1.0);
  
  // Use a strictly positive safe_w for computing scale to avoid div by zero/explosions
  float safe_w = max(abs(centerClip.w), 1e-6);
  
  vec4 upClip = push.viewProj * vec4(up, 0.0);
  vec2 upNDC = upClip.xy / safe_w;
  float upLenPixels = length(upNDC * (push.window_extent / 2.0));
  
  // Avoid division by zero if upLenPixels is near 0
  float scale = (desiredSizePixels / 2.0) / max(upLenPixels, 0.0001);

  vec3 worldPos = cursorPos
                + right * uv.x * scale * 1.8
                + up    * uv.y * scale * 1.8;

  vec4 clipPos = push.viewProj * vec4(worldPos, 1.0);
  
  // If the cursor is in front of the camera, force its w to match the safe_w 
  // used during scale calculation so its screen size exactly matches desiredSizePixels.
  // Also force z to 0.5 * w to prevent near/far plane clipping (NO_DEPTH_TEST is used).
  if (clipPos.w > 0.0) {
      clipPos.w = safe_w;
      clipPos.z = 0.5 * clipPos.w;
  }
  
  gl_Position = clipPos;

  outRo = camPos - cursorPos;
  outWorldPos = worldPos - cursorPos;
  outScale = scale;
}
