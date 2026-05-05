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

  float dist = max(length(camPos - cursorPos), 0.001);
  
  float t = clamp((dist - 3.3) / (10.0 - 3.3), 0.0, 1.0);
  float pct = mix(0.12, 0.07, t);
  
  float min_axis = min(push.window_extent.x, push.window_extent.y);
  float desiredSizePixels = pct * min_axis;
  
  vec4 centerClip = push.viewProj * vec4(cursorPos, 1.0);
  float w = max(abs(centerClip.w), 0.0001);
  
  vec4 upClip = push.viewProj * vec4(up, 0.0);
  vec2 upNDC = upClip.xy / w;
  float upLenPixels = length(upNDC * (push.window_extent / 2.0));
  
  float scale = (desiredSizePixels / 2.0) / max(upLenPixels, 0.0001);

  vec3 worldPos = cursorPos
                + right * uv.x * scale * 1.8
                + up    * uv.y * scale * 1.8;

  gl_Position = push.viewProj * vec4(worldPos, 1.0);

  outRo = camPos - cursorPos;
  outWorldPos = worldPos - cursorPos;
  outScale = scale;
}
