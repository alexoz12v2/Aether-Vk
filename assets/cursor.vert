#version 450 core

layout(push_constant) uniform Push {
  vec4 clipPos; // Packed into the first vec4 of viewProj
  vec4 _pad0;
  vec4 _pad1;
  vec4 _pad2;
  vec4 right_proj11;
  vec4 up_win_x;
  vec4 relative_cam_pos_win_y;
} push;

layout(location = 0) out vec3 outRo;
layout(location = 1) out vec3 outWorldPos;
layout(location = 2) out float outScale;

const vec2 quad[4] = vec2[] (
  vec2(-1.0, -1.0), vec2( 1.0, -1.0),
  vec2(-1.0,  1.0), vec2( 1.0,  1.0)
);

void main() {
  vec2 uv = quad[gl_VertexIndex];
  
  vec3 right = normalize(push.right_proj11.xyz);
  float proj11 = push.right_proj11.w;
  vec3 up = normalize(push.up_win_x.xyz);
  float win_x = push.up_win_x.w;
  vec3 relative_cam_pos = push.relative_cam_pos_win_y.xyz;
  float win_y = push.relative_cam_pos_win_y.w;

  vec3 cursorPos = -relative_cam_pos;
  float dist = max(length(relative_cam_pos), 1e-10);

  float logDist = log(dist * 1e6) / log(10.0);
  float t = clamp(logDist / 12.0, 0.0, 1.0);
  float pct = mix(0.07, 0.12, t);
  float desiredSizePixels = min(win_x, win_y) * pct;

  float fov_tan = 1.0 / max(abs(proj11), 1e-6);
  float scale = dist * fov_tan * (desiredSizePixels / win_y);
  vec3 localOffset = right * uv.x * scale * 1.8 + up * uv.y * scale * 1.8;

  vec4 centerClip = push.clipPos; // Extracted directly from CPU

  if (centerClip.w <= 0.0) {
    gl_Position = vec4(0.0);
    outScale = 0.0;
    return;
  }

  vec2 ndcOffset = vec2(uv.x, -uv.y) * desiredSizePixels * 1.8 / vec2(win_x, win_y);
  vec4 clipPos = centerClip;
  clipPos.xy += ndcOffset * centerClip.w;

  clipPos.z = 0.5 * clipPos.w;
  gl_Position = clipPos;

  float invDist = 1.0 / dist;
  outRo = relative_cam_pos * invDist;
  outWorldPos = localOffset * invDist;
  outScale = scale * invDist;
}
