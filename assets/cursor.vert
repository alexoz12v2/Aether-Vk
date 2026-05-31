#version 450 core

layout(push_constant) uniform Push {
  mat4 viewProj;
  vec4 right_proj11;
  vec4 up_win_x;
  vec4 relative_cam_pos_win_y;
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

  // Unpack CPU-precomputed data from vec4s
  vec3 right = normalize(push.right_proj11.xyz);
  float proj11 = push.right_proj11.w;
  
  vec3 up = normalize(push.up_win_x.xyz);
  float win_x = push.up_win_x.w;
  
  vec3 relative_cam_pos = push.relative_cam_pos_win_y.xyz;
  float win_y = push.relative_cam_pos_win_y.w;

  // In RTE rendering the camera is at origin. Cursor position relative to camera:
  vec3 cursorPos = -relative_cam_pos;
  float dist = max(length(relative_cam_pos), 1e-10);

  // Log-based scaling: cursor size adapts smoothly across AU to km range.
  float logDist = log(dist * 1e6) / log(10.0);
  float t = clamp(logDist / 12.0, 0.0, 1.0);
  float pct = mix(0.07, 0.12, t);
  float desiredSizePixels = min(win_x, win_y) * pct;

  // Compute world-space scale using projection focal length.
  // proj11 = -f where f = 1/tan(fov/2), so |proj11| = f.
  float fov_tan = 1.0 / max(abs(proj11), 1e-6);
  float scale = dist * fov_tan * (desiredSizePixels / win_y);

  // Billboard offset in world space (for fragment shader raycasting)
  vec3 localOffset = right * uv.x * scale * 1.8 
                   + up    * uv.y * scale * 1.8;

  // --- Clip position via NDC offset (avoids near-zero-w singularity) ---
  // Project ONLY the cursor center through viewProj. At extreme zoom the
  // cursor center is near the camera so clipPos.w is small but stable for
  // a single point. We then add the billboard offset in NDC pixel-space,
  // avoiding the w instability of projecting corner vertices independently.
  vec4 centerClip = push.viewProj * vec4(cursorPos, 1.0);

  // Cull if behind camera
  if (centerClip.w <= 0.0) {
    gl_Position = vec4(0.0);
    outScale = 0.0;
    return;
  }

  // NDC offset in pixels, converted to clip-space.
  // Y is NEGATED to correct for the projection's Y-flip:
  //   AetherVk maps view +Z (up) to clip -Y, so world-up = screen-up = -NDC_Y.
  //   Without this negation, the fragment shader's raycasted pattern would be
  //   Y-flipped relative to the screen position, causing apparent rotation.
  vec2 ndcOffset = vec2(uv.x, -uv.y) * desiredSizePixels * 1.8 / vec2(win_x, win_y);

  // Scale NDC offset by w to stay in clip-space (NDC = clip.xy / clip.w)
  vec4 clipPos = centerClip;
  clipPos.xy += ndcOffset * centerClip.w;

  // Override depth for NO_DEPTH_TEST rendering (draw on top of everything)
  clipPos.z = 0.5 * clipPos.w;
  gl_Position = clipPos;

  // Normalize all fragment inputs by dist to prevent precision loss at extreme zoom.
  // The raycasting math produces identical UVs because the scaling cancels out:
  //   uv = p.component / scale  ->  (p/dist).component / (scale/dist)  =  same result
  // This keeps all interpolated values O(1) regardless of camera-cursor distance.
  float invDist = 1.0 / dist;
  outRo = relative_cam_pos * invDist;
  outWorldPos = localOffset * invDist;
  outScale = scale * invDist;
}
