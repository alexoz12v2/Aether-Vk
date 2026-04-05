#version 450 core

layout(push_constant) uniform Push {
  mat4 view;         // 64 bytes
  mat4 viewProj;     // 64 bytes
  vec3 cursorPos;    // 12 bytes
  float cursorSize;  // 4 bytes | 0.05 -> 5% screen
} push;

// Note: no input. This will get called with an indexed draw with 4 indices, no vertex data
// `gl_VertexIndex` will tell us which corner of the BillBoard we are rendering

layout(location = 0) out vec2 outUV;

// Quad vertices for a triangle strip
const vec2 quad[4] = vec2[] (
  vec2(-1.0, -1.0),
  vec2( 1.0, -1.0),
  vec2(-1.0,  1.0),
  vec2( 1.0,  1.0)
);

void main() {
  vec2 uv = quad[gl_VertexIndex];
  outUV = uv;

  // Camera basis (world space)
  vec3 right = vec3(push.view[0][0], push.view[1][0], push.view[2][0]);
  vec3 up    = vec3(push.view[0][1], push.view[1][1], push.view[2][1]);

  // Center in view space (needed for proper scaling)
  vec4 viewPos = push.view * vec4(push.cursorPos, 1.0);

  // Perspective scale factor (this is the magic)
  float scale = -viewPos.z * push.cursorSize;

  // Build billboard in world space
  vec3 worldPos = push.cursorPos
                + right * uv.x * scale
                + up    * uv.y * scale;

  gl_Position = push.viewProj * vec4(worldPos, 1.0);
}
