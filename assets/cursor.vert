#version 450 core

layout(push_constant) uniform Push {
  mat4 view;         
  mat4 viewProj;     
  mat4 model;
  float cursorSize;  
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

  vec3 right = vec3(push.view[0][0], push.view[1][0], push.view[2][0]);
  vec3 up    = vec3(push.view[0][1], push.view[1][1], push.view[2][1]);

  vec3 cursorPos = push.model[3].xyz;
  vec4 viewPos = push.view * vec4(cursorPos, 1.0);

  float scale = -viewPos.z * push.cursorSize;

  vec3 worldPos = cursorPos
                + right * uv.x * scale * 1.8
                + up    * uv.y * scale * 1.8;

  gl_Position = push.viewProj * vec4(worldPos, 1.0);

  mat4 invView = inverse(push.view);
  vec3 camPos = invView[3].xyz;

  outRo = camPos - cursorPos;
  outWorldPos = worldPos - cursorPos;
  outScale = scale;
}
