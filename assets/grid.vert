#version 450 core

layout(location = 0) out vec2 outUV;

const vec2 quad[4] = vec2[] (
  vec2(-1.0, -1.0),
  vec2(-1.0,  1.0),
  vec2( 1.0, -1.0),
  vec2( 1.0,  1.0)
);

void main() {
  vec2 uv = quad[gl_VertexIndex];
  outUV = uv;
  gl_Position = vec4(uv, 0.0, 1.0);
}
