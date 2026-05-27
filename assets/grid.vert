#version 450 core

layout(push_constant) uniform Push {
  mat4 viewProj;
  vec3 cameraPos;
  float nearPlane;
  float farPlane;
  float density;
  vec3 gridColor;
} push;

layout(location = 0) out vec2 outNDC;

// A single full-screen triangle covers the viewport with no diagonal seam.
// gl_VertexIndex 0,1,2 produce NDC corners that fully enclose [-1,+1]^2.
// Using this instead of a 4-vertex TRIANGLE_STRIP eliminates the triangle-edge
// discontinuity that caused fwidth() to produce wrong values along the diagonal.
void main() {
  // Generates:
  //  0: (-1, -1)
  //  1: (-1,  3)
  //  2: ( 3, -1)
  vec2 uv = vec2(
    -1.0 + float((gl_VertexIndex & 1) << 2),
    -1.0 + float((gl_VertexIndex & 2) << 1)
  );

  outNDC = uv;
  gl_Position = vec4(uv.x, uv.y, 0.0, 1.0);
}