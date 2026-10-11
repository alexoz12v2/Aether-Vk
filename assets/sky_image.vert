#version 450 core

// Sky image overlay (HUD): a telescope image registered on the sky through its WCS
// (`scene/sky_image.rs`). The vertices are points of the image's tangent plane in ecliptic J2000
// (directions, not normalised) drawn at infinity: `w = 0` under the camera's real projection and
// its rotation-only view. Pixel → tangent plane is affine and the projection is a pinhole, so the
// rasteriser's perspective-correct interpolation of `inUV` registers every texel exactly.

layout(push_constant) uniform Push {
  mat4 viewProj; // proj · view (rotation only)
  vec4 params;   // opacity, asinh stretch, black, white (normalised units)
} push;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 outUV;

void main() {
  outUV = inUV;
  vec4 c = push.viewProj * vec4(inPos, 0.0);
  // depth is unused (no depth test / write); z = w / 2 keeps 0 ≤ z ≤ w exactly when w ≥ 0, so
  // clipping removes what lies behind the eye
  c.z = 0.5 * c.w;
  gl_Position = c;
}
