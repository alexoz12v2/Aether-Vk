#version 450 core

// Sky image overlay: normalised image value (black = 0, white = 1 at the 1st / 99.5th
// percentiles, `scene/sky_image.rs`) → asinh stretch → grey, blended over the scene with the
// user's opacity. Blank pixels (NaN) are transparent. The display value is sRGB encoded, the
// attachment is sRGB: decode it here so the screen shows the stretched value.

layout(push_constant) uniform Push {
  mat4 viewProj;
  vec4 params; // opacity, asinh stretch (≤ 1e-3: linear), black, white
} push;

layout(set = 0, binding = 0) uniform sampler2D skyImage;

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

float srgbToLinear(float c) {
  return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

void main() {
  float t = texture(skyImage, inUV).r;
  if (isnan(t)) {
    discard;
  }
  float x = clamp((t - push.params.z) / max(push.params.w - push.params.z, 1e-6), 0.0, 1.0);
  float s = push.params.y;
  float y = s > 1e-3 ? asinh(s * x) / asinh(s) : x;
  outColor = vec4(vec3(srgbToLinear(y)), clamp(push.params.x, 0.0, 1.0));
}
