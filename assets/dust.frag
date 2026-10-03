// @assets/dust.frag
//
// Dust v3: gaussian splat blended premultiplied "over" (PipelineFlags::PREMULTIPLIED_BLEND).
// With the stream color c and per-fragment opacity a_i, a pixel converges to
//   c * (1 - prod(1 - a_i))  ~  c * (1 - exp(-sum a_i))
// i.e. linear (like additive light) where the dust is faint, saturating exactly at the stream
// color where it is dense: an exposure curve instead of a per-channel clip to white. Order
// independent for one color. Alpha accumulates the same way, so the composite pass (premultiplied
// over, empty-without-alpha test) sees the right coverage.
#version 450 core

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in float v_opacity;

layout(location = 0) out vec4 fragColor;

// 1 / integral of exp(-4 r^2) over the unit disc = 1 / (pi/4 (1 - e^-4))
const float GAUSS_NORM = 1.297;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;
    float a = clamp(v_opacity * exp(-4.0 * r2) * GAUSS_NORM, 0.0, 1.0);
    fragColor = vec4(v_color * a, a);
}
