#version 450 core
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D skyMap;

layout(push_constant) uniform Push {
  layout(offset = 0) mat4 invViewProj;
} push;

vec2 octEncode(vec3 v) {
  v /= (abs(v.x) + abs(v.y) + abs(v.z));
  vec2 uv = v.z >= 0.0 ? v.xy : (1.0 - abs(v.yx)) * sign(v.xy);
  return uv * 0.5 + 0.5;
}

void main() {
  vec4 ndc = vec4(inUV * 2.0 - 1.0, 1.0, 1.0);
  vec4 worldDir = push.invViewProj * ndc;
  vec3 dir = worldDir.w != 0.0 ? (worldDir.xyz / worldDir.w) : worldDir.xyz;
  outColor = vec4(texture(skyMap, octEncode(normalize(dir))).rgb, 1.0);
}
