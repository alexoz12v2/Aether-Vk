#version 450 core

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBitangent;

layout(location = 0) out vec4 outColor;

// Texture bindings
layout(binding = 0) uniform sampler2D albedoMap;
layout(binding = 1) uniform sampler2D normalMap;
layout(binding = 2) uniform sampler2D roughnessMap;
layout(binding = 3) uniform sampler2D aoMap;
layout(binding = 4) uniform sampler2D skyMap;

layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer MeshExtra {
  mat4 model;
  vec3 sunPos;
  uint textureFlags;
  vec4 sunColor;
  vec3 cameraPos;
  float emissiveIntensity;
  vec3 emissiveColor;
};

layout(push_constant, std430) uniform Push {
  mat4 modelViewProj;
  MeshExtra extra;
} push;

// --- Specialization Constants ---
// Note: GLSL requires these to be scalar types.
layout(constant_id = 0) const float BASE_ALBEDO_R = 0.04;
layout(constant_id = 1) const float BASE_ALBEDO_G = 0.04;
layout(constant_id = 2) const float BASE_ALBEDO_B = 0.04;
layout(constant_id = 3) const float BASE_ROUGHNESS = 0.9;
layout(constant_id = 4) const float BASE_AO = 1.0;

// Bitfield definitions
const uint FLAG_ALBEDO    = 1u << 0;
const uint FLAG_NORMAL    = 1u << 1;
const uint FLAG_ROUGHNESS = 1u << 2;
const uint FLAG_AO        = 1u << 3;

// Oren-Nayar approximation
vec3 orenNayar(vec3 viewDir, vec3 lightDir, vec3 normal, vec3 albedo, float roughness) {
  float VdotN = max(dot(viewDir, normal), 0.0);
  float LdotN = max(dot(lightDir, normal), 0.0);
  float cosThetaI = LdotN;
  float cosThetaR = VdotN;

  float thetaI = acos(cosThetaI);
  float thetaR = acos(cosThetaR);
  float alpha = max(thetaI, thetaR);
  float beta = min(thetaI, thetaR);

  float roughness2 = roughness * roughness;
  float A = 1.0 - 0.5 * (roughness2 / (roughness2 + 0.33));
  float B = 0.45 * (roughness2 / (roughness2 + 0.09));

  vec3 v_perp = normalize(viewDir - normal * VdotN);
  vec3 l_perp = normalize(lightDir - normal * LdotN);
  float cosPhi = max(dot(v_perp, l_perp), 0.0);

  return albedo * (A + B * cosPhi * sin(alpha) * tan(beta)) / 3.14159;
}

void main() {
  bool useNormal    = (push.extra.textureFlags & FLAG_NORMAL) != 0u;

  vec3 N = normalize(inNormal);

  if (useNormal) {
    vec3 T = normalize(inTangent);
    vec3 B = normalize(inBitangent);
    mat3 TBN = mat3(T, B, N);

    vec3 sampledNormal = texture(normalMap, inUV).xyz;
    sampledNormal = sampledNormal * 2.0 - 1.0;
    N = normalize(TBN * sampledNormal);
  }

  vec3 normalColor = normalize(N.xyz) * 0.5 + 0.5;

  outColor = vec4(normalColor, 1.0);
}