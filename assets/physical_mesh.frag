#version 450 core

// See Guide on Vulkan extensions for GLSL
// https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GL_KHR_vulkan_glsl.txt

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

layout(push_constant) uniform Push { // std140
  mat4 modelViewProj;
  mat4 model;
  vec3 sunDir;
  uint textureFlags;
  vec4 sunColor;
  vec3 cameraPos;
  uint _unused;
} push;

// --- Specialization Constants ---
// Note: GLSL requires these to be scalar types.
layout(constant_id = 0) const float BASE_ALBEDO_R = 0.8;
layout(constant_id = 1) const float BASE_ALBEDO_G = 0.8;
layout(constant_id = 2) const float BASE_ALBEDO_B = 0.8;
layout(constant_id = 3) const float BASE_ROUGHNESS = 0.9;
layout(constant_id = 4) const float BASE_AO = 1.0;

// Bitfield definitions
const uint FLAG_ALBEDO    = 1u << 0;
const uint FLAG_NORMAL    = 1u << 1;
const uint FLAG_ROUGHNESS = 1u << 2;
const uint FLAG_AO        = 1u << 3;

// Oren-Nayar Fujii approximation
vec3 orenNayar(vec3 viewDir, vec3 lightDir, vec3 normal, vec3 albedo, float roughness) {
  float LdotN = max(dot(lightDir, normal), 0.0);
  float VdotN = max(dot(viewDir, normal), 0.0);

  // Early exit for the dark side of the mesh! No infinities.
  if (LdotN <= 0.0) {
    return vec3(0.0);
  }

  float roughness2 = roughness * roughness;
  float A = 1.0 - 0.5 * (roughness2 / (roughness2 + 0.33));
  float B = 0.45 * (roughness2 / (roughness2 + 0.09));

  // This block replaces all the acos(), tan(), and normalize() danger zones
  float LdotV = dot(lightDir, viewDir);
  float s = LdotV - LdotN * VdotN;
  float t = mix(1.0, max(LdotN, VdotN), step(0.0, s));

  return albedo * LdotN * (A + B * s / t) / 3.14159;
}

void main() {
  bool useAlbedo    = (push.textureFlags & FLAG_ALBEDO) != 0u;
  bool useNormal    = (push.textureFlags & FLAG_NORMAL) != 0u;
  bool useRoughness = (push.textureFlags & FLAG_ROUGHNESS) != 0u;
  bool useAO        = (push.textureFlags & FLAG_AO) != 0u;

  vec3 V = normalize(push.cameraPos - inWorldPos);
  vec3 lightDir = normalize(push.sunDir);
  vec3 lightColor = push.sunColor.xyz;

  // Construct base values from specialization constants
  vec3 albedo = vec3(BASE_ALBEDO_R, BASE_ALBEDO_G, BASE_ALBEDO_B);
  float roughness = BASE_ROUGHNESS;
  float ao = BASE_AO;
  vec3 N = normalize(inNormal);

  if (useAlbedo) {
    albedo = texture(albedoMap, inUV).rgb;
  }

  if (useNormal) {
    vec3 T = normalize(inTangent);
    vec3 B = normalize(inBitangent);
    mat3 TBN = mat3(T, B, N);

    vec3 sampledNormal = texture(normalMap, inUV).xyz;
    sampledNormal = sampledNormal * 2.0 - 1.0;
    N = normalize(TBN * sampledNormal);
  }

  if (useRoughness) {
    roughness = texture(roughnessMap, inUV).r;
  }

  if (useAO) {
    ao = texture(aoMap, inUV).r;
  }

  vec3 diffuse = orenNayar(V, lightDir, N, albedo, roughness);

  outColor = vec4(diffuse * lightColor * ao, 1.0);
}