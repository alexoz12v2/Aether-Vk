#version 450 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(push_constant) uniform Push {
  mat4 modelViewProj; // 64 bytes
  mat4 model;         // 64 bytes
  vec3 sunPos;        // 12 bytes
  uint textureFlags;  // 4 bytes
  vec4 sunColor;      // 16 bytes
  vec3 cameraPos;     // 12 bytes
  float emissiveIntensity; // 4 bytes
  vec3 emissiveColor; // 12 bytes
} push;

layout(location = 0) out vec3 outColor;

void main() {
    mat3 normalMatrix = mat3(push.model);
    vec3 worldNormal = normalize(normalMatrix * inNormal);

    vec4 clipPos = push.modelViewProj * vec4(inPosition, 1.0);
    vec4 normalClip = push.modelViewProj * vec4(worldNormal, 0.0);

    vec2 offset = normalize(normalClip.xy) * 0.015 * clipPos.w;
    clipPos.xy += offset;

    gl_Position = clipPos;

    outColor = push.emissiveColor;
}