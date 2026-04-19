#version 450 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(push_constant) uniform Push {
  layout(offset = 0) mat4 modelViewProj; // 64 bytes
  layout(offset = 64) mat4 model;         // 64 bytes
  layout(offset = 128) vec3 sunPos;        // 12 bytes
  layout(offset = 140) uint textureFlags;  // 4 bytes
  layout(offset = 144) vec4 sunColor;      // 16 bytes
  layout(offset = 160) vec3 cameraPos;     // 12 bytes
  layout(offset = 172) float emissiveIntensity; // 4 bytes
  layout(offset = 176) vec3 emissiveColor; // 12 bytes
} push;

layout(location = 0) out vec3 outColor;

void main() {
    mat3 normalMatrix = mat3(push.model);
    vec3 worldNormal = normalize(normalMatrix * inNormal);

    vec4 clipPos = push.modelViewProj * vec4(inPosition, 1.0);
    
    // Transform normal to clip space
    vec4 normalClip = push.modelViewProj * vec4(worldNormal, 0.0);

    float len = length(normalClip.xy);
    if (len > 0.0001) {
        vec2 offset = (normalClip.xy / len) * 0.015 * clipPos.w;
        clipPos.xy += offset;
    }

    gl_Position = clipPos;

    outColor = push.emissiveColor;
}