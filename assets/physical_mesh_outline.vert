#version 450 core

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

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

layout(location = 0) out vec3 outColor;

void main() {
    vec4 clipPos = push.modelViewProj * vec4(inPosition, 1.0);

    // Transform the normal to clip space.
    // Setting W to 0.0 applies rotation/scaling but ignores translation.
    vec4 clipNormal = push.modelViewProj * vec4(inNormal, 0.0);

    // We only care about the 2D direction on the screen
    float len = length(clipNormal.xy);

    if (len > 0.0001) {
        vec2 screenNormalDir = clipNormal.xy / len;
        float outlineThickness = 0.015;

        // Multiply back by clipPos.w to keep the outline thickness constant
        // in screen-space, regardless of distance from the camera.
        float maxDistance = 10.0;
        float distanceFactor = min(clipPos.w, maxDistance);
        vec2 offset = screenNormalDir * outlineThickness * distanceFactor;

        // offset.x /= aspectRatio; // Apply aspect ratio correction here if needed
        clipPos.xy += offset;
    }

    gl_Position = clipPos;
    outColor = push.extra.emissiveColor;
}