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
    vec4 clipPos = push.modelViewProj * vec4(inPosition, 1.0);

    // Transform the normal to clip space.
    // Setting W to 0.0 applies rotation/scaling but ignores translation.
    vec4 clipNormal = push.modelViewProj * vec4(inNormal, 0.0);

    // We only care about the 2D direction on the screen
    float len = length(clipNormal.xy);

    if (len > 0.0001) {
        vec2 screenNormalDir = clipNormal.xy / len;
        float outlineThickness = 0.15;

        // Multiply back by clipPos.w to keep the outline thickness constant
        // in screen-space, regardless of distance from the camera.
        vec2 offset = screenNormalDir * outlineThickness * clipPos.w;

        // offset.x /= aspectRatio; // Apply aspect ratio correction here if needed
        clipPos.xy += offset;
    }

    gl_Position = clipPos;
    outColor = push.emissiveColor;
}
