#version 450 core

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec3 outColor;

void main() {
    mat4 viewProj = push.scene.viewProj;
    mat4 model = push.object.model;

    vec4 worldPos = model * vec4(inPosition, 1.0);
    vec4 clipPos = viewProj * worldPos;

    mat3 normalMatrix = mat3(model); // Assuming uniform scaling
    vec3 worldNormal = normalMatrix * inNormal;

    vec4 clipNormal = viewProj * vec4(worldNormal, 0.0);

    float aspect = push.scene.windowExtent.x / push.scene.windowExtent.y;
    vec2 screenNormal = clipNormal.xy;
    screenNormal.x *= aspect;

    float len = length(screenNormal);
    if (len > 0.0001) {
        vec2 screenNormalDir = screenNormal / len;
        float outlineThickness = 0.03;
        // Adjust outline to handle aspect ratio automatically using windowExtent.
        vec2 offset = screenNormalDir * outlineThickness * clipPos.w;
        offset.x /= aspect;
        clipPos.xy += offset;
    }

    gl_Position = clipPos;
    outColor = push.material.emissiveColor.rgb;
}
