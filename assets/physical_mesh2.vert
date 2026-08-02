#version 450 core

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outTangent;
layout(location = 4) out vec3 outBitangent;

void main() {
    mat4 model = push.object.model;
    mat4 viewProj = push.scene.viewProj;

    vec4 worldPos = model * vec4(inPosition, 1.0);
    outWorldPos = worldPos.xyz;
    outUV = inUV;

    mat3 normalMatrix = mat3(model); // Assuming uniform scaling
    outNormal = normalize(normalMatrix * inNormal);
    outTangent = normalize(normalMatrix * inTangent.xyz);
    outBitangent = cross(outNormal, outTangent) * inTangent.w;

    vec4 clipPos = viewProj * worldPos;

    // Emissive outline expansion hack (triggers on negative intensity)
    if (push.material.emissiveColor.a < 0.0) {
        vec4 normalClip = viewProj * vec4(outNormal, 0.0);
        vec2 offset = normalize(normalClip.xy) * 0.015 * clipPos.w;
        clipPos.xy += offset;
    }

    gl_Position = clipPos;
}