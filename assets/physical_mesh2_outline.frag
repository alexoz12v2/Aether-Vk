#version 450 core

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 inColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(inColor, 1.0);
}
