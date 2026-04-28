#version 450 core

layout(location = 0) in vec3 inRo;
layout(location = 1) in vec3 inWorldPos;
layout(location = 2) in float inScale;

layout(location = 0) out vec4 outColor;

void main() {
  outColor = vec4(1.0);
}