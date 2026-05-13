#version 450 core
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    vec4 color_top;
    vec4 color_bottom;
} pc;

void main() {
    outColor = mix(pc.color_bottom, pc.color_top, inUV.y);
}