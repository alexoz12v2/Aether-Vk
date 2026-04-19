#version 450 core

layout(location = 0) in vec2 inUV;

layout(binding = 0) uniform sampler2D textMap;

layout(push_constant) uniform Push {
    vec2 pos;
    vec2 scale;
    vec4 color;
    vec4 uv_bounds;
} push;

layout(location = 0) out vec4 outColor;

void main() {
    float alpha = 1.0;
    if (push.uv_bounds.z >= 0.0) {
        alpha = texture(textMap, inUV).r;
    }
    outColor = vec4(push.color.rgb, push.color.a * alpha);
}
