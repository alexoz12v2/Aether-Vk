#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 inUV;
layout(location = 1) flat in uint inTextureId;

// TODO: insert extension GL_EXT_nonuniform_qualifier and use an
// TODO: array of textMaps sampled with nonuniformEXT function
layout(binding = 0) uniform sampler2D[] textureAtlases;

layout(push_constant) uniform Push {
    vec2 pos;
    vec2 scale;
    vec4 color;
    vec4 uv_bounds;
    uint texture_id;
} push;

layout(location = 0) out vec4 outColor;

void main() {
    float alpha = 1.0;
    if (push.uv_bounds.z >= 0.0) {
        alpha = texture(textureAtlases[nonuniformEXT(inTextureId)], inUV).r;
    }
    outColor = vec4(push.color.rgb, push.color.a * alpha);
}
