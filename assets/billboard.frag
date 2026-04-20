#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 inUV;
layout(location = 1) flat in uint inTextureId;

layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform sampler2D textures[];

void main() {
  vec4 texColor = texture(textures[nonuniformEXT(inTextureId)], inUV);
  if (texColor.a < 0.05) {
    discard;
  }
  outColor = texColor;
}
