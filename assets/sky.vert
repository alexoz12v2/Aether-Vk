#version 450 core
layout(location = 0) out vec2 outUV;

void main() {
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    // Depth is 1.0 at far plane
    gl_Position = vec4(outUV * 2.0f - 1.0f, 1.0f, 1.0f);
}
