#version 450 core

// Fullscreen triangle — no vertex buffer needed.
// Emits a triangle covering the entire screen:
//   vertex 0: (-1, -1)   UV (0, 0)
//   vertex 1: ( 3, -1)   UV (2, 0)
//   vertex 2: (-1,  3)   UV (0, 2)
// The GPU clips to the viewport, giving a fullscreen quad.

layout(location = 0) out vec2 outUV;

void main() {
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
    // Flip Y for Vulkan (top-left origin)
    outUV.y = 1.0 - outUV.y;
}
