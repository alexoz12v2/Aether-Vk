#version 450 core

layout(push_constant) uniform Push {
    mat4 modelViewProj;
    mat4 modelInv; // Inverse model matrix to transform camera to local space
    vec3 cameraWorldPos;
} push;

layout(location = 0) out vec3 outLocalPos;
layout(location = 1) out vec3 outLocalCameraPos;

// 14 vertices for a triangle strip cube
const vec3 cube_vertices[14] = vec3[](
    vec3(-1.0,  1.0,  1.0), // Front-top-left
    vec3( 1.0,  1.0,  1.0), // Front-top-right
    vec3(-1.0, -1.0,  1.0), // Front-bottom-left
    vec3( 1.0, -1.0,  1.0), // Front-bottom-right
    vec3( 1.0, -1.0, -1.0), // Back-bottom-right
    vec3( 1.0,  1.0,  1.0), // Front-top-right
    vec3( 1.0,  1.0, -1.0), // Back-top-right
    vec3(-1.0,  1.0,  1.0), // Front-top-left
    vec3(-1.0,  1.0, -1.0), // Back-top-left
    vec3(-1.0, -1.0,  1.0), // Front-bottom-left
    vec3(-1.0, -1.0, -1.0), // Back-bottom-left
    vec3( 1.0, -1.0, -1.0), // Back-bottom-right
    vec3(-1.0,  1.0, -1.0), // Back-top-left
    vec3( 1.0,  1.0, -1.0)  // Back-top-right
);

void main() {
    vec3 inPosition = cube_vertices[gl_VertexIndex];
    outLocalPos = inPosition;
    
    // Transform camera to local space for raymarching
    vec4 localCam = push.modelInv * vec4(push.cameraWorldPos, 1.0);
    outLocalCameraPos = localCam.xyz / localCam.w;
    
    gl_Position = push.modelViewProj * vec4(inPosition, 1.0);
}