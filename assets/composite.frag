#version 450 core

// Compositing fragment shader: merges macro and micro layers by linearizing
// their depths and picking the nearer fragment per pixel.
//
// Uses subpass input attachments for zero-copy reads on tile-based GPUs.

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput macroColor;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput macroDepth;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput microColor;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput microDepth;

layout(push_constant, std430) uniform CompositePush {
    float macroNear;
    float macroFar;
    float microNear;
    float microFar;
};

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// Linearize a reverse-Z depth value to view-space distance.
// Reverse-Z:  depth = 1.0 → at near plane,  depth = 0.0 → at far plane.
// Returns the physical distance from the camera.
float linearizeReverseZ(float d, float near, float far) {
    // For reverse-Z with the standard Vulkan infinite/finite far projection:
    //   z_ndc = near / distance  (for infinite far)
    //   z_ndc = near * (far - distance) / (distance * (far - near))  (finite far)
    //
    // Inverting for finite far:
    //   distance = near * far / (far * d + near * (1.0 - d))
    //            = near * far / mix(near, far, d)      [when d=1 → near, d=0 → far]
    return (near * far) / mix(near, far, d);
}

void main() {
    vec4 cMacro = subpassLoad(macroColor);
    float dMacro = subpassLoad(macroDepth).r;
    vec4 cMicro = subpassLoad(microColor);
    float dMicro = subpassLoad(microDepth).r;

    // Linearize both depths to physical distance (AU)
    float distMacro = linearizeReverseZ(dMacro, macroNear, macroFar);
    float distMicro = linearizeReverseZ(dMicro, microNear, microFar);

    // Pick the fragment that is nearer to the camera.
    // When a layer has no content at a pixel, its depth is 0.0 (reverse-Z clear value).
    // linearizeReverseZ(0.0, near, far) = far → maximum distance → other layer wins.
    if (distMacro <= distMicro) {
        outColor = cMacro;
    } else {
        outColor = cMicro;
    }
}
