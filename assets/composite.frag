#version 450 core

// Compositing fragment shader: merges macro and micro layers by linearizing
// their depths and picking the nearer fragment per pixel.
//
// Uses subpass input attachments for zero-copy reads on tile-based GPUs.
// MRT output[1] (finalGlobalDepth) carries (layer_index_f32, local_distance)
// of the winning fragment for CPU-side UI occlusion queries.

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput macroColor;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput macroDepth;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput macroGlobalDepth;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput microColor;
layout(input_attachment_index = 4, set = 0, binding = 4) uniform subpassInput microDepth;
layout(input_attachment_index = 5, set = 0, binding = 5) uniform subpassInput microGlobalDepth;

layout(push_constant, std430) uniform CompositePush {
    float macroNear;
    float macroFar;
    float microNear;
    float microFar;
    float macroScale;
    float microScale;
};

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 finalGlobalDepth; // (layer_index_f32, local_distance) of winner

// Linearize a reverse-Z depth value to view-space distance.
// Reverse-Z:  depth = 1.0 -> at near plane,  depth = 0.0 -> at far plane.
// Returns the physical distance from the camera.
float linearizeReverseZ(float d, float near, float far) {
    return (near * far) / mix(near, far, d);
}

void main() {
    vec4 cMacro = subpassLoad(macroColor);
    float dMacro = subpassLoad(macroDepth).r;
    vec2 gdMacro = subpassLoad(macroGlobalDepth).rg;
    vec4 cMicro = subpassLoad(microColor);
    float dMicro = subpassLoad(microDepth).r;
    vec2 gdMicro = subpassLoad(microGlobalDepth).rg;

    // Linearize both depths to physical distance (in layer-local scale * macroScale/microScale = AU)
    float distMacro = linearizeReverseZ(dMacro, macroNear, macroFar) * macroScale;
    float distMicro = linearizeReverseZ(dMicro, microNear, microFar) * microScale;

    // Pick the fragment that is nearer to the camera.
    // When a layer has no content at a pixel, its depth is 0.0 (reverse-Z clear value).
    // However, some pipelines (e.g. sphere gizmo wireframes) write color but NOT depth
    // (NO_DEPTH_WRITE). For those pixels, depth stays at clear value but color is valid.
    if (dMicro == 0.0 && cMicro.a == 0.0) {
        // Micro layer is truly empty (no color, no depth), use Macro
        outColor = cMacro;
        finalGlobalDepth = gdMacro;
    } else if (dMicro == 0.0 && cMicro.a > 0.0) {
        // Micro layer has color but no depth (e.g. wireframe gizmo, or depth was cleared
        // between layers) — blend over macro. Use gdMicro since micro content exists here.
        outColor = vec4(mix(cMacro.rgb, cMicro.rgb, cMicro.a), max(cMacro.a, cMicro.a));
        finalGlobalDepth = gdMicro; // micro content present — prefer micro global depth
    } else if (dMacro == 0.0) {
        // Macro layer is empty, use Micro
        outColor = cMicro;
        finalGlobalDepth = gdMicro;
    } else if (distMacro <= distMicro) {
        outColor = cMacro;
        finalGlobalDepth = gdMacro;
    } else {
        outColor = cMicro;
        finalGlobalDepth = gdMicro;
    }
}
