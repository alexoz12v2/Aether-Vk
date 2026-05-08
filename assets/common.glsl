#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// Layout 'scalar' makes GLSL padding perfectly match standard Rust C-structs.
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer SceneData {
    mat4 viewProj; // Passing ViewProj instead of ModelViewProj saves CPU cycles
    vec4 cameraPos;
    vec4 sunPos;
    vec4 sunColor;
    vec2 windowExtent;
    vec2 _pad;
};

layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer MaterialData {
    vec4 baseAlbedo; // w is roughness
    vec4 emissiveColor; // w is intensity
    float baseAO;
    
    // 0 = Normal PBR (RGB is used as standard emissive glow)
    // 1 = Color Paint Mode (Unlit RGB Visualization)
    // 2 = Distribution Paint Mode (Unlit Alpha-to-Grayscale Visualization)
    uint paintDisplayMode;
    uint textureFlags;
    float _pad0;
};

layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer ObjectData {
    mat4 model;
};

// Total Size: Exactly 24 bytes! (Three 64-bit pointers)
layout(push_constant, scalar) uniform Push {
    SceneData scene;
    MaterialData material;
    ObjectData object;
} push;