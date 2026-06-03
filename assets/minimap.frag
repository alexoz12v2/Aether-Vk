#version 450 core

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

layout(location = 0) in vec2 inUV;

struct Planet {
    vec2 pos;
    float size;
    float pad;
    vec4 color;
};

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer PlanetArray {
    Planet planets[];
};

layout(push_constant, std430) uniform Push {
    vec2 offset;
    vec2 size;
    vec2 playerPos;
    float maxDistance;
    uint numPlanets;
    PlanetArray planetsPtr;
} push;

layout(location = 0) out vec4 outColor;

void main() {
    // 1. Semi-transparent rectangular background
    vec4 color = vec4(0.02, 0.02, 0.05, 0.65); // Dark blueish semi-transparent

    // 2. Outline rendering
    vec2 border = vec2(0.015);
    if (inUV.x < border.x || inUV.x > 1.0 - border.x || 
        inUV.y < border.y || inUV.y > 1.0 - border.y) {
        color = vec4(0.7, 0.7, 0.8, 1.0); // Light gray solid border
    } else {
        // Adjust aspect ratio for circular distance
        vec2 aspect = vec2(push.size.x / push.size.y, 1.0);
        
        // 3. Render planets
        for (uint i = 0; i < push.numPlanets && i < 16; ++i) {
            Planet p = push.planetsPtr.planets[i];
            
            // Map planet world pos to UV [0, 1]
            vec2 planetUV = (p.pos / (2.0 * push.maxDistance)) + vec2(0.5);
            
            float d = distance(inUV * aspect, planetUV * aspect);
            
            if (d < p.size) {
                // Anti-aliased circle
                float alpha = smoothstep(p.size, p.size - 0.005, d);
                color = mix(color, p.color, alpha);
            }
        }
        
        // 4. Render Player cross
        vec2 playerUV = (push.playerPos / (2.0 * push.maxDistance)) + vec2(0.5);
        vec2 dPlayer = abs(inUV - playerUV);
        
        // Compensate thickness for aspect ratio to keep the cross lines uniform
        vec2 crossThickness = vec2(0.004) / aspect;
        vec2 crossSize = vec2(0.025) / aspect;
        
        if ((dPlayer.x < crossThickness.x && dPlayer.y < crossSize.y) || 
            (dPlayer.y < crossThickness.y && dPlayer.x < crossSize.x)) {
            color = vec4(1.0, 0.2, 0.2, 1.0); // Red cross for player
        }
    }

    outColor = color;
}