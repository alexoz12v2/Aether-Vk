#version 450 core

layout(location = 0) in  vec3  fragColor;
layout(location = 1) in  float inAlpha;  // 1.0 = front hemisphere / axis, 0.0 = back hemisphere

layout(location = 0) out vec4 outColor;

void main() {
    // Discard back-hemisphere sphere-grid fragments.
    // inAlpha is 0.0 for back-hemisphere sphere lines and 1.0 for front-hemisphere lines
    // and all axis/arrowhead lines. Silhouette lines interpolate 1→0 across the equator,
    // so they are clipped at the 0.5 threshold — clean equatorial cutoff.
    if (inAlpha < 0.5)
        discard;

    outColor = vec4(fragColor, 1.0);
}
