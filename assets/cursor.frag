#version 450 core

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
  float dist = length(inUV);

  // --- 1. Draw the Dashed Ring ---
  float ringThickness = 0.08;
  // Create a solid ring using smoothstep for anti-aliasing
  float ring = smoothstep(1.0, 1.0 - 0.02, dist) - smoothstep(1.0 - ringThickness, 1.0 - ringThickness - 0.02, dist);

  // Create the dashes using polar coordinates (atan)
  float angle = atan(inUV.y, inUV.x);
  // sin(angle * 16.0) creates 16 segments. step(0.0, ...) turns the sine wave into a hard on/off
  float dashes = step(0.0, sin(angle * 16.0)); 
  
  float ringAlpha = ring * dashes;

  // --- 2. Draw the Crosshair ---
  float crossThickness = 0.03;
  float crossLength = 1.3; // Extends slightly past the ring
  
  // X and Y lines
  float crossX = step(abs(inUV.y), crossThickness) * step(abs(inUV.x), crossLength);
  float crossY = step(abs(inUV.x), crossThickness) * step(abs(inUV.y), crossLength);
  float crosshairAlpha = clamp(crossX + crossY, 0.0, 1.0);

  // --- 3. Combine and Color ---
  float totalAlpha = clamp(ringAlpha + crosshairAlpha, 0.0, 1.0);
  if (totalAlpha < 0.05) {
    discard; // Save fill rate for empty pixels
  }

  // Colors: Ring is Black/White (depending on your scene), Crosshair is typically Black or Red
  vec3 ringColor = vec3(0.0); // Black dashed ring
  vec3 crossColor = vec3(1.0, 0.2, 0.2); // Red crosshair
  
  // Mix the colors based on which part we are drawing
  vec3 finalColor = mix(ringColor, crossColor, crosshairAlpha);

  outColor = vec4(finalColor, totalAlpha);
}