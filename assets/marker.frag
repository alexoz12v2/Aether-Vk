#version 450 core
layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec4 outColor;

void main() {
  float dist = length(inUV);
  float alpha = smoothstep(1.0, 0.85, dist);
  if (alpha < 0.01) discard;
  
  // Add a nice white border using SDF
  float border = smoothstep(0.85, 0.75, dist);
  vec3 color = mix(vec3(1.0), inColor, border);
  
  outColor = vec4(color, alpha);
}