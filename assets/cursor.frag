#version 450 core

layout(location = 0) in vec3 inRo;
layout(location = 1) in vec3 inWorldPos;
layout(location = 2) in float inScale;

layout(location = 0) out vec4 outColor;

vec4 getCursorColor(vec2 uv) {
  float dist = length(uv);
  float ringThickness = 0.08;
  float ring = smoothstep(1.0, 1.0 - 0.02, dist) - smoothstep(1.0 - ringThickness, 1.0 - ringThickness - 0.02, dist);

  float angle = atan(uv.y, uv.x);
  float dashes = step(0.0, sin(angle * 16.0)); 
  float ringAlpha = ring * dashes;

  float crossThickness = 0.03;
  float crossLength = 1.3;
  float crossX = step(abs(uv.y), crossThickness) * step(abs(uv.x), crossLength);
  float crossY = step(abs(uv.x), crossThickness) * step(abs(uv.y), crossLength);
  float crosshairAlpha = clamp(crossX + crossY, 0.0, 1.0);

  float totalAlpha = clamp(ringAlpha + crosshairAlpha, 0.0, 1.0);
  
  vec3 ringColor = vec3(0.0);
  vec3 crossColor = vec3(1.0, 0.2, 0.2);
  vec3 col = mix(ringColor, crossColor, crosshairAlpha);
  
  return vec4(col, totalAlpha);
}

void main() {
  vec3 rd = normalize(inWorldPos - inRo);
  
  float t1 = -1.0;
  if (abs(rd.x) > 1e-6) t1 = -inRo.x / rd.x;
  
  float t2 = -1.0;
  if (abs(rd.z) > 1e-6) t2 = -inRo.z / rd.z;
  
  vec4 color1 = vec4(0.0);
  if (t1 > 0.0) {
    vec3 p1 = inRo + t1 * rd;
    if (abs(p1.y) < inScale && abs(p1.z) < inScale) {
      vec2 uv = vec2(p1.z, p1.y) / inScale;
      color1 = getCursorColor(uv);
    }
  }
  
  vec4 color2 = vec4(0.0);
  if (t2 > 0.0) {
    vec3 p2 = inRo + t2 * rd;
    if (abs(p2.x) < inScale && abs(p2.y) < inScale) {
      vec2 uv = vec2(p2.x, p2.y) / inScale;
      color2 = getCursorColor(uv);
    }
  }

  if (color1.a < 0.05 && color2.a < 0.05) discard;

  bool p1Front = (t1 > 0.0 && t2 > 0.0 && t1 < t2) || (t1 > 0.0 && t2 <= 0.0);
  vec4 frontColor = p1Front ? color1 : color2;
  vec4 backColor  = p1Front ? color2 : color1;
  
  vec4 finalColor = mix(backColor, frontColor, frontColor.a);
  if (finalColor.a < 0.05) discard;
  
  outColor = finalColor;
}
