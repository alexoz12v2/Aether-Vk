#version 450 core
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent; // Added for Normal Mapping (w is handedness)

// Shared Push Constant block (Exactly 160 bytes)
layout(push_constant) uniform Push { // std140
  mat4 modelViewProj; // 64 bytes
  mat4 model;         // 64 bytes
  vec3 sunDir;        // 12 bytes
  uint textureFlags;  // 4 bytes (packs perfectly with vec3)
  vec4 sunColor;      // 16 bytes
} push;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outTangent;
layout(location = 4) out vec3 outBitangent;

void main() {
  vec4 worldPos = push.model * vec4(inPosition, 1.0);
  outWorldPos = worldPos.xyz;
  outUV = inUV;

  // Construct the TBN vectors for normal mapping
  mat3 normalMatrix = mat3(push.model); // Assuming uniform scaling
  outNormal = normalMatrix * inNormal;
  outTangent = normalMatrix * inTangent.xyz;

  // Calculate bitangent using the normal, tangent, and handedness sign
  outBitangent = cross(outNormal, outTangent) * inTangent.w;

  gl_Position = push.modelViewProj * vec4(inPosition, 1.0);
}