#version 450 core

// See Guide on Vulkan extensions for GLSL
// https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GL_KHR_vulkan_glsl.txt

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent; // Added for Normal Mapping (w is handedness)

// Shared Push Constant block (Exactly 176 bytes TODO: move to inline unif?)
// Note: cameraPos used for IBL view vector computation in fragment shader
layout(push_constant) uniform Push {
  mat4 modelViewProj;
  mat4 model;
  vec3 sunPos;
  uint textureFlags;
  vec4 sunColor;
  vec3 cameraPos;
  float emissiveIntensity;
  vec3 emissiveColor;
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

  vec4 clipPos = push.modelViewProj * vec4(inPosition, 1.0);

  if (push.emissiveIntensity < 0.0) {
      vec4 normalClip = push.modelViewProj * vec4(outNormal, 0.0);
      vec2 offset = normalize(normalClip.xy) * 0.015 * clipPos.w;
      clipPos.xy += offset;
  }

  gl_Position = clipPos;
}