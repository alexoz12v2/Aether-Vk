#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(push_constant) uniform Push {
  mat4 viewProj;
  float scale;
  uint instance_id;
} push;

layout(set = 0, binding = 0) readonly buffer Gizmos {
  mat4 model;
} gizmos[];

layout(location = 0) out vec3 outColor;

void main() {
  mat4 model = gizmos[nonuniformEXT(push.instance_id)].model;

  vec3 pos = vec3(0.0);
  vec3 color = vec3(0.0);
  
  if (gl_VertexIndex == 0) {
    pos = vec3(0.0);
    color = vec3(1.0, 0.0, 0.0); // Red (x)
  } else if (gl_VertexIndex == 1) {
    pos = vec3(push.scale, 0.0, 0.0);
    color = vec3(1.0, 0.0, 0.0);
  } else if (gl_VertexIndex == 2) {
    pos = vec3(0.0);
    color = vec3(0.0, 1.0, 0.0); // Green (y)
  } else if (gl_VertexIndex == 3) {
    pos = vec3(0.0, push.scale, 0.0);
    color = vec3(0.0, 1.0, 0.0);
  } else if (gl_VertexIndex == 4) {
    pos = vec3(0.0);
    color = vec3(0.0, 0.0, 1.0); // Blue (z)
  } else if (gl_VertexIndex == 5) {
    pos = vec3(0.0, 0.0, push.scale);
    color = vec3(0.0, 0.0, 1.0);
  }

  vec4 worldPos = model * vec4(pos, 1.0);
  gl_Position = push.viewProj * worldPos;
  outColor = color;
}
