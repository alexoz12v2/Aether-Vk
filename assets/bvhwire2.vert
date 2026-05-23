#version 450 core
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

struct BvhData {
    vec4 center_type;   // xyz = center, w = type (0=Sphere, 1=AABB/OOBB)
    vec4 extents;       // xyz = half extents or radius, w = unused
    vec4 axes_x;        // xyz = axis x
    vec4 axes_y;        // xyz = axis y
    vec4 axes_z;        // xyz = axis z
};

// Bindless BDA block
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer BvhArray {
    BvhData bvh[];
};

// 72 bytes total! (Safely under 128 bytes limit)
layout(push_constant, std430) uniform PushConstants {
    BvhArray bvhPtr;    // 8 bytes
    mat4 viewProj;      // 64 bytes
} push;

layout(location = 0) out vec3 fragColor;

const int lineIndices[24] = int[](
    0,1, 1,2, 2,3, 3,0, // Bottom face
    4,5, 5,6, 6,7, 7,4, // Top face
    0,4, 1,5, 2,6, 3,7  // Vertical edges
);

void main() {
    BvhData data = push.bvhPtr.bvh[gl_InstanceIndex];
    int type = int(data.center_type.w);
    vec3 center = data.center_type.xyz;
    vec3 localPos = vec3(0.0);
    bool valid = true;

    if (type == 1) {
        // AABB / OOBB (24 vertices mapped)
        if (gl_VertexIndex < 24) {
            int corner = lineIndices[gl_VertexIndex];
            vec3 signs = vec3(
                (corner == 1 || corner == 2 || corner == 5 || corner == 6) ? 1.0 : -1.0,
                (corner == 2 || corner == 3 || corner == 6 || corner == 7) ? 1.0 : -1.0,
                (corner >= 4) ? 1.0 : -1.0
            );
            
            vec3 halfExtents = data.extents.xyz;
            localPos = signs * halfExtents;
            localPos = data.axes_x.xyz * localPos.x + data.axes_y.xyz * localPos.y + data.axes_z.xyz * localPos.z;
        } else {
            valid = false;
        }
    } else {
        // Sphere (216 vertices)
        if (gl_VertexIndex < 216) {
            int circle = gl_VertexIndex / 72;
            int vertexInCircle = gl_VertexIndex % 72;
            int segment = vertexInCircle / 2;
            int pointInSegment = vertexInCircle % 2;
            float angle = float(segment + pointInSegment) * 6.28318530718 / 36.0;
            
            float radius = data.extents.x;
            if (circle == 0) {
                localPos = vec3(cos(angle)*radius, sin(angle)*radius, 0.0);
            } else if (circle == 1) {
                localPos = vec3(cos(angle)*radius, 0.0, sin(angle)*radius);
            } else {
                localPos = vec3(0.0, cos(angle)*radius, sin(angle)*radius);
            }
        } else {
            valid = false;
        }
    }

    if (valid) {
        vec3 worldPos = center + localPos;
        gl_Position = push.viewProj * vec4(worldPos, 1.0);
    } else {
        // Send unused AABB vertices far outside frustum clip space to instantly discard them
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    }
    
    fragColor = vec3(0.0, 1.0, 0.0); // Green for BVH
}