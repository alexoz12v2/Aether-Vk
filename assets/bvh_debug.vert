#version 450

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 center_type;   // xyz = center, w = type (0=Sphere, 1=AABB/OOBB)
    vec4 extents;       // xyz = half extents or radius, w = unused
    vec4 axes_x;        // xyz = axis x
    vec4 axes_y;        // xyz = axis y
    vec4 axes_z;        // xyz = axis z
} push;

layout(location = 0) out vec3 fragColor;

void main() {
    int type = int(push.center_type.w);
    vec3 center = push.center_type.xyz;
    vec3 localPos = vec3(0.0);

    if (type == 1) {
        // AABB / OOBB (12 lines = 24 vertices)
        int lineIndices[24] = int[](
            0,1, 1,2, 2,3, 3,0, // Bottom face
            4,5, 5,6, 6,7, 7,4, // Top face
            0,4, 1,5, 2,6, 3,7  // Vertical edges
        );
        
        if (gl_VertexIndex < 24) {
            int corner = lineIndices[gl_VertexIndex];
            
            // Generate corner signs:
            // 0: -1, -1, -1
            // 1: +1, -1, -1
            // 2: +1, +1, -1
            // 3: -1, +1, -1
            // 4: -1, -1, +1
            // 5: +1, -1, +1
            // 6: +1, +1, +1
            // 7: -1, +1, +1
            vec3 signs = vec3(
                (corner == 1 || corner == 2 || corner == 5 || corner == 6) ? 1.0 : -1.0,
                (corner == 2 || corner == 3 || corner == 6 || corner == 7) ? 1.0 : -1.0,
                (corner >= 4) ? 1.0 : -1.0
            );
            
            vec3 halfExtents = push.extents.xyz;
            localPos = signs * halfExtents;
            // Transform by axes (for AABB, axes are identity)
            localPos = push.axes_x.xyz * localPos.x + push.axes_y.xyz * localPos.y + push.axes_z.xyz * localPos.z;
        }
    } else {
        // Sphere
        // 3 circles, 36 segments each = 36 * 2 vertices = 72 per circle = 216 total
        if (gl_VertexIndex < 216) {
            int circle = gl_VertexIndex / 72;
            int vertexInCircle = gl_VertexIndex % 72;
            int segment = vertexInCircle / 2;
            int pointInSegment = vertexInCircle % 2;
            float angle = float(segment + pointInSegment) * 6.28318530718 / 36.0;
            
            float radius = push.extents.x;
            if (circle == 0) {
                localPos = vec3(cos(angle)*radius, sin(angle)*radius, 0.0);
            } else if (circle == 1) {
                localPos = vec3(cos(angle)*radius, 0.0, sin(angle)*radius);
            } else {
                localPos = vec3(0.0, cos(angle)*radius, sin(angle)*radius);
            }
        }
    }

    vec3 worldPos = center + localPos;
    gl_Position = push.viewProj * vec4(worldPos, 1.0);
    
    // Wireframe colors
    fragColor = vec3(0.0, 1.0, 0.0); // Green for BVH
}