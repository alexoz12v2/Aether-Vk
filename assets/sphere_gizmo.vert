#version 450 core
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct SphereGizmoData {
    mat4 model;
    float radius;
    float subdivisions; // E.g. 12.0 for every 30 degrees (360/30)
    vec2 _pad;
};

// Bindless BDA block
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer SphereGizmoArray {
    SphereGizmoData gizmos[];
};

layout(push_constant, scalar) uniform PushConstants {
    SphereGizmoArray gizmoPtr; // 8 bytes
    mat4 viewProj;             // 64 bytes
} push;

layout(location = 0) out vec3 fragColor;

const float PI = 3.14159265359;

void main() {
    SphereGizmoData data = push.gizmoPtr.gizmos[gl_InstanceIndex];
    int subDivs = max(4, int(data.subdivisions));
    int pointsPerRing = subDivs * 2; // Line list needs 2 vertices per segment
    int totalRingVertices = pointsPerRing * 3; // 3 Rings: XY, YZ, ZX
    
    // Axes line segments (3 axes * 2 vertices)
    int axesOffset = totalRingVertices;
    int totalAxesVertices = 6;
    
    // Arrowheads: Cone bases (subDivs * 2 lines per axis) + Cone slopes (subDivs * 2 lines per axis)
    // Actually, simpler arrowhead: just draw a few lines for the cone. Let's do 4 lines per arrowhead.
    // 4 lines = 8 vertices per arrowhead. 3 arrowheads = 24 vertices.
    int arrowheadLines = 4;
    int arrowheadVerticesPerAxis = arrowheadLines * 2;
    int totalArrowheadVertices = arrowheadVerticesPerAxis * 3;
    
    int totalExpectedVertices = totalRingVertices + totalAxesVertices + totalArrowheadVertices;
    
    vec3 localPos = vec3(0.0);
    vec3 color = vec3(0.5); // Default grey for rings
    bool valid = true;
    
    if (gl_VertexIndex < totalRingVertices) {
        // Render rings
        int ringIdx = gl_VertexIndex / pointsPerRing;
        int vertexInRing = gl_VertexIndex % pointsPerRing;
        
        // In a LINE_LIST, every 2 vertices make a segment.
        // Vertex 0 -> segment 0, start
        // Vertex 1 -> segment 0, end
        // Vertex 2 -> segment 1, start (which is the same point as segment 0, end)
        // Vertex 3 -> segment 1, end
        
        int segment = vertexInRing / 2;
        int pointInSegment = vertexInRing % 2;
        
        // The angle depends on whether it's the start or end of the segment
        float angle = float(segment + pointInSegment) * (2.0 * PI / float(subDivs));
        float r = data.radius;
        
        if (ringIdx == 0) { // XY plane (Blue ring, normal = Z)
            localPos = vec3(cos(angle)*r, sin(angle)*r, 0.0);
            color = vec3(0.2, 0.2, 0.8);
        } else if (ringIdx == 1) { // YZ plane (Red ring, normal = X)
            localPos = vec3(0.0, cos(angle)*r, sin(angle)*r);
            color = vec3(0.8, 0.2, 0.2);
        } else { // ZX plane (Green ring, normal = Y)
            localPos = vec3(sin(angle)*r, 0.0, cos(angle)*r);
            color = vec3(0.2, 0.8, 0.2);
        }
    } else if (gl_VertexIndex < axesOffset + totalAxesVertices) {
        // Render axes lines
        int axisIdx = (gl_VertexIndex - axesOffset) / 2;
        int pt = (gl_VertexIndex - axesOffset) % 2;
        float r = data.radius * 1.5; // Axes extend beyond the sphere
        
        if (axisIdx == 0) { // X Axis (Right) -> Red
            localPos = vec3(pt == 0 ? 0.0 : r, 0.0, 0.0);
            color = vec3(1.0, 0.0, 0.0);
        } else if (axisIdx == 1) { // Y Axis (Backward) -> Green
            localPos = vec3(0.0, pt == 0 ? 0.0 : r, 0.0);
            color = vec3(0.0, 1.0, 0.0);
        } else { // Z Axis (Up) -> Blue
            localPos = vec3(0.0, 0.0, pt == 0 ? 0.0 : r);
            color = vec3(0.0, 0.0, 1.0);
        }
    } else if (gl_VertexIndex < totalExpectedVertices) {
        // Render arrowheads
        int vIdx = gl_VertexIndex - axesOffset - totalAxesVertices;
        int axisIdx = vIdx / arrowheadVerticesPerAxis;
        int lineIdx = (vIdx % arrowheadVerticesPerAxis) / 2;
        int pt = vIdx % 2;
        
        float r = data.radius * 1.5;
        float headLength = data.radius * 0.2;
        float headWidth = data.radius * 0.1;
        
        float angle = float(lineIdx) * (2.0 * PI / float(arrowheadLines));
        
        vec3 tip = vec3(0.0);
        vec3 baseOffset = vec3(0.0);
        
        if (axisIdx == 0) {
            tip = vec3(r, 0.0, 0.0);
            baseOffset = vec3(-headLength, cos(angle)*headWidth, sin(angle)*headWidth);
            color = vec3(1.0, 0.0, 0.0);
        } else if (axisIdx == 1) {
            tip = vec3(0.0, r, 0.0);
            baseOffset = vec3(cos(angle)*headWidth, -headLength, sin(angle)*headWidth);
            color = vec3(0.0, 1.0, 0.0);
        } else {
            tip = vec3(0.0, 0.0, r);
            baseOffset = vec3(cos(angle)*headWidth, sin(angle)*headWidth, -headLength);
            color = vec3(0.0, 0.0, 1.0);
        }
        
        if (pt == 0) {
            localPos = tip;
        } else {
            localPos = tip + baseOffset;
        }
    } else {
        valid = false;
    }
    
    if (valid) {
        vec4 worldPos = data.model * vec4(localPos, 1.0);
        gl_Position = push.viewProj * worldPos;
        fragColor = color;
    } else {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    }
}
