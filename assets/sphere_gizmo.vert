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
    int latSegments = max(4, int(data.subdivisions));
    int lonSegments = max(4, int(data.subdivisions));
    
    // A UV sphere wireframe rendered as a LINE_LIST.
    // For each latitude segment (except the poles), we draw a horizontal ring segment.
    // For each longitude segment, we draw a vertical meridian segment.
    // Number of horizontal lines = (latSegments - 1) * lonSegments
    // Number of vertical lines = latSegments * lonSegments
    // Total lines = lonSegments * (2 * latSegments - 1)
    // Vertices per line = 2
    int totalSphereVertices = lonSegments * (2 * latSegments - 1) * 2;
    
    // Axes line segments (3 axes * 2 vertices)
    int axesOffset = totalSphereVertices;
    int totalAxesVertices = 6;
    
    // Arrowheads: 4 lines = 8 vertices per arrowhead. 3 arrowheads = 24 vertices.
    int arrowheadLines = 4;
    int arrowheadVerticesPerAxis = arrowheadLines * 2;
    int totalArrowheadVertices = arrowheadVerticesPerAxis * 3;
    
    int totalExpectedVertices = totalSphereVertices + totalAxesVertices + totalArrowheadVertices;
    
    vec3 localPos = vec3(0.0);
    vec3 color = vec3(1.0); // Default white for the sphere
    bool valid = true;
    
    if (gl_VertexIndex < totalSphereVertices) {
        // Render UV Sphere wireframe
        int lineIdx = gl_VertexIndex / 2;
        int isEndVertex = gl_VertexIndex % 2;
        
        int numHorizontalLines = (latSegments - 1) * lonSegments;
        
        float r = data.radius;
        
        if (lineIdx < numHorizontalLines) {
            // Horizontal ring segments
            int latIdx = (lineIdx / lonSegments) + 1; // +1 to skip the pole
            int lonIdx = lineIdx % lonSegments;
            
            float theta = float(latIdx) * PI / float(latSegments);
            float phiStart = float(lonIdx) * 2.0 * PI / float(lonSegments);
            float phiEnd = float(lonIdx + 1) * 2.0 * PI / float(lonSegments);
            
            float phi = (isEndVertex == 0) ? phiStart : phiEnd;
            
            localPos = vec3(cos(phi) * sin(theta) * r, sin(phi) * sin(theta) * r, cos(theta) * r);
        } else {
            // Vertical meridian segments
            int vertLineIdx = lineIdx - numHorizontalLines;
            int latIdx = vertLineIdx / lonSegments;
            int lonIdx = vertLineIdx % lonSegments;
            
            float thetaStart = float(latIdx) * PI / float(latSegments);
            float thetaEnd = float(latIdx + 1) * PI / float(latSegments);
            float phi = float(lonIdx) * 2.0 * PI / float(lonSegments);
            
            float theta = (isEndVertex == 0) ? thetaStart : thetaEnd;
            
            localPos = vec3(cos(phi) * sin(theta) * r, sin(phi) * sin(theta) * r, cos(theta) * r);
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
