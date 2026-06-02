#version 450 core
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_nonuniform_qualifier : require

struct RationalBezier {
    vec4 cp0, cp1, cp2, cp3; 
};

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer SegmentArray {
    RationalBezier segments[];
};

struct Trajectory {
    SegmentArray segmentsPtr;
    vec4 color;               
    float lineWidth;          
    uint textureId;           
};

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer TrajectoryArray {
    Trajectory trajectories[];
};

struct SegmentMap {
    uint trajectoryId;
    uint localSegmentId;
    uint subdivisions; 
};

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer MapArray {
    SegmentMap maps[];
};

layout(push_constant, std430) uniform PushConstants {
    MapArray mapPtr;         
    TrajectoryArray trajPtr; 
    mat4 viewProj;
    vec2 viewportSize;
} pc;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vUV;
layout(location = 2) out flat float vLineWidth;
layout(location = 3) out flat uint vTexId;

void main() {
    uint mapIdx = gl_InstanceIndex; 
    SegmentMap map = pc.mapPtr.maps[mapIdx];
    
    uint validSubdivisions = max(map.subdivisions, 1u);
    
    uint rawStepIdx = gl_VertexIndex / 2;
    uint side = gl_VertexIndex % 2; 

    uint stepIdx = min(rawStepIdx, validSubdivisions);

    uint trajId = map.trajectoryId;

    // Critical: Copy one field at a time. Seems to be an issue of SPIRV-Cross?
    SegmentArray segArray = pc.trajPtr.trajectories[trajId].segmentsPtr;
    vec4 cp0 = segArray.segments[map.localSegmentId].cp0;
    vec4 cp1 = segArray.segments[map.localSegmentId].cp1;
    vec4 cp2 = segArray.segments[map.localSegmentId].cp2;
    vec4 cp3 = segArray.segments[map.localSegmentId].cp3;
    
    float lineWidth = pc.trajPtr.trajectories[trajId].lineWidth;
    vec4 color = pc.trajPtr.trajectories[trajId].color;
    uint textureId = pc.trajPtr.trajectories[trajId].textureId;

    float t = float(stepIdx) / float(validSubdivisions);
    float omt = 1.0 - t;

    float b0 = omt * omt * omt;
    float b1 = 3.0 * omt * omt * t;
    float b2 = 3.0 * omt * t * t;
    float b3 = t * t * t;

    float db0 = -3.0 * omt * omt;
    float db1 = 3.0 * omt * (1.0 - 3.0 * t);
    float db2 = 3.0 * t * (2.0 - 3.0 * t);
    float db3 = 3.0 * t * t;

    vec4 v0 = pc.viewProj * cp0;
    vec4 v1 = pc.viewProj * cp1;
    vec4 v2 = pc.viewProj * cp2;
    vec4 v3 = pc.viewProj * cp3;

    vec4 vClip = v0 * b0 + v1 * b1 + v2 * b2 + v3 * b3;
    vec4 dvClip = v0 * db0 + v1 * db1 + v2 * db2 + v3 * db3;

    float w2 = max(vClip.w * vClip.w, 1e-6); 
    vec2 tNdc = (dvClip.xy * vClip.w - vClip.xy * dvClip.w) / w2;
    
    vec2 tScreen = tNdc * pc.viewportSize;
    float len = length(tScreen);
    
    vec2 nScreen = (len > 1e-5) ? vec2(-tScreen.y, tScreen.x) / len : vec2(0.0, 1.0);
    float lateral = (side == 0) ? -1.0 : 1.0;

    // Minimum fixed screen-space width (e.g. 2 pixels) so it never becomes invisible
    float minThickness = lineWidth;
    
    // Physical world-space width: 0.0001 AU (approx 15,000 km) per line width unit
    float physicalWidthAu = lineWidth * 0.0001;
    
    // Project physical width to screen space. pc.viewportSize.y is used as reference scale
    float projectedThickness = (physicalWidthAu / max(abs(vClip.w), 1e-6)) * pc.viewportSize.y;
    
    // Final thickness ensures the line never shrinks below the minimum width, but grows when close
    float finalThickness = max(minThickness, projectedThickness);

    vec2 offsetNdc = nScreen * lateral * (finalThickness / pc.viewportSize);

    gl_Position = vec4(vClip.xy + offsetNdc * abs(vClip.w), vClip.z, vClip.w);

    vColor = color;
    vUV = vec2(t, lateral); 
    vLineWidth = lineWidth;
    vTexId = textureId;
}