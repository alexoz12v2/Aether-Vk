#version 450 core
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

struct SphereGizmoData {
    mat4 model;
    float radius;
    float subdivisions; // E.g. 12.0 for every 30 degrees (360/30)
    vec2 _pad;
};

// Bindless BDA block
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer SphereGizmoArray {
    SphereGizmoData gizmos[];
};

layout(push_constant, std430) uniform PushConstants {
    mat4 viewProj;             // 64 bytes @ offset  0
    vec3 sunPos;               // 12 bytes @ offset 64
    float _pad;                //  4 bytes @ offset 76
    SphereGizmoArray gizmoPtr; //  8 bytes @ offset 80  (total: 88)
} push;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out float outAlpha;

const float PI = 3.14159265359;

void main() {
    mat4 model = push.gizmoPtr.gizmos[gl_InstanceIndex].model;

    vec4 centerClip = push.viewProj * vec4(model[3].xyz, 1.0);

    // Use clip-space w as the distance proxy for screen-size LOD.
    // Perspective:   w = eye-space depth (orbit distance). Rotation-invariant: computed
    //                from a single dot product, not from length(xyz) which suffers from
    //                f32 component-wise cancellation as the camera rotates around the comet.
    // Orthographic:  w = 1.0, so screenSize = P11 * radius, which scales
    //                correctly with orthographic zoom and is distance-independent.
    // NOTE: viewProj is P * V. The second row of viewProj is P11 * V_row1.
    // Since V_row1 is a unit vector, we can extract the true P11 scale by taking the magnitude of the second row.
    float distForSize = max(0.0001, abs(centerClip.w));
    float p11 = length(vec3(push.viewProj[0][1], push.viewProj[1][1], push.viewProj[2][1]));
    float screenSize = abs(p11 * push.gizmoPtr.gizmos[gl_InstanceIndex].radius / distForSize);

    int lodSubdivs = 8;
    if (screenSize > 0.15) {
        lodSubdivs = 36;
    } else if (screenSize >= 0.05) {
        lodSubdivs = 18;
    }

    int latSegments = max(4, lodSubdivs);
    int lonSegments = max(4, lodSubdivs);

    // A UV sphere wireframe rendered as a LINE_LIST.
    // For each latitude segment (except the poles), we draw a horizontal ring segment.
    // For each longitude segment, we draw a vertical meridian segment.
    // Number of horizontal lines = (latSegments - 1) * lonSegments
    // Number of vertical lines = latSegments * lonSegments
    // Total lines = lonSegments * (2 * latSegments - 1)
    // Vertices per line = 2
    int totalSphereVertices = lonSegments * (2 * latSegments - 1) * 2;

    // 4 axes: X(red), Y(green), Z(blue), Sun(yellow). 2 vertices each.
    int axesOffset = totalSphereVertices;
    int totalAxesVertices = 8; // 4 axes * 2 vertices

    // Arrowheads: 4 lines = 8 vertices per arrowhead. 4 arrowheads = 32 vertices.
    int arrowheadLines = 4;
    int arrowheadVerticesPerAxis = arrowheadLines * 2;
    int totalArrowheadVertices = arrowheadVerticesPerAxis * 4; // 4 axes

    int totalExpectedVertices = axesOffset + totalAxesVertices + totalArrowheadVertices;

    vec3 localPos = vec3(0.0);
    vec3 color = vec3(1.0); // Default white for the sphere
    bool valid = true;
    bool isAxis = false; // Axes receive Z-bias; sphere grid does not

    if (gl_VertexIndex < totalSphereVertices) {
        // ── UV Sphere wireframe ──────────────────────────────────────────────
        int lineIdx = gl_VertexIndex / 2;
        int isEndVertex = gl_VertexIndex % 2;

        int numHorizontalLines = (latSegments - 1) * lonSegments;

        float r = push.gizmoPtr.gizmos[gl_InstanceIndex].radius;

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
        // ── 4 axes (X=red, Y=green, Z=blue, Sun=yellow) ─────────────────────
        isAxis = true;
        int axisIdx = (gl_VertexIndex - axesOffset) / 2;
        int pt = (gl_VertexIndex - axesOffset) % 2;
        float r = push.gizmoPtr.gizmos[gl_InstanceIndex].radius * 1.5; // Axes extend beyond the sphere

        if (axisIdx == 0) { // X Axis (Right) -> Red
            localPos = vec3(pt == 0 ? 0.0 : r, 0.0, 0.0);
            color = vec3(1.0, 0.0, 0.0);
        } else if (axisIdx == 1) { // Y Axis (Backward) -> Green
            localPos = vec3(0.0, pt == 0 ? 0.0 : r, 0.0);
            color = vec3(0.0, 1.0, 0.0);
        } else if (axisIdx == 2) { // Z Axis (Up) -> Blue
            localPos = vec3(0.0, 0.0, pt == 0 ? 0.0 : r);
            color = vec3(0.0, 0.0, 1.0);
        } else { // Axis 3: Sun direction -> Yellow (1.4x length)
            float sunR = r * 1.4;
            // Transform global sun direction into the model's local space.
            // scene_conversion.rs enforces scale=1 for sphere gizmos, so mat3(model)
            // is orthonormal -> inverse == transpose (cheaper, no singularity risk).
            vec3 worldSunDir = normalize(push.sunPos - model[3].xyz);
            vec3 localSunDir = normalize(transpose(mat3(model)) * worldSunDir);
            localPos = pt == 0 ? vec3(0.0) : localSunDir * sunR;
            color = vec3(1.0, 1.0, 0.0);
        }
    } else if (gl_VertexIndex < totalExpectedVertices) {
        // ── Arrowheads (4 axes) ──────────────────────────────────────────────
        isAxis = true;
        int vIdx = gl_VertexIndex - axesOffset - totalAxesVertices;
        int axisIdx = vIdx / arrowheadVerticesPerAxis;
        int lineIdx = (vIdx % arrowheadVerticesPerAxis) / 2;
        int pt = vIdx % 2;

        float r = push.gizmoPtr.gizmos[gl_InstanceIndex].radius * 1.5;
        float headLength = push.gizmoPtr.gizmos[gl_InstanceIndex].radius * 0.2;
        float headWidth = push.gizmoPtr.gizmos[gl_InstanceIndex].radius * 0.1;

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
        } else if (axisIdx == 2) {
            tip = vec3(0.0, 0.0, r);
            baseOffset = vec3(cos(angle)*headWidth, sin(angle)*headWidth, -headLength);
            color = vec3(0.0, 0.0, 1.0);
        } else { // Axis 3: Sun arrowhead -> Yellow
            float sunR = r * 1.4;
            vec3 worldSunDir = normalize(push.sunPos - model[3].xyz);
            vec3 localSunDir = normalize(transpose(mat3(model)) * worldSunDir);
            tip = localSunDir * sunR;

            // Gram-Schmidt: build orthonormal basis perpendicular to localSunDir
            vec3 up     = abs(localSunDir.z) < 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
            vec3 right  = normalize(cross(up, localSunDir));
            vec3 trueUp = cross(localSunDir, right);

            vec3 radial = right * cos(angle) * headWidth + trueUp * sin(angle) * headWidth;
            baseOffset  = -localSunDir * headLength + radial;
            color = vec3(1.0, 1.0, 0.0);
        }

        localPos = (pt == 0) ? tip : tip + baseOffset;
    } else {
        valid = false;
    }

    if (valid) {
        // Project localized vector natively inside clip space
        vec4 localClip = push.viewProj * vec4(mat3(model) * localPos, 0.0);
        gl_Position = centerClip + localClip;

        // Front-hemisphere discard signal.
        // View-direction-independent: a sphere surface point is front-facing if its
        // outward world-space normal has a positive component toward the camera.
        // Camera is at RTE origin; gizmo center is at model[3].xyz.
        // camDir = direction from gizmo center toward camera = -model[3].xyz.
        // Using un-normalized vectors is fine — we only need the sign of the dot product.
        vec3 worldNormal = mat3(model) * localPos;  // outward world-space direction
        vec3 camDir = -model[3].xyz;                // gizmo center → camera (RTE origin)
        outAlpha = isAxis ? 1.0 : (dot(worldNormal, camDir) > 0.0 ? 1.0 : 0.0);

        if (isAxis) {
            // Reverse-Z: SUBTRACT a small margin so axes sit just inside the near-plane clip range.
            // The stencil-based two-pass draw calls (OverMesh / Elsewhere) determine
            // which pixels actually render on top of the comet, not z-bias.
            gl_Position.z -= 0.001 * gl_Position.w;
        } else {
            // Reverse-Z: subtract tiny margin to avoid z-fighting with the solid mesh.
            gl_Position.z -= 0.0001 * gl_Position.w;
        }

        fragColor = color;
    } else {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // cull degenerate vertices outside clip space
        outAlpha = 0.0;
    }
}