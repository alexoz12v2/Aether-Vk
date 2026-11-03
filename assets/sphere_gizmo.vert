#version 450 core
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

#ifdef DEBUG_SHADERS
#extension GL_EXT_debug_printf : enable
#endif

struct SphereGizmoData {
    mat4 model;
    float radius;
    float subdivisions; // E.g. 12.0 for every 30 degrees (360/30)
    float style;        // 0 = sphere wireframe + axes, 1 = dashed polar axis (local Z, ±radius),
                        // 2 = solid jet line (local +Z, 0 → radius, in `color`)
    float dashCount;    // style 1: dashes (dash + gap) along the whole axis
    vec4 centerClip;    // P*V*center composed in f64 on the CPU; all zero = compute it here
    vec4 color;         // style 2: the line's colour; style 0: xyz the comet's velocity direction
                        // (layer-frame axes, zero = unknown), w the parts mask (PART_* bits)
};

// Bindless BDA block
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer SphereGizmoArray {
    SphereGizmoData gizmos[];
};

layout(push_constant, std430) uniform PushConstants {
    mat4 viewProj;             // 64 bytes @ offset  0
    vec3 sunDir;               // 12 bytes @ offset 64: gizmo centre → Sun, unit, layer axes, the
                               //   mesh light's f64 direction (frame::sphere_gizmo_sun_dir)
    float sunDirValid;         //  4 bytes @ offset 76: 1 = sunDir set, 0 = no Sun arrow
    SphereGizmoArray gizmoPtr; //  8 bytes @ offset 80  (total: 88)
} push;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out float outAlpha;
// x: position along the dash pattern (style 1: 0 at the south end, dashCount + ½ at the north end;
// style 0 axis shafts: AXIS_DASHES_PER_RADIUS periods per gizmo radius from the centre),
// y: 1 = dashed line (the fragment shader cuts the gaps), 0 = solid
layout(location = 2) out vec2 outDash;

const float PI = 3.14159265359;

// style 0 parts (scene::SPHERE_GIZMO_PART_*)
const uint PART_GRID      = 1u;
const uint PART_BODY_AXES = 2u;
const uint PART_SPIN_AXIS = 4u;
const uint PART_SUN       = 8u;
const uint PART_VELOCITY  = 16u;
const vec3 VELOCITY_COLOR = vec3(1.0, 0.35, 1.0);
const vec3 SPIN_COLOR     = vec3(0.45, 0.68, 1.0); // lighter than pure blue: readable on black

// Style 0 axes (frame::SPHERE_GIZMO_AXIS_VERTICES = 5 × AXIS_VERTS): each axis starts at the limb
// (1 radius) and is a 3-strand shaft (6 vertices), an 8-spoke arrowhead (16) and the 8-segment
// ring of the arrowhead's base (16), which stays a readable circle when the axis points at the
// viewer. An axis pointing away from the viewer (its far part is behind the nucleus) is dimmed
// and its shaft dashed.
const int   AXIS_VERTS             = 38;
const int   AXIS_SHAFT_VERTS       = 6;
const int   AXIS_SPOKE_VERTS       = 16;
const int   AXIS_HEAD_SEGMENTS     = 8;
const float AXIS_BODY_LENGTH       = 1.5;   // radii, X / Y / spin
const float AXIS_FAR_LENGTH        = 2.1;   // radii, Sun / velocity
const float AXIS_HEAD_LENGTH       = 0.22;  // radii
const float AXIS_HEAD_WIDTH        = 0.10;  // radii
const float AXIS_STRAND_OFFSET     = 0.008; // radii between the shaft's strands
const float AXIS_DASHES_PER_RADIUS = 6.0;
const float AXIS_AWAY_DIM          = 0.45;

// Whether axis `axisIdx` (0 X, 1 Y, 2 Z = spin, 3 Sun, 4 velocity) is drawn.
bool axisShown(int axisIdx, uint parts, vec3 velocity) {
    if (axisIdx <= 1) return (parts & PART_BODY_AXES) != 0u;
    if (axisIdx == 2) return (parts & PART_SPIN_AXIS) != 0u;
    if (axisIdx == 3) return (parts & PART_SUN) != 0u && push.sunDirValid > 0.5;
    return (parts & PART_VELOCITY) != 0u && dot(velocity, velocity) > 0.25;
}

// Unit direction from the gizmo centre towards the viewer, layer axes. Perspective (the clip w row
// of viewProj is non-zero): towards the eye at the RTE origin. Orthographic (w row zero): the
// reverse-Z depth row, which grows towards the viewer (orthographic_vk_reverse_z).
vec3 towardsViewer(vec3 center) {
    vec3 wRow = vec3(push.viewProj[0][3], push.viewProj[1][3], push.viewProj[2][3]);
    vec3 zRow = vec3(push.viewProj[0][2], push.viewProj[1][2], push.viewProj[2][2]);
    bool perspective = dot(wRow, wRow) > 0.0;
    vec3 v = perspective ? -center : zRow;
    if (dot(v, v) <= 0.0) v = perspective ? -wRow : zRow;
    return normalize(v);
}

void main() {
    mat4 model = push.gizmoPtr.gizmos[gl_InstanceIndex].model;
    outDash = vec2(0.0);

    if (push.gizmoPtr.gizmos[gl_InstanceIndex].style > 1.5) {
        // ── Jet direction line: local +Z from the centre to +radius, solid, own colour ───────
        if (gl_VertexIndex > 1) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            outAlpha = 0.0;
            fragColor = vec3(0.0);
            return;
        }
        vec4 lineCenterClip = push.gizmoPtr.gizmos[gl_InstanceIndex].centerClip;
        if (lineCenterClip == vec4(0.0)) lineCenterClip = push.viewProj * vec4(model[3].xyz, 1.0);
        float len = push.gizmoPtr.gizmos[gl_InstanceIndex].radius;
        vec3 lineLocal = vec3(0.0, 0.0, gl_VertexIndex == 0 ? 0.0 : len);
        gl_Position = lineCenterClip + push.viewProj * vec4(mat3(model) * lineLocal, 0.0);
        // Reverse-Z: pulled forward like the body axes
        gl_Position.z = min(gl_Position.z + 0.001 * gl_Position.w, gl_Position.w);
        outAlpha = 1.0;
        fragColor = push.gizmoPtr.gizmos[gl_InstanceIndex].color.rgb;
        return;
    }

    if (push.gizmoPtr.gizmos[gl_InstanceIndex].style > 0.5) {
        // ── Dashed polar axis: one line, local Z from -radius (south) to +radius (north) ─────
        // The same draw issues the sphere's vertex count: every vertex past the first two is
        // pushed outside clip space.
        if (gl_VertexIndex > 1) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            outAlpha = 0.0;
            fragColor = vec3(0.0);
            return;
        }
        vec4 axisCenterClip = push.gizmoPtr.gizmos[gl_InstanceIndex].centerClip;
        if (axisCenterClip == vec4(0.0)) axisCenterClip = push.viewProj * vec4(model[3].xyz, 1.0);
        float halfLength = push.gizmoPtr.gizmos[gl_InstanceIndex].radius;
        float t = float(gl_VertexIndex); // 0 = south end, 1 = north end
        vec3 axisLocal = vec3(0.0, 0.0, mix(-halfLength, halfLength, t));
        gl_Position = axisCenterClip + push.viewProj * vec4(mat3(model) * axisLocal, 0.0);
        // Reverse-Z: pull forward like the comet gizmo's axes so it wins over its own globe
        gl_Position.z = min(gl_Position.z + 0.001 * gl_Position.w, gl_Position.w);
        outAlpha = 1.0;
        // N + ½ periods: the line starts and ends on a dash, so both poles are marked
        outDash = vec2(t * (push.gizmoPtr.gizmos[gl_InstanceIndex].dashCount + 0.5), 1.0);
        fragColor = vec3(0.85, 0.9, 1.0);
        return;
    }

    vec4 centerClip = push.gizmoPtr.gizmos[gl_InstanceIndex].centerClip;
    if (centerClip == vec4(0.0)) centerClip = push.viewProj * vec4(model[3].xyz, 1.0);

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

    vec4 partsSlot = push.gizmoPtr.gizmos[gl_InstanceIndex].color;
    uint parts = uint(partsSlot.w + 0.5);
    vec3 velocityDir = partsSlot.xyz;

    // 5 axes: X(red), Y(green), spin(light blue), Sun(yellow), velocity(magenta)
    int axesOffset = totalSphereVertices;
    int totalExpectedVertices = axesOffset + 5 * AXIS_VERTS;

    vec3 localPos = vec3(0.0);
    vec3 color = vec3(1.0); // Default white for the sphere
    bool valid = true;
    bool isAxis = false; // Axes receive Z-bias; sphere grid does not

    if (gl_VertexIndex < totalSphereVertices) {
        // ── UV Sphere wireframe ──────────────────────────────────────────────
        valid = (parts & PART_GRID) != 0u;
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
    } else if (gl_VertexIndex < totalExpectedVertices) {
        // ── Axes: shaft from the limb, arrowhead spokes, arrowhead base ring ───────────────
        isAxis = true;
        int v = gl_VertexIndex - axesOffset;
        int axisIdx = v / AXIS_VERTS;
        int k = v % AXIS_VERTS;
        int pt = k % 2;
        valid = axisShown(axisIdx, parts, velocityDir);
        float r = push.gizmoPtr.gizmos[gl_InstanceIndex].radius;
        mat3 toLocal = transpose(mat3(model)); // scale 1 (scene_conversion): inverse = transpose

        vec3 dir;
        float len = AXIS_BODY_LENGTH;
        if (axisIdx == 0) {
            dir = vec3(1.0, 0.0, 0.0);
            color = vec3(1.0, 0.0, 0.0);
        } else if (axisIdx == 1) {
            dir = vec3(0.0, 1.0, 0.0);
            color = vec3(0.0, 1.0, 0.0);
        } else if (axisIdx == 2) {
            dir = vec3(0.0, 0.0, 1.0);
            color = SPIN_COLOR;
        } else if (axisIdx == 3) {
            // the mesh light's f64 direction (no sunPos − centre in f32 here)
            dir = normalize(toLocal * push.sunDir);
            len = AXIS_FAR_LENGTH;
            color = vec3(1.0, 1.0, 0.0);
        } else {
            dir = normalize(toLocal * velocityDir);
            len = AXIS_FAR_LENGTH;
            color = VELOCITY_COLOR;
        }

        // pointing away from the viewer: behind the limb, dimmed, the shaft dashed
        vec3 viewer = toLocal * towardsViewer(model[3].xyz);
        bool away = dot(dir, viewer) < 0.0;

        // basis across the axis; `side` faces the screen (perpendicular to the view) when it can
        vec3 side = cross(dir, viewer);
        if (dot(side, side) < 1e-8) side = cross(dir, abs(dir.z) < 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0));
        side = normalize(side);
        vec3 side2 = cross(dir, side);

        float tipS = len * r;
        float baseS = (len - AXIS_HEAD_LENGTH) * r;
        float headW = AXIS_HEAD_WIDTH * r;
        float s = 0.0; // distance from the centre along the axis
        if (k < AXIS_SHAFT_VERTS) {
            // three parallel strands from the limb to the arrowhead's base: a thicker shaft
            float strand = float(k / 2) - 1.0;
            s = pt == 0 ? r : baseS;
            localPos = dir * s + side * (strand * AXIS_STRAND_OFFSET * r);
            outDash = away ? vec2(s / r * AXIS_DASHES_PER_RADIUS, 1.0) : vec2(0.0);
        } else if (k < AXIS_SHAFT_VERTS + AXIS_SPOKE_VERTS) {
            // spokes from the tip to the base ring: a dense (near-filled) cone
            int j = (k - AXIS_SHAFT_VERTS) / 2;
            float a = float(j) * (2.0 * PI / float(AXIS_HEAD_SEGMENTS));
            localPos = pt == 0 ? dir * tipS : dir * baseS + (side * cos(a) + side2 * sin(a)) * headW;
        } else {
            // the base ring: a circle around the axis, readable when it points at the viewer
            int j = (k - AXIS_SHAFT_VERTS - AXIS_SPOKE_VERTS) / 2 + pt;
            float a = float(j) * (2.0 * PI / float(AXIS_HEAD_SEGMENTS));
            localPos = dir * baseS + (side * cos(a) + side2 * sin(a)) * headW;
        }
        if (away) color *= AXIS_AWAY_DIM;
    } else {
        valid = false;
    }

    if (valid) {
        // Project localized vector natively inside clip space
        vec4 localClip = push.viewProj * vec4(mat3(model) * localPos, 0.0);
        gl_Position = centerClip + localClip;

        // Front-hemisphere discard signal.
        // Use the true vector from the vertex to the camera to properly cull when zoomed in close.
        // vec3 worldNormal = mat3(model) * localPos;  // outward world-space direction
        // vec3 trueCamDir = -(model[3].xyz + mat3(model) * localPos); // true vector to camera
        
        // Fix: Use the vector from the GIZMO CENTER to the camera, rather than from the vertex.
        // This guarantees we always render a perfect 180-degree half-sphere, preventing the 
        // "missing outward parts" caused by true perspective horizon shrinking.
        vec3 worldNormal = mat3(model) * localPos;  
        vec3 trueCamDir = -model[3].xyz;
        
        // Calculate distance squared from camera (origin) to the gizmo center (model[3].xyz)
        float distToCenterSq = dot(model[3].xyz, model[3].xyz);
        float radiusSq = push.gizmoPtr.gizmos[gl_InstanceIndex].radius * push.gizmoPtr.gizmos[gl_InstanceIndex].radius;
        bool isInside = distToCenterSq < radiusSq;
        
        // If the camera is inside the gizmo, do not discard any faces.
        outAlpha = (isAxis || isInside) ? 1.0 : (dot(worldNormal, trueCamDir) > 0.0 ? 1.0 : 0.0);

        if (isAxis) {
            // Reverse-Z: ADD a small margin so it renders on top.
            gl_Position.z += 0.001 * gl_Position.w;
        } else {
            // Reverse-Z: add tiny margin to pull forward and win z-fighting with the solid mesh.
            gl_Position.z += 0.0001 * gl_Position.w;
        }
        
        // Prevent depth bias from pushing fragments outside the Reverse-Z near clip plane (1.0).
        gl_Position.z = min(gl_Position.z, gl_Position.w);

        fragColor = color;
        
#ifdef DEBUG_SHADERS
        float trueDist = length(model[3].xyz + mat3(model) * localPos);
        if (gl_VertexIndex == 0) {
            debugPrintfEXT("Sphere Gizmo Vertex 0: dist=%.3f, camDir=(%.2f, %.2f, %.2f)", 
                           trueDist, trueCamDir.x, trueCamDir.y, trueCamDir.z);
        }
#endif
    } else {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // cull degenerate vertices outside clip space
        outAlpha = 0.0;
    }
}