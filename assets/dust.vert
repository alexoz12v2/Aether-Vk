// @assets/dust.vert
//
// Note: We are setting `gl_PointSize` to values bigger than 1.0. As such, we need
// the `largePoints` device feature

#version 450 core

#extension GL_EXT_buffer_reference2      : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_GOOGLE_include_directive   : require

#include "debug_utils.glsl"
#include "bvh_utils.glsl"

uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// 128 bytes
layout(push_constant, std430) uniform PushConstants {
    ParticleChunkBuffer globalParticleBuffer;
    ParticlePageTable   particlePageTable;

    mat4  viewProj;
    vec4  streamColor;
    uint  chunkOffset;
    uint  currentTime;  // In 1/300 seconds
    float maxTtl;       // limit for age fading (in 1/300 s)
    float macroScale;   // base point sprite pixel size multiplier
    float microRadius;  // physical size of the dust specks (eg 0.15)
    uint  numSpots;     // how many spots per blob (eg 12 or 16)
    float dispersionRate;
} pc;

layout(location = 0) out vec4 v_Color;
layout(location = 1) out flat uint v_Seed;
layout(location = 2) out flat float v_PointSize;

void main() {
    uint idx = gl_VertexIndex;

    if (idx > pc.particlePageTable.particleCount) {
        gl_Position = vec4(0.0);
        gl_PointSize = 0.0;
        return;
    }

    // extract components from vertex index OS page table style
    uint logicalChunk = idx >> 8;   // idx / 256
    uint lane         = idx & 255;  // idx % 256
    uint vecIdx       = lane >> 2;  // lane / 4
    uint compIdx      = lane & 3;   // lane % 4

    uint physicalIdx = pc.particlePageTable.chunks[pc.chunkOffset + logicalChunk];

    float px = pc.globalParticleBuffer.chunks[physicalIdx].positionX[vecIdx][compIdx];
    float py = pc.globalParticleBuffer.chunks[physicalIdx].positionY[vecIdx][compIdx];
    float pz = pc.globalParticleBuffer.chunks[physicalIdx].positionZ[vecIdx][compIdx];
    uint spawn = pc.globalParticleBuffer.chunks[physicalIdx].spawnTime[vecIdx][compIdx];

    vec4 worldPos = vec4(px, py, pz, 1.0);
    gl_Position = pc.viewProj * worldPos;

    // time to live fade out
    float age = float(pc.currentTime - spawn);
    float fade = 1.0 - clamp(age / pc.maxTtl, 0.0, 1.0);

    // scale macro-particle cluster based on distance
    // scale macro-particle as it ages so internal spots drift apart
    float pSize = 0;
    if (gl_Position.w > 0.0) {
        float expandedScale = pc.macroScale + (age * pc.dispersionRate);
        pSize = max(1.0, expandedScale / gl_Position.w);
    } else {
        pSize = 0.0;
    }
    gl_PointSize = pSize;
    v_PointSize = pSize; // send it down the pipeline

    v_Color = vec4(pc.streamColor.rgb, pc.streamColor.a * fade);

    // temporaly stable seed based on spawnTime so particles look different when recycled
    v_Seed = spawn ^ pcg(idx * 1973u);
}