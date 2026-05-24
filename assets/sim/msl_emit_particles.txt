#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.msl"

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#ifdef KERNEL_emit_particles
struct PushConstants {
    device uint* particles;
    device uint* candidates;
    device MultiBvhNode* bvh;
    device atomic_uint* counter;
    uint root_index;
    uint num_candidates;
    uint2 pad;
    float3 sun_pos;
};
#endif

#ifndef INTERSECT_RAY_AABB_DEFINED
#define INTERSECT_RAY_AABB_DEFINED
bool intersectRayAABB(float3 rO, float3 rD, float3 invD, float3 mi, float3 mx, float max_t) {
    float3 t0 = (mi - rO) * invD;
    float3 t1 = (mx - rO) * invD;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float tnear = max(max(tmin.x, tmin.y), tmin.z);
    float tfar = min(min(tmax.x, tmax.y), tmax.z);
    return tnear <= tfar && tfar > 0.0f && tnear < max_t;
}
#endif

[[kernel]]
void emit_particles(
    uint3 gl_GlobalInvocationID [[thread_position_in_grid]],
    constant PushConstants& pc [[buffer(0)]]
) {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.num_candidates) return;
    
    uint stride = 10 * SUBGROUP_SIZE;
    uint base = (gid / SUBGROUP_SIZE) * stride + (gid % SUBGROUP_SIZE);

    float pos_x = as_type<float>(pc.candidates[base]);
    float pos_y = as_type<float>(pc.candidates[base + SUBGROUP_SIZE]);
    float pos_z = as_type<float>(pc.candidates[base + 2 * SUBGROUP_SIZE]);
    float3 pos = float3(pos_x, pos_y, pos_z);
    
    float3 dir = pc.sun_pos - pos;
    float dist = length(dir);
    if (dist < 1e-5f) return;
    dir /= dist;
    float3 invDir = 1.0f / dir;

    bool occluded = false;
    uint stack[64];
    int stackPtr = 0;
    if (pc.root_index != 0xFFFFFFFFu) stack[stackPtr++] = pc.root_index;

    while(stackPtr > 0 && !occluded) {
        uint node = stack[--stackPtr];
        for (uint i = 0; i < SUBGROUP_SIZE; ++i) {
            if (!bvh_node_is_valid(pc.bvh[node].valid_mask, i)) continue;
            
            float3 mn = float3(pc.bvh[node].min_x[i], pc.bvh[node].min_y[i], pc.bvh[node].min_z[i]);
            float3 mx = float3(pc.bvh[node].max_x[i], pc.bvh[node].max_y[i], pc.bvh[node].max_z[i]);

            if (intersectRayAABB(pos + dir * 0.1f, dir, invDir, mn, mx, dist)) {
                if (bvh_is_leaf(pc.bvh[node].metadata[i])) { 
                    occluded = true; 
                    break; 
                }
                else if (bvh_get_index(pc.bvh[node].metadata[i]) != 0xFFFFFFFFu) {
                    stack[stackPtr++] = bvh_get_index(pc.bvh[node].metadata[i]);
                }
            }
        }
    }

    if (!occluded) {
        uint out_idx = atomic_fetch_add_explicit(&pc.counter[0], 1, memory_order_relaxed);
        uint out_base = (out_idx / SUBGROUP_SIZE) * stride + (out_idx % SUBGROUP_SIZE);
        for (int i = 0; i < 10; ++i) {
            pc.particles[out_base + i * SUBGROUP_SIZE] = pc.candidates[base + i * SUBGROUP_SIZE];
        }
    }
}
