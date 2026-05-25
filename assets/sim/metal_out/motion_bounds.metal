#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.metal"

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#ifndef PRIMITIVE_TYPE
#define PRIMITIVE_TYPE 0
#endif

struct PushConstants_motion_bounds {
    device MultiBvhNode* bvh;
    device uint* primitive_data;
    uint num_primitives;
    float dt;
    float particle_radius;
};

[[kernel]]
void motion_bounds(
    constant PushConstants_motion_bounds& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.num_primitives) return;

    if (PRIMITIVE_TYPE == 0) {
        uint base = (idx / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (idx % SUBGROUP_SIZE);
        
        float pos_x = as_type<float>(pc.primitive_data[base + 0]);
        float pos_y = as_type<float>(pc.primitive_data[base + 1 * SUBGROUP_SIZE]);
        float pos_z = as_type<float>(pc.primitive_data[base + 2 * SUBGROUP_SIZE]);
        float3 pos = float3(pos_x, pos_y, pos_z);
        
        float vel_x = as_type<float>(pc.primitive_data[base + 3 * SUBGROUP_SIZE]);
        float vel_y = as_type<float>(pc.primitive_data[base + 4 * SUBGROUP_SIZE]);
        float vel_z = as_type<float>(pc.primitive_data[base + 5 * SUBGROUP_SIZE]);
        float3 vel = float3(vel_x, vel_y, vel_z);

        float3 p1 = pos + vel * pc.dt;
        float3 min_p = min(pos, p1) - pc.particle_radius;
        float3 max_p = max(pos, p1) + pc.particle_radius;

        uint leaf_idx = (pc.num_primitives - 1) + idx;
        uint parent = pc.bvh[leaf_idx].parent_idx;
        uint is_right = (pc.bvh[parent].child_indices[1] == leaf_idx) ? 1 : 0;

        pc.bvh[parent].min_x[is_right] = min_p.x; 
        pc.bvh[parent].max_x[is_right] = max_p.x;
        pc.bvh[parent].min_y[is_right] = min_p.y; 
        pc.bvh[parent].max_y[is_right] = max_p.y;
        pc.bvh[parent].min_z[is_right] = min_p.z; 
        pc.bvh[parent].max_z[is_right] = max_p.z;
    }
}
