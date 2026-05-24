#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.h"

struct PushConstants {
    device MultiBvhNode* bvh;
    device DepthIndices* depth_indices;
    uint total_nodes_at_depth;
};

[[kernel]]
void motion_refit(
    constant PushConstants& pc [[buffer(0)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= pc.total_nodes_at_depth) return;

    uint node_idx = pc.depth_indices->indices[global_id + 4];
    for (uint i = 0; i < 2; ++i) {
        uint child = pc.bvh[node_idx].child_indices[i];
        if (bvh_is_leaf(pc.bvh[node_idx].metadata[i])) {
            pc.bvh[node_idx].min_x[i] = pc.bvh[child].min_x[0];
            pc.bvh[node_idx].max_x[i] = pc.bvh[child].max_x[0];
            pc.bvh[node_idx].min_y[i] = pc.bvh[child].min_y[0];
            pc.bvh[node_idx].max_y[i] = pc.bvh[child].max_y[0];
            pc.bvh[node_idx].min_z[i] = pc.bvh[child].min_z[0];
            pc.bvh[node_idx].max_z[i] = pc.bvh[child].max_z[0];
        } else {
            pc.bvh[node_idx].min_x[i] = min(pc.bvh[child].min_x[0], pc.bvh[child].min_x[1]);
            pc.bvh[node_idx].max_x[i] = max(pc.bvh[child].max_x[0], pc.bvh[child].max_x[1]);
            pc.bvh[node_idx].min_y[i] = min(pc.bvh[child].min_y[0], pc.bvh[child].min_y[1]);
            pc.bvh[node_idx].max_y[i] = max(pc.bvh[child].max_y[0], pc.bvh[child].max_y[1]);
            pc.bvh[node_idx].min_z[i] = min(pc.bvh[child].min_z[0], pc.bvh[child].min_z[1]);
            pc.bvh[node_idx].max_z[i] = max(pc.bvh[child].max_z[0], pc.bvh[child].max_z[1]);
        }
    }
}
