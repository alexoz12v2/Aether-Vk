#include <metal_stdlib>
using namespace metal;

#include "../bvh_utils.msl"

#ifdef KERNEL_LBVH_PREPASS

struct PushConstants_lbvh_prepass {
    device MultiBvhNode* bvh;
    device uint* counters;
    uint num_internal_nodes;
};

kernel void lbvh_prepass(
    constant PushConstants_lbvh_prepass& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.num_internal_nodes) return;
    
    pc.counters[idx] = 0u;
    
    if (idx == 0u) {
        pc.bvh[0].parent_idx = 0xFFFFFFFFu;
    }
}

#endif // KERNEL_LBVH_PREPASS
