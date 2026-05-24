#include <metal_stdlib>
using namespace metal;

#include "../debug_utils.h"
#include "../bvh_utils.h"

struct PushConstants {
    device uint* raw_scene_pairs;
    device uint* out_rb_rb;
    device uint* out_rb_ps;
    device uint* out_rb_lca;
    device uint* internal_pairs;
};

[[kernel]]
void bp_clear(
    constant PushConstants& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    *pc.raw_scene_pairs = 0u;
    *pc.out_rb_rb = 0u;
    *pc.out_rb_ps = 0u;
    *pc.out_rb_lca = 0u;
    *pc.internal_pairs = 0u;
}