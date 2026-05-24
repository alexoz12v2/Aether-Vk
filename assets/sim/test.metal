#include <metal_stdlib>
using namespace metal;
kernel void test(uint lid [[thread_position_in_threadgroup]]) {
    bool match = (lid == 0);
    uint sg_match_count = simd_sum(match ? 1 : 0);
    uint my_sg_offset = simd_prefix_exclusive_sum(match ? 1 : 0);
}
