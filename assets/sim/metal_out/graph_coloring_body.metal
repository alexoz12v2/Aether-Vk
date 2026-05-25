#include <metal_stdlib>
using namespace metal;

struct ColliderId {
    uint entity_id;
    uint primitive_index;
};

struct PackedPair {
    ColliderId a;
    ColliderId b;
    float toi;
    float4 contact_normal;
    float4 contact_point;
    float penetration_depth;
};

struct PackedCollisions {
    uint dispatch_x;
    uint dispatch_y;
    uint dispatch_z;
    uint count;
    PackedPair pairs[1];
};

struct PushConstants_graph_coloring {
    device PackedCollisions* collisions;
    device uint* colors;
    device uint* weights;
    uint total_pairs;
};

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

[[kernel]]
void graph_coloring(
    constant PushConstants_graph_coloring& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.total_pairs) return;

    // NVIDIA Parallel ILU factorization graph coloring adapted for Vulkan 1.1 SPV1.4 Memory Model
    // We color the contact pairs (edges) so that independent contacts can be solved in parallel.

    // 1. Initialize weights
    pc.weights[idx] = hash(idx + 1);
    pc.colors[idx] = 0; // 0 means uncolored

    // Memory barrier to ensure all weights are visible
    threadgroup_barrier(mem_flags::mem_device);

    // 2. Luby's algorithm for independent sets
    bool colored = false;
    uint my_color = 1;
    uint my_weight = pc.weights[idx];
    
    PackedPair my_pair = pc.collisions->pairs[idx];
    uint my_a = my_pair.a.primitive_index;
    uint my_b = my_pair.b.primitive_index;

    for (int iter = 0; iter < 10; ++iter) {
        if (!colored) {
            bool is_max = true;
            
            // Check adjacent contacts (contacts sharing body A or body B)
            for (uint j = 0; j < pc.total_pairs; ++j) {
                if (idx == j) continue;
                PackedPair other_pair = pc.collisions->pairs[j];
                uint other_a = other_pair.a.primitive_index;
                uint other_b = other_pair.b.primitive_index;
                
                if (my_a == other_a || my_a == other_b || my_b == other_a || my_b == other_b) {
                    uint other_color = pc.colors[j];
                    if (other_color == 0 || other_color == my_color) {
                        uint other_weight = pc.weights[j];
                        if (other_weight > my_weight || (other_weight == my_weight && j > idx)) {
                            is_max = false;
                            break;
                        }
                    }
                }
            }
            
            if (is_max) {
                pc.colors[idx] = my_color;
                colored = true;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        if (!colored) {
            my_color++;
        }
    }
}
