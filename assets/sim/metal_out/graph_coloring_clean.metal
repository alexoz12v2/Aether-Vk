#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
using namespace metal;

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

// Shared macros
#define P_READ(addr, offset) as_type<float>((addr)[(offset)])
#define P_WRITE(addr, offset, val) ((addr)[(offset)] = as_type<uint>(val))

// Typedefs
typedef device uint* ParticleData;

// Structs
struct MultiBvhNode {
    float min_x[SUBGROUP_SIZE]; float max_x[SUBGROUP_SIZE];
    float min_y[SUBGROUP_SIZE]; float max_y[SUBGROUP_SIZE];
    float min_z[SUBGROUP_SIZE]; float max_z[SUBGROUP_SIZE];
    uint  child_indices[SUBGROUP_SIZE]; uint metadata[SUBGROUP_SIZE];
    float masses[SUBGROUP_SIZE];
    float com_x[SUBGROUP_SIZE]; float com_y[SUBGROUP_SIZE]; float com_z[SUBGROUP_SIZE];
    uint  particle_start[SUBGROUP_SIZE]; uint particle_count[SUBGROUP_SIZE];
    uint2 valid_mask;
    uint  parent_idx;
    uint  pad;
    uint  permutations[8][SUBGROUP_SIZE];
};

struct DepthIndices {
    uint indices[1];
};

struct Wrench { uint force_x; uint force_y; uint force_z; uint torque_x; uint torque_y; uint torque_z; };

// Atomics
inline void AtomicAddFloatBDA(device uint* addr, uint offset, float val) {
    device atomic_uint* a = (device atomic_uint*)(addr + offset);
    uint e = atomic_load_explicit(a, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(a, &e, as_type<uint>(as_type<float>(e) + val), memory_order_relaxed, memory_order_relaxed));
}

// Bvh helpers
inline bool bvh_node_is_valid(uint2 mask, uint lane_id) {
    if (lane_id < 32) return (mask.x & (1u << lane_id)) != 0;
    return (mask.y & (1u << (lane_id - 32))) != 0;
}

inline bool bvh_is_leaf(uint metadata) {
    return (metadata & 0x80000000) != 0;
}

inline uint bvh_leaf_count(uint metadata) {
    return metadata & 0x7FFFFFFF;
}
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
