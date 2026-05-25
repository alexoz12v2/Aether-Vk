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
#if defined(KERNEL_stream_compact)

struct PushConstants_stream_compact {
    device void* sparse_in;
    device void* packed_out;
    uint total_elements;
};

[[kernel]]
void stream_compact(
    constant PushConstants_stream_compact& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
#ifdef DEBUG_SHADERS
    if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0 && thread_position_in_grid.z == 0) {
        // MSL doesn't typically support debugPrintfEXT natively, but we can log or ignore
    }
#endif

    uint id = thread_position_in_grid.x;
    
    device uint* sparse_in_count = (device uint*)pc.sparse_in;
    uint in_count = *sparse_in_count;

    if (id == 0) {
        device uint* packed_out_dispatch = (device uint*)pc.packed_out;
        packed_out_dispatch[3] = in_count; // count at offset 12
        uint blocks = (in_count + 127) / 128;
        packed_out_dispatch[0] = blocks;   // dispatch_x
        packed_out_dispatch[1] = 1;        // dispatch_y
        packed_out_dispatch[2] = 1;        // dispatch_z
    }

    if (id < in_count) {
        device SparseCollisionData* sparse_pairs = (device SparseCollisionData*)((device char*)pc.sparse_in + 16);
        device PackedPair* packed_pairs = (device PackedPair*)((device char*)pc.packed_out + 16);
        
        SparseCollisionData in_data = sparse_pairs[id];
        
        packed_pairs[id].a.entity_id = in_data.entity_a;
        packed_pairs[id].a.primitive_index = in_data.prim_a;
        packed_pairs[id].b.entity_id = in_data.entity_b;
        packed_pairs[id].b.primitive_index = in_data.prim_b;
        packed_pairs[id].toi = in_data.toi;
        packed_pairs[id].contact_normal = in_data.contact_normal;
        packed_pairs[id].contact_point = in_data.contact_point;
        packed_pairs[id].penetration_depth = in_data.penetration_depth;
    }
}

#endif // KERNEL_stream_compact
