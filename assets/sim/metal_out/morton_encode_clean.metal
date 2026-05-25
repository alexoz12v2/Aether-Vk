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
// @assets/sim/morton_encode.comp
//
// Calculates a 30-bit Morton Code for each particle to be used for radix sorting
//
// Target: MSL Metal 3.0

struct PushConstants_morton_encode {
    MortonArray morton_out;
    ParticleData particles;
    uint num_particles;
    float3 scene_min;
    float3 scene_max;
};

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
inline uint morton_encode_expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline uint morton_encode_morton3D(float3 norm_pos) {
    norm_pos = clamp(norm_pos, 0.0f, 1.0f);
    uint x = uint(norm_pos.x * 1023.0f);
    uint y = uint(norm_pos.y * 1023.0f);
    uint z = uint(norm_pos.z * 1023.0f);
    return (morton_encode_expandBits(x) << 2) | (morton_encode_expandBits(y) << 1) | morton_encode_expandBits(z);
}

[[kernel]]
void morton_encode(constant PushConstants_morton_encode& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.num_particles) return;

    // AOSOA unpacking matching your particle structure
    uint block_idx = idx / SUBGROUP_SIZE;
    uint local_idx = idx % SUBGROUP_SIZE;
    uint base = block_idx * (10 * SUBGROUP_SIZE) + local_idx;

    float3 pos = float3(
        P_READ(pc.particles, base + 0 * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 1 * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 2 * SUBGROUP_SIZE)
    );

    // Normalize relative to scene bounds
    float3 extents = pc.scene_max - pc.scene_min;
    float3 norm_pos = (pos - pc.scene_min) / max(extents, float3(1e-5f));

    uint m_code = morton_encode_morton3D(norm_pos);

    pc.morton_out.entries[idx] = uint2(m_code, idx);
}
